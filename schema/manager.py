"""
Manages database schema for TIA-1.2, supporting SQL Server and S3.
Builds schema dictionary with tables, columns, keys, views, and indexes.
Preserves TIA-1.1 functionality, fixes performance and schema issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based paths (e.g., BIKE-STORES/BikeStores/schema_cache/)
- Async queries and cache operations for <5s processing
- Logging to logs/DISPLAY_NAME_app.log at application root
- Non-hardcoding with display_name and db_name from app-configs/db_configurations.json
- S3 schema support (e.g., stores.csv)
Addresses Scenarios 1 (S3 support), 3 (connection handling), 5 (performance), and 6 (schema validation).
"""
import os
import json
import logging
import asyncio
import pyodbc
import boto3
import pandas as pd
from typing import Dict, Optional, Any, List
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from botocore.client import BaseClient
from config.logging_setup import setup_logging
from config.utils import load_db_config

class SchemaManager:
    """Manages schema information for SQL Server and S3 in TIA-1.2."""
    
    def __init__(self, config_name: str = None, schemas: List[str] = None, tables: List[str] = None):
        """
        Initialize SchemaManager with cache and logging, reading configs from app-configs/db_configurations.json.

        Args:
            config_name (str, optional): Configuration name (e.g., 'bikestores') or display_name (e.g., 'BIKE-STORES').
            schemas (List[str], optional): Schemas to include.
            tables (List[str], optional): Tables to include (schema.table or S3 table).
        """
        # Load database config
        config = load_db_config(config_name)
        self.display_name = config.get("display_name", os.getenv("DISPLAY_NAME", "BIKE-STORES"))
        self.db_name = config.get("database", os.getenv("DB_NAME", "BikeStores"))

        # Initialize logging
        setup_logging(self.display_name, self.db_name)
        self.logger = logging.getLogger("schema")

        # Setup cache paths
        self.schema_cache_dir = Path(self.display_name, self.db_name, "schema_cache")
        self.schema_file = self.schema_cache_dir / "schema.json"
        self.metadata_file = self.schema_cache_dir / "metadata.json"
        self.schema_cache_dir.mkdir(parents=True, exist_ok=True)

        # System schemas
        self.system_schemas = [
            'dbo', 'sys', 'information_schema',
            'db_accessadmin', 'db_backupoperator', 'db_datareader', 'db_datawriter',
            'db_ddladmin', 'db_denydatareader', 'db_denydatawriter', 'db_owner',
            'db_securityadmin', 'guest'
        ]

        # Validate tables
        self.schemas = schemas or []
        self.tables = tables or []
        valid_tables = [t for t in self.tables if '.' in t or t in [tbl for s in self.tables if s.startswith('s3.') or '.' not in s]]
        if len(valid_tables) != len(self.tables):
            self.logger.warning(f"Removed invalid table formats: {[t for t in self.tables if t not in valid_tables]}")
            self.tables = valid_tables

        self.logger.info(f"Initialized SchemaManager for {self.display_name}/{self.db_name}, schemas={self.schemas}, tables={self.tables}")

    @contextmanager
    def _get_cursor(self, connection: pyodbc.Connection):
        """Context manager for SQL Server cursor handling."""
        cursor = None
        try:
            cursor = connection.cursor()
            yield cursor
        except pyodbc.Error as e:
            self.logger.error(f"Cursor creation failed: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected cursor error: {e}", exc_info=True)
            raise
        finally:
            if cursor:
                try:
                    cursor.close()
                    self.logger.debug("Cursor closed")
                except pyodbc.Error as e:
                    self.logger.error(f"Error closing cursor: {e}", exc_info=True)

    async def needs_refresh(self, connection: Any) -> bool:
        """
        Check if schema cache needs refreshing asynchronously (Scenarios 3, 5).

        Args:
            connection: SQL Server connection or S3 client.

        Returns:
            bool: True if refresh needed, False otherwise.
        """
        try:
            if not connection:
                self.logger.error("No valid connection for refresh check")
                return True

            async with asyncio.Lock():
                if not self.schema_file.exists():
                    self.logger.debug("Schema file not found, needs refresh")
                    return True

                metadata = await self._load_cache_metadata()
                last_updated = metadata.get('last_updated')
                cached_version = metadata.get('schema_version', '1.0')
                cached_schemas = metadata.get('schemas', [])
                cached_tables = metadata.get('tables', [])

                if cached_schemas != self.schemas or cached_tables != self.tables:
                    self.logger.debug("Schema or table configuration changed, needs refresh")
                    return True

            if isinstance(connection, pyodbc.Connection):
                return await self._needs_refresh_sql(connection)
            elif isinstance(connection, BaseClient):
                return await self._needs_refresh_s3(connection)
            else:
                self.logger.error("Unsupported connection type")
                return True
        except Exception as e:
            self.logger.warning(f"Error checking schema refresh: {e}", exc_info=True)
            return True

    async def _needs_refresh_sql(self, connection: pyodbc.Connection) -> bool:
        """Check SQL Server schema refresh needs."""
        try:
            with self._get_cursor(connection) as cursor:
                if self.tables:
                    table_count = len(self.tables)
                    column_count = 0
                    index_count = 0
                    for table in self.tables:
                        schema, table_name = table.split('.') if '.' in table else ('s3', table)
                        if schema != 's3':
                            result = await asyncio.to_thread(cursor.execute, """
                                SELECT COUNT(*)
                                FROM information_schema.columns
                                WHERE table_schema = ? AND table_name = ? AND table_catalog = ?
                            """, (schema, table_name, self.db_name))
                            column_count += (await asyncio.to_thread(cursor.fetchone))[0]
                            result = await asyncio.to_thread(cursor.execute, """
                                SELECT COUNT(*)
                                FROM sys.indexes
                                WHERE object_id = OBJECT_ID(?)
                            """, (f"{schema}.{table_name}",))
                            index_count += (await asyncio.to_thread(cursor.fetchone))[0]
                else:
                    schemas = self.schemas if self.schemas else []
                    if not schemas:
                        result = await asyncio.to_thread(cursor.execute, """
                            SELECT schema_name
                            FROM information_schema.schemata
                            WHERE catalog_name = ?
                            AND schema_name NOT IN ({})
                        """.format(','.join('?' * len(self.system_schemas))), (self.db_name, *self.system_schemas))
                        schemas = [row[0] for row in (await asyncio.to_thread(cursor.fetchall))]

                    table_count = 0
                    column_count = 0
                    index_count = 0
                    for schema in schemas:
                        result = await asyncio.to_thread(cursor.execute, """
                            SELECT COUNT(*)
                            FROM information_schema.tables
                            WHERE table_schema = ? AND table_catalog = ? AND table_type = 'BASE TABLE'
                        """, (schema, self.db_name))
                        table_count += (await asyncio.to_thread(cursor.fetchone))[0]
                        result = await asyncio.to_thread(cursor.execute, """
                            SELECT COUNT(*)
                            FROM information_schema.columns
                            WHERE table_schema = ? AND table_catalog = ?
                        """, (schema, self.db_name))
                        column_count += (await asyncio.to_thread(cursor.fetchone))[0]
                        result = await asyncio.to_thread(cursor.execute, """
                            SELECT COUNT(*)
                            FROM sys.indexes i
                            JOIN sys.tables t ON i.object_id = t.object_id
                            JOIN sys.schemas s ON t.schema_id = s.schema_id
                            WHERE s.name = ?
                        """, (schema,))
                        index_count += (await asyncio.to_thread(cursor.fetchone))[0]

                async with asyncio.Lock():
                    with self.schema_file.open() as f:
                        cached_schema = json.load(f)
                    cached_table_count = sum(len(tables) for tables in cached_schema.get('tables', {}).values())
                    cached_column_count = sum(
                        len(cols) for schema in cached_schema.get('columns', {}).values()
                        for cols in schema.values()
                    )
                    cached_index_count = sum(
                        len(cached_schema.get('indexes', {}).get(schema, {}).get(table, []))
                        for schema in cached_schema.get('tables', {})
                        for table in cached_schema['tables'][schema]
                    )

                needs_refresh = (
                    table_count != cached_table_count or
                    column_count != cached_column_count or
                    index_count != cached_index_count or
                    cached_version < '1.2'
                )

                if last_updated:
                    last_updated_dt = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S")
                    if (datetime.now() - last_updated_dt).days > 7:
                        self.logger.debug("Schema cache older than 7 days, needs refresh")
                        needs_refresh = True

                self.logger.debug(f"SQL schema refresh needed: {needs_refresh}")
                return needs_refresh
        except Exception as e:
            self.logger.warning(f"Error checking SQL schema refresh: {e}", exc_info=True)
            return True

    async def _needs_refresh_s3(self, s3_client: BaseClient) -> bool:
        """Check S3 schema refresh needs (Scenario 1)."""
        try:
            bucket = load_db_config('bikestores-csv').get('bucket', '')
            response = await asyncio.to_thread(s3_client.list_objects_v2, Bucket=bucket, Prefix='', Delimiter='/')
            s3_tables = [obj['Key'].split('.')[0] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
            table_count = len(s3_tables)
            column_count = 0
            for table in s3_tables:
                obj = await asyncio.to_thread(s3_client.get_object, Bucket=bucket, Key=f"{table}.csv")
                df = pd.read_csv(obj['Body'], nrows=0)
                column_count += len(df.columns)

            async with asyncio.Lock():
                with self.schema_file.open() as f:
                    cached_schema = json.load(f)
                cached_table_count = len(cached_schema.get('tables', {}).get('s3', []))
                cached_column_count = sum(
                    len(cols) for cols in cached_schema.get('columns', {}).get('s3', {}).values()
                )

            needs_refresh = table_count != cached_table_count or column_count != cached_column_count
            self.logger.debug(f"S3 schema refresh needed: {needs_refresh}")
            return needs_refresh
        except Exception as e:
            self.logger.warning(f"Error checking S3 schema refresh: {e}", exc_info=True)
            return True

    async def build_data_dict(self, connection: Any) -> Dict:
        """Alias for build_schema_dictionary (Scenario 5)."""
        self.logger.debug("Building schema dictionary")
        return await self.build_schema_dictionary(connection)

    async def build_schema_dictionary(self, connection: Any) -> Dict:
        """
        Build schema dictionary for SQL Server or S3 with retry logic (Scenarios 1, 3, 5, 6).

        Args:
            connection: SQL Server connection or S3 client.

        Returns:
            Dict: Schema dictionary with tables, columns, keys, views, indexes.
        """
        try:
            self.logger.debug(f"Building schema with connection type: {type(connection).__name__ if connection else 'None'}")
            if not connection:
                cached_schema = await self.load_from_cache()
                if cached_schema:
                    self.logger.info("Using cached schema due to invalid connection")
                    return cached_schema
                self.logger.warning("Invalid connection and no cached schema, returning empty schema")
                return {}

            schema_dict = {
                'tables': {},
                'columns': {},
                'primary_keys': {},
                'foreign_keys': {},
                'views': {},
                'indexes': {}
            }

            if isinstance(connection, pyodbc.Connection):
                return await self._build_sql_schema(connection, schema_dict)
            elif isinstance(connection, BaseClient):
                return await self._build_s3_schema(connection, schema_dict)
            else:
                self.logger.error("Unsupported connection type")
                cached_schema = await self.load_from_cache()
                if cached_schema:
                    self.logger.info("Using cached schema due to unsupported connection")
                    return cached_schema
                self.logger.warning("Returning empty schema for unsupported connection")
                return schema_dict
        except Exception as e:
            self.logger.error(f"Error building schema dictionary: {e}", exc_info=True)
            cached_schema = await self.load_from_cache()
            if cached_schema:
                self.logger.info("Using cached schema as fallback")
                return cached_schema
            self.logger.warning("No cached schema available, returning empty schema")
            return {}

    async def _build_sql_schema(self, connection: pyodbc.Connection, schema_dict: Dict) -> Dict:
        """Build schema dictionary for SQL Server (Scenarios 3, 5, 6)."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                with self._get_cursor(connection) as cursor:
                    self.logger.debug(f"Attempt {attempt}: Fetching SQL schemas")
                    result = await asyncio.to_thread(cursor.execute, """
                        SELECT schema_name
                        FROM information_schema.schemata
                        WHERE catalog_name = ?
                    """, (self.db_name,))
                    available_schemas = [row[0] for row in (await asyncio.to_thread(cursor.fetchall))]
                    if self.schemas:
                        missing_schemas = [s for s in self.schemas if s not in available_schemas]
                        if missing_schemas:
                            self.logger.warning(f"Schemas not found in {self.db_name}: {missing_schemas}")
                            self.schemas = [s for s in available_schemas if s not in self.system_schemas]
                            if not self.schemas:
                                self.logger.error(f"No valid schemas in {self.db_name}")
                                cached_schema = await self.load_from_cache()
                                if cached_schema:
                                    self.logger.info("Using cached schema")
                                    return cached_schema
                                return schema_dict

                    schemas = await self._get_schemas(cursor)
                    for schema in schemas:
                        schema_dict['tables'][schema] = []
                        schema_dict['columns'][schema] = {}
                        schema_dict['primary_keys'][schema] = {}
                        schema_dict['foreign_keys'][schema] = {}
                        schema_dict['views'][schema] = []
                        schema_dict['indexes'][schema] = {}

                        tables = await self._get_tables(cursor, schema)
                        schema_dict['tables'][schema] = tables
                        self.logger.debug(f"Found {len(tables)} tables in {schema}: {tables}")

                        views = await self._get_views(cursor, schema)
                        schema_dict['views'][schema] = views
                        self.logger.debug(f"Found {len(views)} views in {schema}: {views}")

                        if tables:
                            schema_dict['columns'][schema] = await self._get_columns(cursor, schema, tables)
                            schema_dict['primary_keys'][schema] = await self._get_primary_keys(cursor, schema, tables)
                            schema_dict['foreign_keys'][schema] = await self._get_foreign_keys(cursor, schema, tables)
                            schema_dict['indexes'][schema] = await self._get_indexes(cursor, schema, tables)

                        for view in views:
                            schema_dict['columns'][schema][view] = await self._get_view_columns(cursor, schema, view)

                    if await self.validate_schema(schema_dict):
                        await self.save_schema(str(self.schema_file), schema_dict)
                        self.logger.debug("SQL schema dictionary built successfully")
                        return schema_dict
                    else:
                        self.logger.warning(f"Schema validation failed on attempt {attempt}")
                        if attempt == max_attempts:
                            self.logger.error("Schema validation failed after all attempts")
                            break
            except pyodbc.Error as e:
                self.logger.warning(f"Query attempt {attempt} failed: {e}", exc_info=True)
                if attempt < max_attempts:
                    await asyncio.sleep(0.1 * (2 ** (attempt - 1)))
                else:
                    self.logger.error(f"Failed to build SQL schema after {max_attempts} attempts: {e}")
                    break
            except Exception as e:
                self.logger.warning(f"Unexpected error on attempt {attempt}: {e}", exc_info=True)
                if attempt < max_attempts:
                    await asyncio.sleep(0.1 * (2 ** (attempt - 1)))
                else:
                    self.logger.error(f"Failed to build SQL schema after {max_attempts} attempts: {e}")
                    break

        cached_schema = await self.load_from_cache()
        if cached_schema:
            self.logger.info("Using cached schema as fallback")
            return cached_schema
        return schema_dict

    async def _build_s3_schema(self, s3_client: BaseClient, schema_dict: Dict) -> Dict:
        """Build schema dictionary for S3 (Scenario 1)."""
        try:
            schema = 's3'
            schema_dict['tables'][schema] = []
            schema_dict['columns'][schema] = {}
            schema_dict['primary_keys'][schema] = {}
            schema_dict['foreign_keys'][schema] = {}
            schema_dict['views'][schema] = []
            schema_dict['indexes'][schema] = {}

            bucket = load_db_config('bikestores-csv').get('bucket', '')
            response = await asyncio.to_thread(s3_client.list_objects_v2, Bucket=bucket, Prefix='', Delimiter='/')
            tables = [obj['Key'].split('.')[0] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
            if self.tables:
                tables = [t for t in tables if t in [tbl.split('.')[1] if '.' in tbl else tbl for tbl in self.tables]]
            schema_dict['tables'][schema] = tables
            self.logger.debug(f"Found {len(tables)} S3 tables: {tables}")

            for table in tables:
                obj = await asyncio.to_thread(s3_client.get_object, Bucket=bucket, Key=f"{table}.csv")
                df = pd.read_csv(obj['Body'], nrows=0)
                columns = {col: {'type': str(df[col].dtype), 'nullable': True} for col in df.columns}
                schema_dict['columns'][schema][table] = columns
                self.logger.debug(f"Loaded {len(columns)} columns for S3 table {table}")

            if await self.validate_schema(schema_dict):
                await self.save_schema(str(self.schema_file), schema_dict)
                self.logger.debug("S3 schema dictionary built successfully")
                return schema_dict
            else:
                self.logger.error("S3 schema validation failed")
                cached_schema = await self.load_from_cache()
                if cached_schema:
                    self.logger.info("Using cached schema as fallback")
                    return cached_schema
                return schema_dict
        except Exception as e:
            self.logger.error(f"Error building S3 schema: {e}", exc_info=True)
            cached_schema = await self.load_from_cache()
            if cached_schema:
                self.logger.info("Using cached schema as fallback")
                return cached_schema
            return schema_dict

    async def _get_schemas(self, cursor: pyodbc.Cursor) -> List[str]:
        """Get schemas for SQL Server (Scenario 5)."""
        if self.tables:
            schemas = list(set(table.split('.')[0] for table in self.tables if '.' in table))
            self.logger.debug(f"Using schemas from tables: {schemas}")
        elif self.schemas:
            schemas = self.schemas
            self.logger.debug(f"Using specified schemas: {schemas}")
        else:
            result = await asyncio.to_thread(cursor.execute, """
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE catalog_name = ?
                AND schema_name NOT IN ({})
            """.format(','.join('?' * len(self.system_schemas))), (self.db_name, *self.system_schemas))
            schemas = [row[0] for row in (await asyncio.to_thread(cursor.fetchall))]
            self.logger.debug(f"Using all non-system schemas: {schemas}")
        return schemas

    async def _get_tables(self, cursor: pyodbc.Cursor, schema: str) -> List[str]:
        """Get tables for a schema (Scenario 5)."""
        if self.tables:
            tables = [table.split('.')[1] for table in self.tables if table.split('.')[0] == schema]
            valid_tables = []
            for table in tables:
                result = await asyncio.to_thread(cursor.execute, """
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = ? AND table_name = ? AND table_catalog = ?
                    AND table_type = 'BASE TABLE'
                """, (schema, table, self.db_name))
                if (await asyncio.to_thread(cursor.fetchone))[0] > 0:
                    valid_tables.append(table)
                else:
                    self.logger.warning(f"Table not found: {schema}.{table}")
            return valid_tables
        result = await asyncio.to_thread(cursor.execute, """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = ? 
            AND table_catalog = ?
            AND table_type = 'BASE TABLE'
        """, (schema, self.db_name))
        return [row[0] for row in (await asyncio.to_thread(cursor.fetchall))]

    async def _get_views(self, cursor: pyodbc.Cursor, schema: str) -> List[str]:
        """Get views for a schema (Scenario 5)."""
        result = await asyncio.to_thread(cursor.execute, """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = ? 
            AND table_catalog = ?
            AND table_type = 'VIEW'
        """, (schema, self.db_name))
        return [row[0] for row in (await asyncio.to_thread(cursor.fetchall))]

    async def _get_columns(self, cursor: pyodbc.Cursor, schema: str, tables: List[str]) -> Dict:
        """Get columns for tables (Scenario 5)."""
        columns_by_table = {}
        table_placeholders = ','.join('?' * len(tables))
        result = await asyncio.to_thread(cursor.execute, """
            SELECT table_name, column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_schema = ? 
            AND table_name IN ({})
            AND table_catalog = ?
        """.format(table_placeholders), (schema, *tables, self.db_name))
        for row in (await asyncio.to_thread(cursor.fetchall)):
            table = row[0]
            columns_by_table.setdefault(table, {})[row[1]] = {
                'type': row[0],
                'nullable': row[3] == 'YES',
                'default': row[4]
            }
        return {table: columns_by_table.get(table, {}) for table in tables}

    async def _get_primary_keys(self, cursor: pyodbc.Cursor, schema: str, tables: List[str]) -> Dict:
        """Get primary keys for tables (Scenario 5)."""
        primary_keys = {}
        for table in tables:
            result = await asyncio.to_thread(cursor.execute, """
                SELECT column_name 
                FROM information_schema.key_column_usage 
                WHERE table_schema = ? 
                AND table_name = ? 
                AND table_catalog = ? 
                AND constraint_name LIKE 'PK%'
            """, (schema, table, self.db_name))
            pks = [row[0] for row in (await asyncio.to_thread(cursor.fetchall))]
            if pks:
                primary_keys[table] = pks
        return primary_keys

    async def _get_foreign_keys(self, cursor: pyodbc.Cursor, schema: str, tables: List[str]) -> Dict:
        """Get foreign keys for tables (Scenario 5)."""
        foreign_keys = {}
        for table in tables:
            result = await asyncio.to_thread(cursor.execute, """
                SELECT 
                    kcu.column_name,
                    ccu.table_schema AS referenced_schema,
                    ccu.table_name AS referenced_table,
                    ccu.column_name AS referenced_column
                FROM information_schema.key_column_usage kcu
                JOIN information_schema.constraint_column_usage ccu
                ON kcu.constraint_name = ccu.constraint_name
                WHERE kcu.table_schema = ?
                AND kcu.table_name = ?
                AND kcu.table_catalog = ?
                AND kcu.constraint_name LIKE 'FK%'
            """, (schema, table, self.db_name))
            fks = [
                {
                    'column': row[0],
                    'referenced_schema': row[1],
                    'referenced_table': row[2],
                    'referenced_column': row[3]
                }
                for row in (await asyncio.to_thread(cursor.fetchall))
            ]
            if fks:
                foreign_keys[table] = fks
        return foreign_keys

    async def _get_indexes(self, cursor: pyodbc.Cursor, schema: str, tables: List[str]) -> Dict:
        """Get indexes for tables (Scenario 5)."""
        indexes = {}
        for table in tables:
            result = await asyncio.to_thread(cursor.execute, """
                SELECT i.name
                FROM sys.indexes i
                JOIN sys.tables t ON i.object_id = t.object_id
                JOIN sys.schemas s ON t.schema_id = s.schema_id
                WHERE s.name = ? AND t.name = ?
            """, (schema, table))
            idxs = [row[0] for row in (await asyncio.to_thread(cursor.fetchall))]
            if idxs:
                indexes[table] = idxs
        return indexes

    async def _get_view_columns(self, cursor: pyodbc.Cursor, schema: str, view: str) -> Dict:
        """Get columns for a view (Scenario 5)."""
        result = await asyncio.to_thread(cursor.execute, """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = ? 
            AND table_name = ? 
            AND table_catalog = ?
        """, (schema, view, self.db_name))
        return {
            row[0]: {
                'type': row[1],
                'nullable': row[2] == 'YES'
            }
            for row in (await asyncio.to_thread(cursor.fetchall))
        }

    async def validate_schema(self, schema_dict: Dict) -> bool:
        """
        Validate schema dictionary for completeness (Scenario 6).

        Args:
            schema_dict (Dict): Schema dictionary to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            if not schema_dict or not isinstance(schema_dict, dict):
                self.logger.error("Invalid schema dictionary")
                return False

            required_keys = {'tables', 'columns', 'primary_keys', 'foreign_keys', 'views', 'indexes'}
            if not required_keys.issubset(schema_dict.keys()):
                missing = required_keys - set(schema_dict.keys())
                self.logger.error(f"Missing schema keys: {', '.join(missing)}")
                return False

            has_tables = False
            for schema in schema_dict['tables']:
                if schema_dict['tables'][schema]:
                    has_tables = True
                    for table in schema_dict['tables'][schema]:
                        if table not in schema_dict['columns'].get(schema, {}):
                            self.logger.error(f"No columns defined for table: {schema}.{table}")
                            return False
                        if not schema_dict['columns'][schema][table]:
                            self.logger.error(f"Empty columns for table: {schema}.{table}")
                            return False
                        if schema != 's3' and table in schema_dict['primary_keys'].get(schema, {}) and not schema_dict['primary_keys'][schema][table]:
                            self.logger.warning(f"Empty primary keys for table: {schema}.{table}")

            if not has_tables:
                self.logger.warning("No tables found in any schema")
                return False

            self.logger.debug("Schema validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating schema: {e}", exc_info=True)
            return False

    async def load_from_cache(self) -> Dict:
        """
        Load schema from cache asynchronously (Scenario 3).

        Returns:
            Dict: Cached schema dictionary or empty dict if invalid.
        """
        try:
            async with asyncio.Lock():
                if self.schema_file.exists():
                    with self.schema_file.open() as f:
                        schema_dict = json.load(f)
                    if await self.validate_schema(schema_dict):
                        self.logger.debug(f"Loaded schema from {self.schema_file}")
                        return schema_dict
                    self.logger.warning(f"Cached schema failed validation: {self.schema_file}")
                else:
                    self.logger.warning(f"Schema cache file not found: {self.schema_file}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading schema cache: {e}", exc_info=True)
            return {}

    async def save_schema(self, schema_path: str, schema_dict: Dict = None):
        """
        Save schema to file asynchronously (Scenario 5).

        Args:
            schema_path (str): Path to save schema.
            schema_dict (Dict, optional): Schema dictionary to save.
        """
        try:
            if not schema_dict:
                self.logger.error("Cannot save empty schema dictionary")
                return

            if not await self.validate_schema(schema_dict):
                self.logger.error("Schema validation failed, not saving")
                return

            async with asyncio.Lock():
                with open(schema_path, 'w') as f:
                    json.dump(schema_dict, f, indent=2)
            self.logger.debug(f"Saved schema to {schema_path}")

            await self._update_cache_metadata()
        except Exception as e:
            self.logger.error(f"Error saving schema: {e}", exc_info=True)

    async def _load_cache_metadata(self) -> Dict:
        """Load cache metadata asynchronously (Scenario 5)."""
        try:
            async with asyncio.Lock():
                if self.metadata_file.exists():
                    with self.metadata_file.open() as f:
                        metadata = json.load(f)
                    self.logger.debug(f"Loaded metadata from {self.metadata_file}")
                    return metadata
            return {'last_updated': None, 'schema_version': '1.2', 'schemas': [], 'tables': []}
        except Exception as e:
            self.logger.error(f"Error loading cache metadata: {e}", exc_info=True)
            return {'last_updated': None, 'schema_version': '1.2', 'schemas': [], 'tables': []}

    async def _update_cache_metadata(self):
        """Update cache metadata asynchronously (Scenario 5)."""
        try:
            metadata = {
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'schema_version': '1.2',
                'db_name': self.db_name,
                'schemas': self.schemas,
                'tables': self.tables
            }
            async with asyncio.Lock():
                with self.metadata_file.open('w') as f:
                    json.dump(metadata, f, indent=2)
            self.logger.debug(f"Updated metadata in {self.metadata_file}")
        except Exception as e:
            self.logger.error(f"Error updating cache metadata: {e}", exc_info=True)

    async def get_table_metadata(self, schema: str, table: str) -> Dict:
        """
        Retrieve metadata for a specific table (Scenario 6).

        Args:
            schema (str): Schema name (e.g., sales, s3).
            table (str): Table name.

        Returns:
            Dict: Metadata with columns, primary keys, foreign keys, indexes.
        """
        try:
            schema_dict = await self.load_from_cache()
            if not schema_dict:
                self.logger.warning(f"No cached schema for {schema}.{table}")
                return {}

            metadata = {
                'columns': schema_dict['columns'].get(schema, {}).get(table, {}),
                'primary_keys': schema_dict['primary_keys'].get(schema, {}).get(table, []),
                'foreign_keys': schema_dict['foreign_keys'].get(schema, {}).get(table, []),
                'indexes': schema_dict['indexes'].get(schema, {}).get(table, [])
            }
            self.logger.debug(f"Retrieved metadata for {schema}.{table}")
            return metadata
        except Exception as e:
            self.logger.error(f"Error retrieving table metadata for {schema}.{table}: {e}", exc_info=True)
            return {}

    async def validate_cache_consistency(self, connection: Any) -> bool:
        """
        Validate cache consistency with database/S3 (Scenarios 3, 5).

        Args:
            connection: SQL Server connection or S3 client.

        Returns:
            bool: True if consistent, False otherwise.
        """
        try:
            cached_schema = await self.load_from_cache()
            if not cached_schema:
                self.logger.warning("No cached schema to validate")
                return False

            live_schema = await self.build_schema_dictionary(connection)
            if not live_schema:
                self.logger.error("Failed to build live schema for validation")
                return False

            for schema in cached_schema['tables']:
                if schema not in live_schema['tables']:
                    self.logger.error(f"Schema {schema} missing in live database")
                    return False
                for table in cached_schema['tables'][schema]:
                    if table not in live_schema['tables'][schema]:
                        self.logger.error(f"Table {schema}.{table} missing in live database")
                        return False
                    cached_columns = set(cached_schema['columns'][schema][table].keys())
                    live_columns = set(live_schema['columns'][schema][table].keys())
                    if not cached_columns.issubset(live_columns):
                        self.logger.error(f"Column mismatch for {schema}.{table}: missing {cached_columns - live_columns}")
                        return False

            self.logger.debug("Cache consistency validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating cache consistency: {e}", exc_info=True)
            return False