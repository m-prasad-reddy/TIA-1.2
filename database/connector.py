"""
Database connection manager for TIA-1.2, handling SQL Server and S3 connections.
Provides async operations for query execution, schema retrieval, and S3 data access.
Implements TIA-1.1 functionality with TIA-1.2 enhancements:
- DISPLAY_NAME-based logging to logs/DISPLAY_NAME_app.log
- Async SQL Server (aioodbc) and S3 (aiobotocore) operations for <5s processing
- Non-hardcoding with display_name and db_name from app-configs/db_configurations.json
- Retry logic for connection reliability
- S3 schema inference from CSV files
Addresses Scenarios 1 (S3 support), 3 (connection reliability), and 5 (performance).
"""
import os
import asyncio
import logging
import pandas as pd
import aioodbc
import pyodbc
import aiobotocore.session
import io
from botocore.exceptions import ClientError
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.logging_setup import setup_logging
from config.utils import load_db_config

class DatabaseConnection:
    """
    Manages SQL Server and S3 connections for TIA-1.2 (Scenarios 1, 3, 5).
    """
    def __init__(self, config_name: str):
        """
        Initialize DatabaseConnection with credentials from app-configs/db_configurations.json.

        Args:
            config_name (str): Configuration name (e.g., 'bikestores') or display_name (e.g., 'BIKE-STORES').
        """
        config = load_db_config(config_name)
        self.config_name = config_name.lower()
        self.config_type = config.get("type", "SQL Server")
        self.display_name = config.get("display_name", os.getenv("DISPLAY_NAME", "BIKE-STORES"))
        self.db_name = config.get("database", os.getenv("DB_NAME", "BikeStores"))

        if self.config_type.lower() == "s3":
            self.sql_credentials = {
                "host": "",
                "user": "",
                "password": "",
                "database": self.db_name,
                "driver": config.get("driver", os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server"))
            }
        else:
            self.sql_credentials = {
                "host": config.get("server", os.getenv("SQL_HOST", "")),
                "user": config.get("username", os.getenv("SQL_USER", "")),
                "password": config.get("password", os.getenv("SQL_PASSWORD", "")),
                "database": self.db_name,
                "driver": config.get("driver", os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server"))
            }

        self.s3_credentials = {
            "access_key": config.get("access_key", os.getenv("AWS_ACCESS_KEY_ID", "")),
            "secret_key": config.get("secret_key", os.getenv("AWS_SECRET_ACCESS_KEY", "")),
            "bucket": config.get("bucket_name", os.getenv("AWS_S3_BUCKET", "")),
            "endpoint_url": config.get("s3_endpoint", os.getenv("AWS_S3_ENDPOINT", None))
        }

        setup_logging(self.display_name, self.db_name)
        self.logger = logging.getLogger("database_connector")
        self.sql_pool = None
        self.s3_session = None
        self.s3_client = None
        self.logger.debug(f"Config type: {self.config_type}, Credentials: SQL={self.sql_credentials}, S3={self.s3_credentials}")
        self.logger.info(f"Initialized DatabaseConnection for {self.display_name}/{self.db_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((pyodbc.OperationalError, ConnectionError)),
        before_sleep=lambda retry_state: logging.getLogger("database_connector").warning(
            f"Retrying SQL connection attempt {retry_state.attempt_number}"
        )
    )
    async def connect(self) -> None:
        """
        Establish SQL Server and S3 connections with retries (Scenario 3).

        Skips SQL for S3-only configs (type: s3) or incomplete SQL credentials.
        Skips S3 for SQL Server configs without S3 credentials.

        Raises:
            RuntimeError: If all required connections fail.
        """
        connection_success = False
        self.logger.debug("Starting connection setup")

        is_s3_only = self.config_type.lower() == "s3"

        if not is_s3_only and self.sql_credentials["host"] and self.sql_credentials["password"] and self.sql_credentials["user"]:
            try:
                dsn = (
                    f"DRIVER={{{self.sql_credentials['driver']}}};"
                    f"SERVER={self.sql_credentials['host']};"
                    f"DATABASE={self.sql_credentials['database']};"
                    f"UID={self.sql_credentials['user']};"
                    f"PWD={self.sql_credentials['password']}"
                )
                self.sql_pool = await aioodbc.create_pool(dsn=dsn, minsize=1, maxsize=5)
                self.logger.info("SQL Server connection pool established")
                connection_success = True
            except Exception as e:
                self.logger.error(f"Failed to establish SQL Server connection: {e}", exc_info=True)
        else:
            self.logger.info("Skipping SQL Server connection: S3-only config or incomplete credentials")

        if (is_s3_only or self.config_type.lower() == "sql server") and self.s3_credentials["access_key"] and self.s3_credentials["secret_key"] and self.s3_credentials["bucket"]:
            try:
                self.s3_session = aiobotocore.session.get_session()
                async with self.s3_session.create_client(
                    's3',
                    aws_access_key_id=self.s3_credentials['access_key'],
                    aws_secret_access_key=self.s3_credentials['secret_key'],
                    endpoint_url=self.s3_credentials['endpoint_url']
                ) as client:
                    self.s3_client = client
                    await client.list_buckets()
                self.logger.info("S3 client initialized")
                connection_success = True
            except Exception as e:
                self.logger.error(f"Failed to establish S3 connection: {e}", exc_info=True)
                self.s3_client = None
        else:
            self.logger.info("Skipping S3 connection: non-S3 config or incomplete credentials")

        if not connection_success:
            self.logger.error("No connections established: both SQL Server and S3 credentials are incomplete or failed")
            raise RuntimeError("Cannot establish database connections")
        self.logger.debug("Connection setup completed")

    async def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute SQL query asynchronously on SQL Server (Scenario 5).

        Args:
            query (str): SQL query to execute.

        Returns:
            List[Dict[str, Any]]: Query results as list of dictionaries.

        Raises:
            RuntimeError: If query execution fails or SQL connection is not established.
        """
        if not self.sql_pool:
            self.logger.error("SQL connection not established")
            raise RuntimeError("SQL connection not established")

        try:
            async with self.sql_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(query)
                    columns = [col[0] for col in cursor.description] if cursor.description else []
                    results = await cursor.fetchall()
                    self.logger.debug(f"Executed query: {query}, returned {len(results)} rows")
                    return [dict(zip(columns, row)) for row in results]
        except Exception as e:
            self.logger.error(f"Error executing query: {query}, {e}", exc_info=True)
            raise RuntimeError(f"Query execution failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ClientError, ConnectionError)),
        before_sleep=lambda retry_state: logging.getLogger("database_connector").warning(
            f"Retrying S3 access attempt {retry_state.attempt_number}"
        )
    )
    async def get_s3_data(self, s3_path: str) -> Dict[str, Any]:
        """
        Retrieve S3 CSV data and infer schema asynchronously (Scenario 1).

        Args:
            s3_path (str): S3 path (e.g., s3://bucket/stores.csv).

        Returns:
            Dict[str, Any]: Dictionary with columns and sample data.

        Raises:
            RuntimeError: If S3 access fails.
        """
        if not self.s3_client:
            self.logger.error("S3 client not initialized")
            raise RuntimeError("S3 client not initialized")

        try:
            if not s3_path.startswith("s3://"):
                raise ValueError("Invalid S3 path, must start with s3://")
            bucket, key = s3_path[5:].split("/", 1)

            response = await self.s3_client.get_object(Bucket=bucket, Key=key)
            async with response['Body'] as stream:
                content = await stream.read()

            df = pd.read_csv(io.BytesIO(content), nrows=10)
            columns = [{"name": col, "type": str(df[col].dtype)} for col in df.columns]
            sample_data = df.to_dict(orient="records")[:5]

            self.logger.debug(f"Retrieved S3 data from {s3_path}, {len(columns)} columns")
            return {"columns": columns, "sample_data": sample_data}
        except Exception as e:
            self.logger.error(f"Error accessing S3 path {s3_path}: {e}", exc_info=True)
            raise RuntimeError(f"S3 access failed: {e}")

    async def get_schema(self) -> Dict[str, Any]:
        """
        Retrieve schema metadata from SQL Server (Scenario 5).

        Returns:
            Dict[str, Any]: Schema dictionary with tables, columns, and relationships.

        Raises:
            RuntimeError: If schema retrieval fails or SQL connection is not established.
        """
        if not self.sql_pool:
            self.logger.error("SQL connection not established")
            raise RuntimeError("SQL connection not established")

        schema = {"tables": {}, "columns": {}, "relationships": []}
        try:
            async with self.sql_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        SELECT TABLE_SCHEMA, TABLE_NAME
                        FROM INFORMATION_SCHEMA.TABLES
                        WHERE TABLE_TYPE = 'BASE TABLE'
                    """)
                    tables = await cursor.fetchall()
                    for table_schema, table_name in tables:
                        schema["tables"].setdefault(table_schema, []).append(table_name)

                    await cursor.execute("""
                        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
                        FROM INFORMATION_SCHEMA.COLUMNS
                    """)
                    columns = await cursor.fetchall()
                    for table_schema, table_name, column_name, data_type in columns:
                        schema["columns"].setdefault(table_schema, {}).setdefault(table_name, {})[column_name] = data_type

                    await cursor.execute("""
                        SELECT 
                            fk.TABLE_SCHEMA,
                            fk.TABLE_NAME,
                            fk.COLUMN_NAME,
                            pk.TABLE_SCHEMA AS REF_TABLE_SCHEMA,
                            pk.TABLE_NAME AS REF_TABLE_NAME,
                            pk.COLUMN_NAME AS REF_COLUMN_NAME
                        FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE fk
                            ON rc.CONSTRAINT_NAME = fk.CONSTRAINT_NAME
                        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE pk
                            ON rc.UNIQUE_CONSTRAINT_NAME = pk.CONSTRAINT_NAME
                    """)
                    relationships = await cursor.fetchall()
                    for fk_schema, fk_table, fk_column, pk_schema, pk_table, pk_column in relationships:
                        schema["relationships"].append({
                            "fk_table": f"{fk_schema}.{fk_table}",
                            "fk_column": fk_column,
                            "pk_table": f"{pk_schema}.{pk_table}",
                            "pk_column": pk_column
                        })

            self.logger.info(f"Retrieved schema: {len(schema['tables'])} schemas, {len(schema['relationships'])} relationships")
            return schema
        except Exception as e:
            self.logger.error(f"Error retrieving schema: {e}", exc_info=True)
            raise RuntimeError(f"Schema retrieval failed: {e}")

    async def close(self) -> None:
        """
        Close SQL Server and S3 connections gracefully.
        """
        try:
            if self.sql_pool:
                self.sql_pool.close()
                await self.sql_pool.wait_closed()
                self.logger.info("SQL Server connection pool closed")
                self.sql_pool = None
            if self.s3_client:
                await self.s3_client.close()
                self.logger.info("S3 client closed")
                self.s3_client = None
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}", exc_info=True)

    async def check_health(self) -> bool:
        """
        Check connection health for SQL Server and S3 (Scenario 3).

        Returns:
            bool: True if at least one connection is healthy, False if both fail.
        """
        sql_healthy = False
        s3_healthy = False

        try:
            if self.sql_pool:
                async with self.sql_pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("SELECT 1")
                        await cursor.fetchall()
                        sql_healthy = True
                        self.logger.debug("SQL Server connection healthy")
            else:
                self.logger.warning("SQL connection not established")

            if self.s3_client:
                await self.s3_client.list_buckets()
                s3_healthy = True
                self.logger.debug("S3 connection healthy")
            else:
                self.logger.warning("S3 client not initialized")

            if sql_healthy or s3_healthy:
                self.logger.debug(f"Connection health check passed: SQL={sql_healthy}, S3={s3_healthy}")
                return True
            else:
                self.logger.error("Connection health check failed: no healthy connections")
                return False
        except Exception as e:
            self.logger.error(f"Connection health check failed: {e}", exc_info=True)
            return False