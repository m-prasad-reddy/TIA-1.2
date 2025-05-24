"""
Manages regex patterns for matching tables and columns in TIA-1.2 queries.
Supports schema/table filtering, S3 table names (e.g., stores.csv), and verbose logging.
Preserves TIA-1.1 functionality, fixes table selection and performance issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based logging (e.g., codebase/logs/BIKE-STORES_app.log)
- Pattern caching for <5s query processing
- Non-hardcoding with display_name and db_name
- S3 table pattern support
- Verbose logging restored to match TIA-1.1
Addresses Scenarios 1 (table identification), 2 (synonym quality), and 5 (performance).
"""
import logging
import asyncio
import os
import re
from typing import Dict, List, Optional
from config.logging_setup import setup_logging

class PatternManager:
    """
    Manages regex patterns for matching tables and columns in queries.
    Supports filtering by schemas and tables, handles S3 table names, and provides detailed logging.
    """
    def __init__(self, schema_dict: Dict, display_name: str, db_name: str, 
                 schemas: List[str] = None, tables: List[str] = None):
        """
        Initialize PatternManager with schema dictionary, filters, and logging.

        Args:
            schema_dict (Dict): Schema dictionary with tables and columns.
            display_name (str): Display name (e.g., BIKE-STORES).
            db_name (str): Database name (e.g., BikeStores).
            schemas (List[str], optional): Schemas to include (e.g., ['sales']). Defaults to all.
            tables (List[str], optional): Tables to include (e.g., ['sales.stores']). Defaults to all.
        """
        try:
            setup_logging(display_name)
            self.logger = logging.getLogger("pattern_manager")
            self.logger.info(f"Setting up PatternManager for {display_name}/{db_name}")
        except Exception as e:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(os.path.join("codebase", "logs", f"{display_name}_app.log")),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger("pattern_manager")
            self.logger.error(f"Failed to setup logging: {e}", exc_info=True)

        self.display_name = display_name
        self.db_name = db_name
        self.schema_dict = schema_dict
        self.schemas = schemas or []
        self.tables = tables or []
        self.patterns = {}

        # Validate table formats
        valid_tables = []
        for table in self.tables:
            if '.' in table:
                valid_tables.append(table)
            else:
                self.logger.warning(f"Invalid table format: '{table}' (expected schema.table)")
        self.tables = valid_tables
        self.logger.debug(f"Validated tables: {self.tables}")

        # Generate and cache patterns
        asyncio.run(self._generate_patterns())
        self.logger.debug(f"Initialized PatternManager for {display_name}/{db_name}, schemas={self.schemas}, tables={self.tables}")

    async def _generate_patterns(self):
        """
        Generate regex patterns for tables and columns asynchronously (Scenarios 1, 5).

        Caches patterns, supports S3 table names (e.g., stores), and includes flexible patterns.
        """
        try:
            self.logger.debug("Starting pattern generation")
            self.patterns.clear()
            if not self.schema_dict or 'tables' not in self.schema_dict:
                self.logger.error("Invalid or empty schema dictionary")
                return

            # Determine schemas/tables to process
            if self.tables:
                self.logger.debug(f"Processing specified tables: {self.tables}")
                target_tables = [(t.split('.')[0], t.split('.')[1]) for t in self.tables]
            else:
                target_schemas = self.schemas if self.schemas else self.schema_dict['tables'].keys()
                self.logger.debug(f"Processing schemas: {list(target_schemas)}")
                target_tables = []
                for schema in target_schemas:
                    if schema in self.schema_dict['tables']:
                        for table in self.schema_dict['tables'][schema]:
                            target_tables.append((schema, table))

            # Generate patterns
            for schema, table in target_tables:
                if schema not in self.schema_dict['tables'] or table not in self.schema_dict['tables'][schema]:
                    self.logger.warning(f"Skipping invalid table: {schema}.{table}")
                    continue

                self.logger.debug(f"Processing table: {schema}.{table}")
                table_full = f"{schema}.{table}" if schema != 's3' else table  # S3 tables are schema-less
                # Table pattern with word boundaries
                table_pattern = r'\b' + re.escape(table.lower()) + r'\b'
                self.patterns[table_pattern] = [table_full]
                self.logger.debug(f"Generated table pattern: '{table_pattern}' -> {table_full}")

                # Flexible pattern for singular/plural (Scenario 1)
                if table.endswith('s'):
                    singular = table[:-1]
                    singular_pattern = r'\b' + re.escape(singular.lower()) + r'\b'
                    self.patterns[singular_pattern] = self.patterns.get(singular_pattern, []) + [table_full]
                    self.logger.debug(f"Generated flexible pattern: '{singular_pattern}' -> {table_full}")

                # Column patterns
                columns = self.schema_dict['columns'].get(schema, {}).get(table, [])
                self.logger.debug(f"Processing {len(columns)} columns for {schema}.{table}")
                for column in columns:
                    col_pattern = r'\b' + re.escape(column.lower()) + r'\b'
                    self.patterns[col_pattern] = self.patterns.get(col_pattern, []) + [table_full]
                    self.logger.debug(f"Generated column pattern: '{col_pattern}' -> {table_full}")

            self.logger.debug(f"Completed pattern generation, total patterns: {len(self.patterns)}")
        except Exception as e:
            self.logger.error(f"Error generating patterns: {e}", exc_info=True)

    def get_patterns(self) -> Dict[str, List[str]]:
        """
        Retrieve cached patterns (Scenario 5).

        Returns:
            Dict[str, List[str]]: Regex patterns mapped to table names (schema.table or S3 table).
        """
        try:
            self.logger.debug(f"Retrieving {len(self.patterns)} patterns")
            return self.patterns
        except Exception as e:
            self.logger.error(f"Error retrieving patterns: {e}", exc_info=True)
            return {}

    async def update_patterns(self, schema_dict: Dict):
        """
        Update patterns with a new schema dictionary asynchronously (Scenario 1).

        Args:
            schema_dict (Dict): Updated schema dictionary.
        """
        try:
            self.logger.debug("Updating patterns with new schema dictionary")
            self.schema_dict = schema_dict
            await self._generate_patterns()
            self.logger.debug("Patterns updated successfully")
        except Exception as e:
            self.logger.error(f"Error updating patterns: {e}", exc_info=True)
            

"""
Configuration file for TIA-1.2 patterns and synonyms.
Provides static synonyms for table/column names and query patterns for training data generation.
Supports Scenario 4 (synonym quality) with a SYNONYMS dictionary for BIKE-STORES schema.
"""

# Static synonyms for table/column names and common terms (Scenario 4)
SYNONYMS = {
    "store": ["shop", "retail", "outlet", "branch"],
    "order": ["purchase", "booking", "transaction"],
    "customer": ["client", "buyer", "user"],
    "product": ["item", "goods", "merchandise"],
    "category": ["type", "group", "class"],
    "order_id": ["orderid", "order number"],
    "order_date": ["order date", "date"],
    "store_id": ["storeid", "store number"],
    "store_name": ["store name", "name"],
    "customer_id": ["customerid", "client id"],
    "product_id": ["productid", "item id"],
    "category_id": ["categoryid", "type id"],
    "category_name": ["category name", "type name"]
}

# Placeholder for query patterns (e.g., regex for query parsing)
QUERY_PATTERNS = {
    "select": r"select\s+.*?\s+from\s+\w+",
    "where": r"where\s+.*?(?=\s*(?:group\s+by|order\s+by|$))",
    # Add more patterns as needed
}