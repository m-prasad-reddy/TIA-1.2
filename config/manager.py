"""
Manages database configurations and connections for TIA-1.2.
Supports SQL Server (pyodbc) and S3 (boto3) connections with async operations.
Preserves TIA-1.1 functionality, fixes connection and performance issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based paths (e.g., BIKE-STORES/BikeStores/)
- Async connections for <5s query processing
- Logging to codebase/logs/DISPLAY_NAME_app.log
- Non-hardcoding with display_name and db_name
- S3 connection support
Addresses Scenarios 1 (S3 support), 3 (connection validation), and 5 (performance).
"""
import os
import json
import pyodbc
import boto3
import asyncio
from typing import List, Dict, Optional
import logging
from config.logging_setup import setup_logging

class DatabaseConnection:
    """Handles SQL Server (pyodbc) and S3 (boto3) connections with async support."""
    
    def __init__(self, display_name: str, db_name: str):
        """
        Initialize connection manager with logging.

        Args:
            display_name (str): Display name (e.g., BIKE-STORES).
            db_name (str): Database name (e.g., BikeStores).
        """
        setup_logging(display_name)
        self.logger = logging.getLogger("database_connection")
        self.display_name = display_name
        self.db_name = db_name
        self.connection = None  # SQL Server connection
        self.s3_client = None   # S3 client
        self.current_config = None
        self.logger.debug(f"Initialized DatabaseConnection for {display_name}/{db_name}")

    async def connect(self, config: Dict, retries: int = 3, delay: float = 1.0) -> bool:
        """
        Establish SQL Server connection with retries (Scenarios 3, 5).

        Args:
            config (Dict): Configuration with server, database, username, password, driver.
            retries (int): Number of retry attempts.
            delay (float): Delay between retries in seconds.

        Returns:
            bool: True if connected, False otherwise.
        """
        try:
            if 'driver' not in config:
                self.logger.error("Missing driver in config")
                return False

            # Validate ODBC driver (Scenario 3)
            drivers = [d for d in pyodbc.drivers() if config['driver'].lower() in d.lower()]
            if not drivers:
                self.logger.error(f"ODBC driver '{config['driver']}' not found. Available: {pyodbc.drivers()}")
                return False

            conn_str = (
                f"DRIVER={config['driver']};"
                f"SERVER={config['server']};"
                f"DATABASE={config['database']};"
                f"UID={config['username']};"
                f"PWD={config['password']}"
            )
            self.logger.debug(f"Attempting SQL Server connection: DRIVER={config['driver']}, SERVER={config['server']}, DATABASE={config['database']}")

            for attempt in range(1, retries + 1):
                try:
                    self.connection = await asyncio.to_thread(pyodbc.connect, conn_str, timeout=5)
                    if self.connection and self.connection.cursor():
                        self.current_config = config
                        self.logger.info(f"Connected to {config['database']} on attempt {attempt}")
                        return True
                    self.logger.error("Connection established but invalid")
                    return False
                except pyodbc.Error as e:
                    self.logger.warning(f"Connection attempt {attempt} failed: {e}")
                    if 'IM002' in str(e):
                        self.logger.error("ODBC driver not found. Install 'ODBC Driver 17 for SQL Server'.")
                        return False
                    if attempt < retries:
                        await asyncio.sleep(delay)
            self.logger.error(f"Failed to connect after {retries} attempts")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected connection error: {e}", exc_info=True)
            return False

    async def connect_s3(self, config: Dict) -> bool:
        """
        Establish S3 connection using boto3 (Scenario 1).

        Args:
            config (Dict): Configuration with bucket, access_key, secret_key, region.

        Returns:
            bool: True if connected, False otherwise.
        """
        try:
            required_keys = {'bucket', 'access_key', 'secret_key', 'region'}
            if not all(key in config for key in required_keys):
                self.logger.error(f"Missing S3 config keys: {required_keys - set(config.keys())}")
                return False

            self.logger.debug(f"Attempting S3 connection: bucket={config['bucket']}, region={config['region']}")
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config['access_key'],
                aws_secret_access_key=config['secret_key'],
                region_name=config['region']
            )
            # Test S3 connection by listing bucket
            await asyncio.to_thread(self.s3_client.list_objects_v2, Bucket=config['bucket'], MaxKeys=1)
            self.current_config = config
            self.logger.info(f"Connected to S3 bucket {config['bucket']}")
            return True
        except Exception as e:
            self.logger.error(f"S3 connection failed: {e}", exc_info=True)
            return False

    async def disconnect(self):
        """Close SQL Server and S3 connections asynchronously (Scenario 5)."""
        try:
            if self.connection:
                await asyncio.to_thread(self.connection.close)
                self.logger.debug("SQL Server connection closed")
            if self.s3_client:
                self.s3_client.close()
                self.logger.debug("S3 client closed")
            self.connection = None
            self.s3_client = None
            self.current_config = None
            self.logger.info("All connections closed")
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}", exc_info=True)

    async def is_connected(self) -> bool:
        """
        Validate active connection asynchronously (Scenario 3).

        Returns:
            bool: True if connected, False otherwise.
        """
        try:
            if self.connection:
                cursor = await asyncio.to_thread(self.connection.cursor)
                await asyncio.to_thread(cursor.execute, "SELECT 1")
                await asyncio.to_thread(cursor.close)
                self.logger.debug("SQL Server connection validated")
                return True
            if self.s3_client and self.current_config:
                await asyncio.to_thread(self.s3_client.list_objects_v2, Bucket=self.current_config['bucket'], MaxKeys=1)
                self.logger.debug("S3 connection validated")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}", exc_info=True)
            return False

    async def get_cursor(self) -> Optional[pyodbc.Cursor]:
        """
        Get SQL Server cursor asynchronously (Scenario 5).

        Returns:
            Optional[pyodbc.Cursor]: Cursor if connected, None otherwise.
        """
        try:
            if self.connection:
                cursor = await asyncio.to_thread(self.connection.cursor)
                self.logger.debug("Retrieved SQL Server cursor")
                return cursor
            self.logger.warning("No SQL Server connection for cursor")
            return None
        except Exception as e:
            self.logger.error(f"Error getting cursor: {e}", exc_info=True)
            return None

class DBConfigManager:
    """Manages loading and validation of database and S3 configurations."""
    
    def __init__(self, display_name: str, db_name: str):
        """
        Initialize config manager with logging.

        Args:
            display_name (str): Display name (e.g., BIKE-STORES).
            db_name (str): Database name (e.g., BikeStores).
        """
        setup_logging(display_name)
        self.logger = logging.getLogger("db_config_manager")
        self.display_name = display_name
        self.db_name = db_name
        self.logger.debug(f"Initialized DBConfigManager for {display_name}/{db_name}")

    async def load_configs(self, config_path: str = None) -> List[Dict]:
        """
        Load and validate configurations asynchronously (Scenarios 1, 5).

        Args:
            config_path (str, optional): Path to config file. Defaults to DISPLAY_NAME/db-name/database_configurations.json.

        Returns:
            List[Dict]: List of validated configurations.
        """
        try:
            if not config_path:
                config_path = os.path.join(self.display_name, self.db_name, "database_configurations.json")
                self.logger.debug(f"Using default config path: {config_path}")

            if not os.path.exists(config_path):
                self.logger.error(f"Config file not found: {config_path}")
                raise FileNotFoundError(f"Config file not found at {config_path}")

            async with asyncio.Lock():
                with open(config_path) as f:
                    config_dict = json.load(f)

            configs = await self._validate_and_convert_configs(config_dict)
            self.logger.info(f"Loaded {len(configs)} configurations from {config_path}")
            return configs
        except Exception as e:
            self.logger.error(f"Error loading configs: {e}", exc_info=True)
            return []

    async def _validate_and_convert_configs(self, config_dict: Dict) -> List[Dict]:
        """
        Validate and convert config dictionary asynchronously (Scenarios 1, 3).

        Args:
            config_dict (Dict): Raw config dictionary.

        Returns:
            List[Dict]: Validated configurations.

        Raises:
            ValueError: If configs are invalid.
        """
        try:
            if not isinstance(config_dict, dict):
                self.logger.error("Config file must contain a dictionary")
                raise ValueError("Config file should contain a dictionary of configurations")

            configs = []
            sql_required = {'server', 'database', 'username', 'password', 'driver'}
            s3_required = {'bucket', 'access_key', 'secret_key', 'region'}

            for config_name, config in config_dict.items():
                if not isinstance(config, dict):
                    self.logger.error(f"Configuration '{config_name}' must be a dictionary")
                    raise ValueError(f"Configuration '{config_name}' must be a dictionary")

                config_with_name = config.copy()
                config_with_name['name'] = config_name

                if sql_required.issubset(config.keys()):
                    self.logger.debug(f"Valid SQL Server config: {config_name}")
                    configs.append(config_with_name)
                elif s3_required.issubset(config.keys()):
                    self.logger.debug(f"Valid S3 config: {config_name}")
                    configs.append(config_with_name)
                else:
                    missing_sql = sql_required - set(config.keys())
                    missing_s3 = s3_required - set(config.keys())
                    self.logger.error(f"Invalid config '{config_name}': missing SQL keys {missing_sql} or S3 keys {missing_s3}")
                    raise ValueError(f"Invalid config '{config_name}'")

            if not configs:
                self.logger.error("No valid configurations found")
                raise ValueError("No valid configurations found")

            return configs
        except Exception as e:
            self.logger.error(f"Error validating configs: {e}", exc_info=True)
            raise