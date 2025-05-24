"""
Utility functions for loading configurations in TIA-1.2.
Handles dynamic selection of database configurations from db_configurations.json.
Supports TIA-1.2 non-hardcoding and dynamic configuration selection.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

def load_db_config(config_name: str = None) -> Dict:
    """
    Load database configuration from app-configs/db_configurations.json.

    Args:
        config_name (str, optional): Configuration name (e.g., 'bikestores') or display_name (e.g., 'BIKE-STORES').

    Returns:
        Dict: Selected configuration.

    Raises:
        ValueError: If config file is missing, invalid, or contains no valid configurations.
    """
    logger = logging.getLogger("tia")
    config_path = Path("app-configs/db_configurations.json")
    
    # Check if config file exists
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise ValueError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            configs = json.load(f)

        # Validate JSON structure
        if not isinstance(configs, dict) or "databases" not in configs or not isinstance(configs["databases"], list):
            logger.error(f"Invalid config structure in {config_path}: 'databases' must be a non-empty list")
            raise ValueError(f"Invalid config structure in {config_path}: 'databases' must be a non-empty list")

        if not configs["databases"]:
            logger.error(f"No configurations found in {config_path}")
            raise ValueError(f"No configurations found in {config_path}")

        valid_configs = []
        required_fields = {"name", "display_name", "database", "type"}
        sqlserver_fields = {"server", "username", "password", "driver"}
        s3_fields = {"bucket_name"}

        # Validate each config
        for config in configs["databases"]:
            if not isinstance(config, dict):
                logger.warning(f"Skipping invalid config entry: {config} (not a dictionary)")
                continue

            # Check required fields
            if not all(field in config for field in required_fields):
                missing = required_fields - set(config.keys())
                logger.warning(f"Skipping config {config.get('name', 'unknown')}: missing fields {missing}")
                continue

            # Validate connection-specific fields
            config_type = config["type"].lower()
            if config_type == "sql server":
                if not all(field in config for field in sqlserver_fields):
                    missing = sqlserver_fields - set(config.keys())
                    logger.warning(f"Skipping SQL Server config {config['name']}: missing fields {missing}")
                    continue
            elif config_type == "s3":
                if not all(field in config for field in s3_fields):
                    missing = s3_fields - set(config.keys())
                    logger.warning(f"Skipping S3 config {config['name']}: missing fields {missing}")
                    continue
            else:
                logger.warning(f"Skipping config {config['name']}: unsupported type {config_type}")
                continue

            # Map 'type' to 'connection_type'
            config = config.copy()
            config["connection_type"] = config.pop("type")
            valid_configs.append(config)

        if not valid_configs:
            logger.error(f"No valid configurations found in {config_path}")
            raise ValueError(f"No valid configurations found in {config_path}")

        # Default to first config if config_name is None
        if not config_name:
            config = valid_configs[0]
            logger.debug(f"No config_name provided, using default: {config['name']}")
            return config

        # Find config by name or display_name
        for config in valid_configs:
            if config["name"] == config_name or config["display_name"] == config_name:
                logger.debug(f"Loaded config for {config_name} from {config_path}")
                return config

        logger.error(f"No config found for {config_name}")
        raise ValueError(f"No config found for {config_name}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {config_path}: {e}", exc_info=True)
        raise ValueError(f"Invalid JSON in {config_path}: {e}")
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}", exc_info=True)
        raise ValueError(f"Error loading config from {config_path}: {e}")

def list_configurations() -> List[Tuple[str, str]]:
    """
    List available database configurations for UI/CLI selection.

    Returns:
        List[Tuple[str, str]]: List of (name, display_name) tuples.

    Raises:
        ValueError: If config file is missing, invalid, or contains no valid configurations.
    """
    logger = logging.getLogger("tia")
    config_path = Path("app-configs/db_configurations.json")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise ValueError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            configs = json.load(f)

        if not isinstance(configs, dict) or "databases" not in configs or not isinstance(configs["databases"], list):
            logger.error(f"Invalid config structure in {config_path}: 'databases' must be a non-empty list")
            raise ValueError(f"Invalid config structure in {config_path}: 'databases' must be a non-empty list")

        if not configs["databases"]:
            logger.error(f"No configurations found in {config_path}")
            raise ValueError(f"No configurations found in {config_path}")

        result = []
        required_fields = {"name", "display_name"}

        for config in configs["databases"]:
            if not isinstance(config, dict) or not all(field in config for field in required_fields):
                logger.warning(f"Skipping invalid config entry: {config}")
                continue
            result.append((config["name"], config["display_name"]))

        if not result:
            logger.error(f"No valid configurations found in {config_path}")
            raise ValueError(f"No valid configurations found in {config_path}")

        logger.debug(f"Loaded {len(result)} configurations from {config_path}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {config_path}: {e}", exc_info=True)
        raise ValueError(f"Invalid JSON in {config_path}: {e}")
    except Exception as e:
        logger.error(f"Error listing configurations from {config_path}: {e}", exc_info=True)
        raise ValueError(f"Error listing configurations from {config_path}: {e}")