"""
Logging setup for TIA-1.2, reading configuration from app-configs/logging_config.json.
Provides synchronous logging setup for consistency across components.
Logs to logs/{display_name}_app.log at the application root (e.g., logs/BIKE-STORES_app.log).
Addresses Scenario 5 (performance) with optimized logging and Scenario 4 (logging consistency).
"""
import json
import logging
import logging.config
from pathlib import Path

def setup_logging(display_name: str, db_name: str = None) -> None:
    """
    Configure logging for TIA-1.2 using app-configs/logging_config.json.

    Logs are written to logs/{display_name}_app.log at the application root.
    Falls back to a default file-based configuration if the config file is missing or invalid.

    Args:
        display_name (str): Display name for the log file (e.g., BIKE-STORES).
        db_name (str, optional): Database name (e.g., BikeStores). Defaults to None.
    """
    try:
        # Define log path
        log_dir = Path("logs")
        log_file = log_dir / f"{display_name}_app.log"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Load config from app-configs
        config_path = Path("app-configs/logging_config.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Logging config not found: {config_path}")

        with config_path.open("r") as f:
            config = json.load(f)

        # Update file handler path
        if "handlers" in config and "file" in config["handlers"]:
            config["handlers"]["file"]["filename"] = str(log_file)
        else:
            raise ValueError("Invalid logging config: missing file handler")

        # Apply logging configuration
        logging.config.dictConfig(config)
        logger = logging.getLogger("tia")
        logger.info(f"Logging configured for {display_name}/{db_name or 'default'} using app-configs")
    except Exception as e:
        # Fallback to default file-based configuration
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file)
            ]
        )
        logger = logging.getLogger("tia")
        logger.error(f"Failed to configure logging from app-configs: {e}", exc_info=True)
        logger.info(f"Using default logging configuration for {display_name}/{db_name or 'default'}")