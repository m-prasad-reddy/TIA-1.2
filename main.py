"""
Main entry point for the Table Identifier Agent (TIA-1.2).
Orchestrates configuration loading, SQL Server/S3 connections, manager initialization, and CLI interaction.
Preserves TIA-1.1 functionality, fixes invalid table selection and path issues, and adds TIA-1.2 enhancements:
- S3 CSV support with async I/O
- Training data generation
- DISPLAY_NAME-based paths and logging
- Configuration validation
- Performance optimizations (<5s query time, <2s S3 latency)
- Integration fix for FeedbackManager with is_s3_config
Addresses scenarios 1–8 (e.g., table identification, feedback deduplication).
"""
import logging
import os
import asyncio
from typing import Dict, List
from config.manager import DBConfigManager, DatabaseConnection
from config.patterns import PatternManager
from config.metadata_initializer import MetadataInitializer
from config.cache_synchronizer import CacheSynchronizer
from schema.manager import SchemaManager
from feedback.manager import FeedbackManager
from analysis.table_identifier import TableIdentifier
from analysis.name_match_manager import NameMatchManager
from analysis.processor import NLPPipeline
from nlp.QueryProcessor import QueryProcessor
from cli.interface import DatabaseAnalyzerCLI
from generate_training_data import TrainingDataGenerator
from config.logging_setup import setup_logging

class DatabaseAnalyzer:
    """Main class for TIA-1.2, managing database analysis and query processing."""
    def __init__(self):
        """
        Initialize analyzer with logging and core components.
        Sets up logger and placeholders for managers and configuration.
        """
        self.logger = logging.getLogger("analyzer")
        self.connection_manager = DatabaseConnection()
        self.config_manager = DBConfigManager()
        self.schema_manager = None
        self.schema_dict = {}
        self.pattern_manager = None
        self.feedback_manager = None
        self.nlp_pipeline = None
        self.name_matcher = None
        self.table_identifier = None
        self.query_processor = None
        self.cache_synchronizer = None
        self.current_config = None
        self.logger.debug("Initialized DatabaseAnalyzer")

    async def run(self):
        """
        Run the analyzer, handling configuration selection, connection, and CLI.
        Supports training data generation and scenario-based query processing (Scenarios 1–8).
        """
        try:
            # Load and validate configurations
            configs = self.load_configs()
            if not configs:
                print("No configurations loaded.")
                self.logger.error("No configurations loaded")
                return

            # Display configuration options
            print("\nAvailable Configurations:")
            for i, config in enumerate(configs, 1):
                print(f"{i}. {config.get('display_name', config.get('database', 'Unknown'))}")
            print(f"{len(configs) + 1}. Generate training data")
            print(f"{len(configs) + 2}. Cancel")
            choice = input("Select option: ").strip()

            # Validate user choice
            if not choice.isdigit() or int(choice) < 1 or int(choice) > len(configs) + 2:
                print("Invalid choice.")
                self.logger.error(f"Invalid choice: {choice}")
                return

            if int(choice) == len(configs) + 2:
                print("Cancelled.")
                self.logger.info("Application cancelled by user")
                return

            # Handle training data generation
            if int(choice) == len(configs) + 1:
                config_choice = input(f"Select config for training data (1-{len(configs)}): ").strip()
                if not config_choice.isdigit() or int(config_choice) < 1 or int(config_choice) > len(configs):
                    print("Invalid config choice.")
                    self.logger.error(f"Invalid config choice for training data: {config_choice}")
                    return
                await self.generate_training_data(configs[int(config_choice) - 1])
                return

            # Set configuration and connect
            self.set_current_config(configs[int(choice) - 1])
            if not await self.connect_to_database():
                print(f"Failed to connect to {self.current_config['database']}")
                self.logger.error(f"Failed to connect to {self.current_config['database']}")
                return

            # Initialize CLI with schema and feedback
            cli = DatabaseAnalyzerCLI(
                self.current_config['display_name'],
                self.current_config['database'],
                schema_dict=self.schema_dict,
                feedback_manager=self.feedback_manager,
                schemas=self.current_config.get('schemas', []),
                tables=self.current_config.get('tables', []),
                query_processor=self.query_processor
            )
            await cli.run()
        except Exception as e:
            self.logger.error(f"Error running application: {e}", exc_info=True)
            print(f"Error: {e}")
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """
        Clean up resources, saving feedback and closing connections.
        Ensures weights and name matches are saved on exit.
        """
        try:
            if self.table_identifier:
                await self.table_identifier.update_weights_from_feedback("", [])
                self.logger.debug("Saved name matches and weights")
            if self.cache_synchronizer:
                self.cache_synchronizer.close()
                self.logger.debug("Closed cache synchronizer")
            if self.connection_manager:
                await self.connection_manager.close()
                self.logger.debug("Closed database/S3 connection")
            self.logger.info(f"Application shutdown for {self.current_config.get('display_name', 'unknown')}")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)

    def load_configs(self, config_path: str = "database_configurations.json") -> List[Dict]:
        """
        Load and validate configurations from JSON file (Scenario 7).
        Ensures unique display_name and required fields.
        Args:
            config_path (str): Path to configuration file.
        Returns:
            List[Dict]: Validated configurations.
        """
        try:
            configs = self.config_manager.load_configs(config_path)
            for config in configs:
                config['schemas'] = config.get('schemas', [])
                config['tables'] = config.get('tables', [])
            self.logger.debug(f"Loaded {len(configs)} configurations")
            return configs
        except Exception as e:
            self.logger.error(f"Error loading configs: {e}", exc_info=True)
            return []

    def set_current_config(self, config: Dict):
        """
        Set the current configuration and configure logging.
        Args:
            config (Dict): Selected configuration.
        """
        self.current_config = config
        display_name = config.get('display_name', 'unknown')
        db_name = config.get('database', 'unknown')
        setup_logging(display_name)  # Configure logging to codebase/logs/DISPLAY_NAME_app.log
        self.logger = logging.getLogger("analyzer")  # Rebind logger after setup
        self.logger.debug(f"Set config: {display_name}/{db_name}")

    async def connect_to_database(self) -> bool:
        """
        Connect to SQL Server or S3 based on configuration.
        Supports async S3 connections for <2s latency.
        Returns:
            bool: True if connection successful, False otherwise.
        """
        if not self.current_config:
            self.logger.error("No configuration selected")
            return False
        if not await self.connection_manager.connect(self.current_config):
            self.logger.error(f"Database/S3 connection failed for {self.current_config['database']}")
            return False
        try:
            await self._initialize_managers()
            self.logger.info(f"Connected to {self.current_config['database']}")
            return True
        except Exception as e:
            self.logger.error(f"Initialization error: {e}", exc_info=True)
            return False

    async def _initialize_managers(self):
        """
        Initialize managers for query processing and caching.
        Uses DISPLAY_NAME/db-name/ paths for cache and metadata.
        Passes is_s3_config to FeedbackManager to support RDBMS and S3 validation.
        """
        display_name = self.current_config['display_name']
        db_name = self.current_config['database']
        
        # Initialize schema manager
        self.schema_manager = SchemaManager(
            display_name,
            db_name,
            self.current_config.get('schemas', []),
            self.current_config.get('tables', [])
        )
        
        # Initialize metadata with S3 support
        metadata_initializer = MetadataInitializer(
            display_name,
            db_name,
            self.schema_manager,
            self.connection_manager
        )
        s3_config = {
            k: self.current_config[k]
            for k in ['s3_endpoint', 'access_key', 'secret_key', 'bucket_name']
            if k in self.current_config
        }
        if not await metadata_initializer.initialize(s3_config if s3_config else None):
            raise RuntimeError(f"Metadata initialization failed for {display_name}/{db_name}")

        # Load schema
        self.schema_dict = self.schema_manager.load_from_cache()
        if not self.schema_dict:
            raise RuntimeError(f"Schema loading failed for {display_name}/{db_name}")

        # Initialize remaining managers
        self.cache_synchronizer = CacheSynchronizer(display_name, db_name)
        self.pattern_manager = PatternManager(
            self.schema_dict,
            self.current_config.get('schemas', []),
            self.current_config.get('tables', [])
        )
        # Pass is_s3_config based on presence of s3_endpoint
        self.feedback_manager = FeedbackManager(
            display_name,
            db_name,
            self.cache_synchronizer,
            is_s3_config=bool(self.current_config.get('s3_endpoint'))
        )
        self.nlp_pipeline = NLPPipeline(
            self.pattern_manager,
            display_name,
            db_name
        )
        self.name_matcher = NameMatchManager(
            display_name,
            db_name
        )
        self.table_identifier = TableIdentifier(
            self.schema_dict,
            self.feedback_manager,
            self.pattern_manager,
            self.cache_synchronizer
        )
        self.query_processor = QueryProcessor(
            self.connection_manager,
            self.schema_dict,
            self.nlp_pipeline,
            self.table_identifier,
            self.name_matcher,
            self.pattern_manager,
            display_name,
            db_name,
            self.cache_synchronizer
        )
        self.logger.debug(f"Initialized managers for {display_name}/{db_name}")

    async def generate_training_data(self, config: Dict):
        """
        Generate training data CSV for the selected configuration (TIA-1.2).
        Stores output in DISPLAY_NAME/db-name/{db_name}_training.csv.
        Args:
            config (Dict): Configuration for training data generation.
        """
        display_name = config['display_name']
        db_name = config['database']
        setup_logging(display_name)
        self.logger = logging.getLogger("analyzer")
        try:
            # Initialize schema manager for training data
            schema_manager = SchemaManager(
                display_name,
                db_name,
                config.get('schemas', []),
                config.get('tables', [])
            )
            # Generate training data
            generator = TrainingDataGenerator(
                display_name,
                db_name,
                schema_manager,
                self.connection_manager
            )
            s3_config = {
                k: config[k]
                for k in ['s3_endpoint', 'access_key', 'secret_key', 'bucket_name']
                if k in config
            }
            output_file = await asyncio.to_thread(generator.generate, s3_config if s3_config else None)
            print(f"Training data generated: {output_file}")
            self.logger.info(f"Training data generated: {output_file}")
        except Exception as e:
            self.logger.error(f"Error generating training data: {e}", exc_info=True)
            print(f"Error generating training data: {e}")

if __name__ == "__main__":
    setup_logging("default")  # Initial logging setup
    analyzer = DatabaseAnalyzer()
    asyncio.run(analyzer.run())