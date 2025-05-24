"""
Singleton for SentenceTransformer model in TIA-1.2.
Manages a single instance of all-MiniLM-L6-v2 for embedding operations.
Preserves TIA-1.1 functionality, fixes performance and initialization issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based logging (e.g., logs/BIKE-STORES_app.log at application root)
- Async model initialization with lazy loading for <5s query processing
- Non-hardcoding with config_name from app-configs/db_configurations.json
Addresses Scenarios 3 (model initialization) and 5 (performance).
"""
import logging
import asyncio
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from config.logging_setup import setup_logging
from config.utils import load_db_config

class ModelSingleton:
    """Singleton class to manage a single SentenceTransformer model instance."""
    
    _instance = None
    
    def __new__(cls, config_name: str = None):
        """
        Ensure only one instance of ModelSingleton is created, using app-configs/db_configurations.json.

        Args:
            config_name (str, optional): Configuration name (e.g., 'bikestores').
        """
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            # Load database config
            config = load_db_config(config_name)
            display_name = config.get("display_name", os.getenv("DISPLAY_NAME", "BIKE-STORES"))
            db_name = config.get("database", os.getenv("DB_NAME", "BikeStores"))
            cls._instance._initialize(display_name, db_name)
        return cls._instance

    def _initialize(self, display_name: str, db_name: str):
        """
        Initialize ModelSingleton with logging and configuration, deferring model loading.

        Args:
            display_name (str): Display name (e.g., BIKE-STORES).
            db_name (str): Database name (e.g., BikeStores).
        """
        setup_logging(display_name, db_name)
        self.logger = logging.getLogger("model_singleton")
        self.display_name = display_name
        self.db_name = db_name
        self.model = None
        self.logger.info(f"Initialized ModelSingleton for {display_name}/{db_name} (model loading deferred)")

    async def ensure_model(self):
        """
        Ensure SentenceTransformer model is initialized asynchronously with retries (Scenarios 3, 5).

        Raises:
            RuntimeError: If model cannot be initialized after retries.
        """
        if self.model is None:
            self.logger.info(f"Loading SentenceTransformer model for {self.display_name}/{self.db_name}")
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    self.model = await asyncio.to_thread(SentenceTransformer, 'all-MiniLM-L6-v2')
                    self.logger.info(f"Initialized SentenceTransformer model on attempt {attempt}")
                    return
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt} failed to initialize SentenceTransformer: {e}")
                    if attempt < max_attempts:
                        await asyncio.sleep(0.5 * (2 ** (attempt - 1)))  # Exponential backoff
                    else:
                        self.logger.error(f"Failed to initialize SentenceTransformer after {max_attempts} attempts: {e}", exc_info=True)
                        raise RuntimeError(f"Cannot initialize SentenceTransformer model for {self.display_name}/{self.db_name}")
        else:
            self.logger.debug(f"Model already initialized for {self.display_name}/{self.db_name}")

    def encode(self, text: str, **kwargs) -> np.ndarray:
        """
        Encode text using the SentenceTransformer model.

        Args:
            text (str): Text to encode.
            **kwargs: Additional arguments for SentenceTransformer.encode.

        Returns:
            np.ndarray: Text embedding.

        Raises:
            RuntimeError: If model is not initialized.
        """
        if self.model is None:
            raise RuntimeError(f"SentenceTransformer model not initialized for {self.display_name}/{self.db_name}")
        try:
            embedding = self.model.encode(text, **kwargs)
            self.logger.debug(f"Encoded text: {text[:50]}...")
            return embedding
        except Exception as e:
            self.logger.error(f"Error encoding text: {e}", exc_info=True)
            raise