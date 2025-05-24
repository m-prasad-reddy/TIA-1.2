"""
Manages feedback storage, validation, and retrieval for TIA-1.2.
Stores query feedback in cache.db via CacheSynchronizer, supporting SQL Server (RDBMS) and S3 CSV data sources.
Preserves TIA-1.1 functionality, fixes deduplication, table validation, and hardcoding issues, and adds TIA-1.2 enhancements:
- Async operations for performance
- DISPLAY_NAME-based paths and logging
- Dynamic S3 table validation using csv_metadata.json
- Explicit support for RDBMS and S3 data sources
Addresses Scenarios 1 (table identification), 2 (synonym quality), 3 (feedback usage),
5 (performance), and 8 (deduplication).
"""
import logging
import asyncio
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import util
from config.cache_synchronizer import CacheSynchronizer
from config.model_singleton import ModelSingleton
from config.logging_setup import setup_logging

class FeedbackManager:
    """Manages feedback storage, validation, and retrieval for query processing."""
    def __init__(self, display_name: str, db_name: str, cache_synchronizer: CacheSynchronizer, is_s3_config: bool = False):
        """
        Initialize FeedbackManager with cache and model dependencies.

        Args:
            display_name (str): Display name (e.g., BIKE-STORES).
            db_name (str): Database name (e.g., BikeStores).
            cache_synchronizer (CacheSynchronizer): Cache manager instance.
            is_s3_config (bool): True if S3 CSV configuration, False for RDBMS (default: False).
        """
        setup_logging(display_name)
        self.display_name = display_name
        self.db_name = db_name
        self.cache_synchronizer = cache_synchronizer
        self.is_s3_config = is_s3_config  # Indicates S3 or RDBMS configuration
        self.model = ModelSingleton(display_name, db_name).model
        self.logger = logging.getLogger("feedback_manager")
        self.embedding_cache = {}  # Cache for query embeddings
        self.logger.debug(f"Initialized FeedbackManager for {display_name}/{db_name}, is_s3_config: {is_s3_config}")

    async def add_feedback(self, query: str, tables: List[str], schema_dict: Dict, count: int = 1) -> bool:
        """
        Add feedback entry with validation and embedding (Scenarios 1, 3, 8).

        Validates tables against schema_dict (RDBMS) or S3 metadata, generates embedding, and stores in cache.
        Checks for similar feedback (threshold ≥0.90) to prevent duplicates.

        Args:
            query (str): User query (e.g., "stores in USA").
            tables (List[str]): Matched tables (e.g., ["sales.stores"]).
            schema_dict (Dict): Schema dictionary for validation.
            count (int): Feedback count (default: 1).

        Returns:
            bool: True if added successfully, False otherwise.
        """
        try:
            # Validate feedback
            if not self.validate_feedback(query, tables, schema_dict):
                self.logger.warning(f"Invalid feedback for query: {query}, tables: {tables}")
                return False

            # Generate or retrieve embedding
            if query in self.embedding_cache:
                embedding = self.embedding_cache[query]
                self.logger.debug(f"Retrieved cached embedding for query: {query}")
            else:
                embedding = self.model.encode(query, show_progress_bar=False) if self.model else np.zeros(384, dtype=np.float32)
                self.embedding_cache[query] = embedding
                self.logger.debug(f"Generated embedding for query: {query}")

            # Check for similar feedback to prevent duplicates (Scenario 8)
            similar = await self.find_similar_feedback(query, threshold=0.90)
            for fb_query, fb_tables, sim in similar:
                if sim >= 0.90 and set(fb_tables) == set(tables):
                    await self.update_feedback_count(fb_query, increment=count)
                    self.logger.debug(f"Updated count for similar feedback: {fb_query}, similarity: {sim}")
                    return True

            # Store new feedback
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            await asyncio.to_thread(
                self.cache_synchronizer.write_feedback,
                timestamp,
                query,
                tables,
                embedding,
                count
            )
            self.logger.info(f"Added feedback for query: {query}, tables: {tables}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding feedback for query {query}: {e}", exc_info=True)
            return False

    def validate_feedback(self, query: str, tables: List[str], schema_dict: Dict) -> bool:
        """
        Validate feedback entry (Scenario 1).

        Ensures query is non-empty and tables are valid for the data source:
        - RDBMS (SQL Server): Tables must exist in schema_dict.
        - S3 CSV: Tables must match table names from csv_metadata.json in DISPLAY_NAME/db-name/.

        Args:
            query (str): User query.
            tables (List[str]): Matched tables.
            schema_dict (Dict): Schema dictionary.

        Returns:
            bool: True if valid, False otherwise.
        """
        if not query or not tables:
            self.logger.warning("Empty query or tables in feedback")
            return False

        valid_tables = []
        if not self.is_s3_config:
            # RDBMS (SQL Server) validation
            self.logger.debug(f"Validating feedback for RDBMS data source: {self.display_name}/{self.db_name}")
            for table in tables:
                if table in schema_dict:
                    valid_tables.append(table)
                else:
                    self.logger.warning(f"Invalid RDBMS table in feedback: {table}")
                    return False
        else:
            # S3 CSV validation
            self.logger.debug(f"Validating feedback for S3 CSV data source: {self.display_name}/{self.db_name}")
            metadata_file = os.path.join(self.display_name, self.db_name, "csv_metadata.json")
            s3_tables = []
            try:
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    s3_tables = [entry["table_name"] for entry in metadata]
                    self.logger.debug(f"Loaded S3 table names from {metadata_file}: {s3_tables}")
                else:
                    # Fallback to default table names if metadata is unavailable
                    s3_tables = [
                        "brands", "candidates", "categories", "customers", "employees",
                        "orders", "order_items", "products", "staffs", "stocks", "stores"
                    ]
                    self.logger.warning(f"csv_metadata.json not found at {metadata_file}, using default S3 table names")
            except Exception as e:
                self.logger.error(f"Error loading csv_metadata.json: {e}", exc_info=True)
                s3_tables = [
                    "brands", "candidates", "categories", "customers", "employees",
                    "orders", "order_items", "products", "staffs", "stocks", "stores"
                ]
                self.logger.warning(f"Fallback to default S3 table names due to error: {s3_tables}")

            for table in tables:
                if any(table.endswith(s3_table) for s3_table in s3_tables):
                    valid_tables.append(table)
                else:
                    self.logger.warning(f"Invalid S3 table in feedback: {table}")
                    return False

        return bool(valid_tables)

    async def get_feedback(self) -> List[Tuple[str, str, List[str], np.ndarray, int]]:
        """
        Retrieve all feedback entries from cache (Scenario 3).

        Returns:
            List[Tuple[str, str, List[str], np.ndarray, int]]: Feedback data
                (timestamp, query, tables, embedding, count).
        """
        try:
            feedback = await asyncio.to_thread(self.cache_synchronizer.get_feedback)
            self.logger.debug(f"Retrieved {len(feedback)} feedback entries")
            return feedback
        except Exception as e:
            self.logger.error(f"Error retrieving feedback: {e}", exc_info=True)
            return []

    async def find_similar_feedback(self, query: str, threshold: float = 0.75) -> List[Tuple[str, List[str], float]]:
        """
        Find similar feedback entries based on embedding similarity (Scenario 2).

        Uses cosine similarity with threshold ≥0.75 to ensure high-quality matches.

        Args:
            query (str): Input query.
            threshold (float): Similarity threshold (default: 0.75).

        Returns:
            List[Tuple[str, List[str], float]]: Similar queries, tables, and similarity scores.
        """
        try:
            if not self.model:
                self.logger.warning("No model available for embedding similarity")
                return []

            # Generate or retrieve query embedding
            if query in self.embedding_cache:
                query_embedding = self.embedding_cache[query]
            else:
                query_embedding = self.model.encode(query, show_progress_bar=False)
                self.embedding_cache[query] = query_embedding

            feedback = await self.get_feedback()
            similar = []

            for timestamp, fb_query, tables, fb_embedding, _ in feedback:
                similarity = float(util.cos_sim(query_embedding, fb_embedding)[0][0])
                if similarity >= threshold:
                    similar.append((fb_query, tables, similarity))

            self.logger.debug(f"Found {len(similar)} similar feedback entries for query: {query}")
            return sorted(similar, key=lambda x: x[2], reverse=True)

        except Exception as e:
            self.logger.error(f"Error finding similar feedback for query {query}: {e}", exc_info=True)
            return []

    async def update_feedback_count(self, query: str, increment: int = 1) -> None:
        """
        Update feedback count for an existing query (Scenario 8).

        Args:
            query (str): Query to update.
            increment (int): Count increment (default: 1).
        """
        try:
            await asyncio.to_thread(self.cache_synchronizer.update_feedback_count, query, increment)
            self.logger.debug(f"Updated feedback count for query: {query}, increment: {increment}")
        except Exception as e:
            self.logger.error(f"Error updating feedback count for query {query}: {e}", exc_info=True)