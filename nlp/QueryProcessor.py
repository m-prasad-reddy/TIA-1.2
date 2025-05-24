```python
"""
Processes user queries to identify tables and columns for TIA-1.2.
Uses spaCy for tokenization, sentence-transformers for embeddings, and feedback for table identification.
Supports SQL Server and S3 CSV data sources with async operations.
Preserves TIA-1.1 functionality, fixes table selection and performance issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based paths and logging
- Async operations for <5s query processing
- S3 table support
Addresses Scenarios 1 (table identification), 2 (synonym quality), 3 (feedback usage),
5 (performance), 6 (ignored queries), and 8 (deduplication).
"""
import logging
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional
from spacy.language import Language
from sentence_transformers import util
from config.manager import DatabaseConnection
from feedback.manager import FeedbackManager
from analysis.table_identifier import TableIdentifier
from analysis.name_match_manager import NameMatchManager
from analysis.processor import NLPPipeline
from config.cache_synchronizer import CacheSynchronizer
from config.model_singleton import ModelSingleton
from config.logging_setup import setup_logging

class QueryProcessor:
    """Processes user queries to identify relevant tables and columns."""
    def __init__(
        self,
        connection_manager: DatabaseConnection,
        schema_dict: Dict,
        nlp_pipeline: NLPPipeline,
        table_identifier: TableIdentifier,
        name_matcher: NameMatchManager,
        pattern_manager: 'PatternManager',
        display_name: str,
        db_name: str,
        cache_synchronizer: CacheSynchronizer
    ):
        """
        Initialize QueryProcessor with NLP and database dependencies.

        Args:
            connection_manager (DatabaseConnection): Database/S3 connection.
            schema_dict (Dict): Schema dictionary for table validation.
            nlp_pipeline (NLPPipeline): spaCy pipeline for tokenization.
            table_identifier (TableIdentifier): Table matching logic.
            name_matcher (NameMatchManager): Synonym matching.
            pattern_manager (PatternManager): Regex patterns for entities.
            display_name (str): Display name (e.g., BIKE-STORES).
            db_name (str): Database name (e.g., BikeStores).
            cache_synchronizer (CacheSynchronizer): Cache manager.
        """
        setup_logging(display_name)
        self.connection_manager = connection_manager
        self.schema_dict = schema_dict
        self.nlp_pipeline = nlp_pipeline
        self.table_identifier = table_identifier
        self.name_matcher = name_matcher
        self.pattern_manager = pattern_manager
        self.display_name = display_name
        self.db_name = db_name
        self.cache_synchronizer = cache_synchronizer
        self.model = ModelSingleton(display_name, db_name).model
        self.logger = logging.getLogger("query_processor")
        self.embedding_cache = {}  # Cache for query embeddings
        self.logger.debug(f"Initialized QueryProcessor for {display_name}/{db_name}")

    async def process_query(self, query: str) -> Tuple[List[str], List[str], float, Optional[str]]:
        """
        Process a user query to identify tables, columns, and confidence (Scenarios 1, 3, 5, 6).

        Uses spaCy for keywords, feedback for suggestions, and TableIdentifier for matching.
        Returns empty lists and reason if no tables are found.

        Args:
            query (str): User query (e.g., "stores in USA").

        Returns:
            Tuple[List[str], List[str], float, Optional[str]]: Matched tables, columns,
                confidence score, and reason (if no matches).
        """
        try:
            if not query:
                self.logger.warning("Empty query provided")
                return [], [], 0.0, "empty_query"

            # Extract keywords
            keywords = self._extract_keywords(query)
            if not keywords:
                self.logger.warning(f"No valid keywords extracted from query: {query}")
                return [], [], 0.0, "no_valid_keywords"

            # Check cache
            cache_key = f"{query}:{','.join(keywords)}"
            cached_result = await asyncio.to_thread(self.cache_synchronizer.get_cached_result, cache_key)
            if cached_result:
                tables, columns, confidence = cached_result
                self.logger.debug(f"Cache hit for query: {query}, tables: {tables}")
                return tables, columns, confidence, None

            # Get similar feedback (Scenario 3)
            feedback_tables = await self._get_similar_feedback(query)
            if feedback_tables:
                self.logger.debug(f"Feedback tables for query {query}: {feedback_tables}")

            # Identify tables and columns (Scenario 1)
            tables, columns, confidence = await asyncio.to_thread(
                self.table_identifier.process,
                query,
                keywords,
                self.schema_dict,
                feedback_tables
            )

            if not tables:
                reason = "no_matching_tables"
                self.logger.warning(f"No tables matched for query: {query}, reason: {reason}")
                return [], [], 0.0, reason

            # Cache result
            await asyncio.to_thread(
                self.cache_synchronizer.cache_result,
                cache_key,
                (tables, columns, confidence)
            )
            self.logger.info(f"Processed query: {query}, tables: {tables}, confidence: {confidence}")
            return tables, columns, confidence, None

        except Exception as e:
            self.logger.error(f"Error processing query {query}: {e}", exc_info=True)
            return [], [], 0.0, f"error: {str(e)}"

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract non-stopword tokens from query using spaCy (Scenario 5).

        Args:
            query (str): User query.

        Returns:
            List[str]: Extracted keywords.
        """
        try:
            doc = self.nlp_pipeline.nlp(query)
            keywords = [
                token.text.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and token.text.strip()
            ]
            self.logger.debug(f"Extracted keywords from query '{query}': {keywords}")
            return keywords
        except Exception as e:
            self.logger.error(f"Error extracting keywords from query {query}: {e}", exc_info=True)
            return []

    async def _get_similar_feedback(self, query: str, threshold: float = 0.75) -> List[str]:
        """
        Retrieve similar feedback tables using embedding similarity (Scenarios 2, 3).

        Uses cosine similarity with threshold ≥0.75 for high-quality matches.

        Args:
            query (str): User query.
            threshold (float): Similarity threshold (default: 0.75).

        Returns:
            List[str]: Tables from similar feedback.
        """
        try:
            if not self.model:
                self.logger.warning("No model available for feedback similarity")
                return []

            # Generate or retrieve query embedding
            if query in self.embedding_cache:
                query_embedding = self.embedding_cache[query]
                self.logger.debug(f"Retrieved cached embedding for query: {query}")
            else:
                query_embedding = self.model.encode(query, show_progress_bar=False)
                self.embedding_cache[query] = query_embedding
                self.logger.debug(f"Generated embedding for query: {query}")

            # Get feedback from FeedbackManager
            feedback = await self.table_identifier.feedback_manager.find_similar_feedback(query, threshold)
            if not feedback:
                self.logger.debug(f"No similar feedback found for query: {query}")
                return []

            # Aggregate tables from high-similarity feedback
            tables = []
            for fb_query, fb_tables, similarity in feedback:
                if similarity >= threshold:
                    tables.extend(fb_tables)
            tables = list(set(tables))  # Remove duplicates
            self.logger.debug(f"Similar feedback tables for query '{query}': {tables}")
            return tables

        except Exception as e:
            self.logger.error(f"Error retrieving similar feedback for query {query}: {e}", exc_info=True)
            return []
```