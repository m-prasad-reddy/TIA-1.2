"""
Identifies tables from user queries for TIA-1.2.
Combines feedback, patterns, semantic similarity, weights, and name matches with typo correction and feedback cleaning.
Supports SQL Server and S3 CSV data sources.
Preserves TIA-1.1 functionality, fixes table selection and performance issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based paths (e.g., BIKE-STORES/BikeStores/cache.db)
- Async cache and feedback operations for <5s query processing
- Logging to codebase/logs/DISPLAY_NAME_app.log
- Non-hardcoding with config_name
- S3 table support (e.g., stores.csv)
Addresses Scenarios 1 (table identification), 2 (synonym quality), 5 (performance), and 8 (feedback deduplication).
"""
import logging
import asyncio
import os
import re
from typing import Dict, List, Tuple, Optional
import spacy
import numpy as np
from sentence_transformers import util
from config.model_singleton import ModelSingleton
from config.patterns import PatternManager
from feedback.manager import FeedbackManager
from config.cache_synchronizer import CacheSynchronizer
from config.logging_setup import setup_logging
from datetime import datetime, timedelta

class TableIdentifier:
    """Identifies relevant tables from queries using feedback, patterns, similarity, weights, and synonyms."""
    def __init__(self, schema_dict: Dict, feedback_manager: FeedbackManager, 
                 pattern_manager: PatternManager, cache_synchronizer: CacheSynchronizer, 
                 display_name: str, config_name: str = "bikestores"):
        """
        Initialize TableIdentifier with dependencies and models.

        Args:
            schema_dict (Dict): Schema dictionary with tables and columns.
            feedback_manager (FeedbackManager): Feedback manager instance.
            pattern_manager (PatternManager): Pattern manager for regex patterns.
            cache_synchronizer (CacheSynchronizer): Cache synchronizer instance.
            display_name (str): Display name (e.g., BIKE-STORES).
            config_name (str): Configuration name (e.g., bikestores).
        """
        setup_logging(display_name)
        self.logger = logging.getLogger("table_identifier")
        self.display_name = display_name
        self.config_name = config_name
        self.schema_dict = schema_dict
        self.feedback_manager = feedback_manager
        self.pattern_manager = pattern_manager
        self.cache_synchronizer = cache_synchronizer
        self.model = ModelSingleton(config_name).model
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self.logger.debug("Loaded spaCy model: en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {e}", exc_info=True)
            raise RuntimeError("Spacy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
        self.weights = self.cache_synchronizer.load_weights() or {}
        self.name_matches = self.cache_synchronizer.read_name_matches('dynamic') or {}
        self.similarity_threshold = 0.75  # Increased for quality (Scenario 2)
        self.max_confidence = 0.95
        self.last_decay = datetime.now()
        self.logger.debug(f"Initialized TableIdentifier for {display_name} (config: {config_name})")

    def _decay_weights(self):
        """
        Decay weights weekly for low-confidence matches (Scenario 5).
        """
        try:
            if datetime.now() - self.last_decay > timedelta(days=7):
                for table, cols in self.weights.items():
                    for col, weight in cols.items():
                        self.weights[table][col] = max(0.05, weight * 0.9)
                asyncio.run(self.cache_synchronizer.write_weights(self.weights))
                self.last_decay = datetime.now()
                self.logger.debug("Applied weight decay")
        except Exception as e:
            self.logger.error(f"Error decaying weights: {e}", exc_info=True)

    async def check_duplicate_feedback(self, query: str, tables: List[str]) -> Tuple[bool, Optional[str], Optional[List[str]]]:
        """
        Check for duplicate feedback (Scenario 8).

        Args:
            query (str): User query.
            tables (List[str]): Identified tables.

        Returns:
            Tuple[bool, Optional[str], Optional[List[str]]]: (is_duplicate, feedback_id, existing_tables).
        """
        try:
            normalized_query = self.cache_synchronizer.normalize_query(query)
            similar_feedback = await self.cache_synchronizer.find_similar_feedback(normalized_query, threshold=0.95)
            for fb_query, fb_tables, sim in similar_feedback:
                if sim > 0.95:
                    fb_id = next(ts for ts, data in self.cache_synchronizer.read_feedback().items() if data['query'] == fb_query)
                    self.logger.debug(f"Found duplicate: '{normalized_query}' ~ '{fb_query}' (sim={sim:.2f})")
                    return True, fb_id, fb_tables
            return False, None, None
        except Exception as e:
            self.logger.error(f"Error checking duplicate feedback: {e}", exc_info=True)
            return False, None, None

    async def clean_feedback_cache(self) -> None:
        """
        Remove feedback entries with invalid table names asynchronously (Scenario 8).
        """
        try:
            feedback = self.cache_synchronizer.get_feedback()
            valid_feedback = []
            all_tables = {f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]}

            for timestamp, query, tables, embedding, count in feedback:
                valid = True
                for table in tables:
                    if table not in all_tables:
                        self.logger.debug(f"Removing feedback with invalid table: '{table}' in query '{query}'")
                        valid = False
                        break
                if valid:
                    valid_feedback.append((timestamp, query, tables, embedding, count))
                else:
                    await self.cache_synchronizer.delete_feedback(query)

            self.logger.debug(f"Cleaned feedback cache, retained {len(valid_feedback)} valid entries")
        except Exception as e:
            self.logger.error(f"Error cleaning feedback cache: {e}", exc_info=True)
            raise

    def correct_table_typo(self, table: str) -> str:
        """
        Correct common table name typos (Scenario 1).

        Args:
            table (str): Table name.

        Returns:
            str: Corrected table name.
        """
        try:
            if table.endswith(".order_item"):
                corrected = table[:-5] + "items"
                self.logger.debug(f"Corrected typo: '{table}' -> '{corrected}'")
                return corrected
            return table
        except Exception as e:
            self.logger.error(f"Error correcting table typo: {e}", exc_info=True)
            return table

    def validate_tables(self, tables: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate if tables exist in schema and correct typos (Scenario 1).

        Args:
            tables (List[str]): List of table names.

        Returns:
            Tuple[List[str], List[str]]: (valid_tables, invalid_tables).
        """
        try:
            valid, invalid = [], []
            all_tables = {f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]}
            for table in tables:
                if table in all_tables:
                    valid.append(table)
                    continue
                corrected_table = self.correct_table_typo(table)
                if corrected_table in all_tables:
                    self.logger.debug(f"Corrected table name: '{table}' -> '{corrected_table}'")
                    valid.append(corrected_table)
                else:
                    invalid.append(table)
                    self.logger.debug(f"Invalid table: '{table}'")
            self.logger.debug(f"Validated tables: valid={valid}, invalid={invalid}")
            return valid, invalid
        except Exception as e:
            self.logger.error(f"Error validating tables: {e}", exc_info=True)
            return [], tables

    async def identify_tables(self, query: str, column_scores: Dict[str, float]) -> Tuple[List[str], float]:
        """
        Identify tables using feedback, patterns, similarity, weights, and custom rules (Scenarios 1, 2, 5, 8).

        Args:
            query (str): User query.
            column_scores (Dict[str, float]): Column scores from NameMatchManager.

        Returns:
            Tuple[List[str], float]: Identified tables and confidence score.
        """
        try:
            self.logger.debug(f"Processing query: {query}")
            self._decay_weights()
            query_lower = self.preprocess_query(query)
            doc = await asyncio.to_thread(self.nlp, query_lower)
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
            table_scores: Dict[str, float] = {}
            match_details = []

            # Clean feedback cache
            await self.clean_feedback_cache()

            # Short-circuit for high-similarity feedback
            if self.feedback_manager:
                similar_feedback = await self.cache_synchronizer.find_similar_feedback(query, threshold=0.9)  # Lowered (Scenario 1)
                for fb_query, fb_tables, sim in similar_feedback:
                    valid_tables, invalid_tables = self.validate_tables(fb_tables)
                    if invalid_tables:
                        self.logger.debug(f"Invalid tables in feedback: {invalid_tables}")
                        continue
                    if sim > 0.9 and valid_tables:
                        self.logger.debug(f"Using feedback: '{fb_query}' -> {valid_tables} (sim={sim:.2f})")
                        return valid_tables, 0.95
                    if sim > 0.7:
                        for table_full in valid_tables:
                            table_scores[table_full] = table_scores.get(table_full, 0) + sim * 4.0
                            match_details.append(f"Feedback: '{fb_query}' -> '{table_full}' (sim={sim:.2f}, score=+{sim*4.0:.2f})")
                if similar_feedback:
                    self.logger.debug(f"Found {len(similar_feedback)} similar feedback entries")

            # Exact table name matching
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    if table.lower() in query_lower:
                        table_full = f"{schema}.{table}"
                        table_scores[table_full] = table_scores.get(table_full, 0) + 0.9
                        match_details.append(f"Exact match: '{table}' -> '{table_full}' (+0.9)")

            # Pattern-based matching
            patterns = self.pattern_manager.get_patterns()
            for pattern, tables in patterns.items():
                if re.search(pattern, query_lower):
                    valid_tables, _ = self.validate_tables(tables)
                    for table_full in valid_tables:
                        table_scores[table_full] = table_scores.get(table_full, 0) + 0.7
                        match_details.append(f"Pattern: '{pattern}' -> '{table_full}' (+0.7)")

            # Semantic matching
            query_embedding = await asyncio.to_thread(self.model.encode, query_lower, show_progress_bar=False)
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    table_full = f"{schema}.{table}"
                    table_embedding = await asyncio.to_thread(self.model.encode, table.lower(), show_progress_bar=False)
                    similarity = float(util.cos_sim(query_embedding, table_embedding)[0][0])
                    if similarity > self.similarity_threshold:
                        table_scores[table_full] = table_scores.get(table_full, 0) + similarity
                        match_details.append(f"Semantic: '{table}' (sim={similarity:.2f}) -> '{table_full}' (+{similarity:.2f})")

            # Weight-based matching
            for table_full, cols in self.weights.items():
                if table_full not in [f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                    continue
                for col, weight in cols.items():
                    col_lower = col.lower()
                    if col_lower in query_lower or any(token == col_lower for token in tokens):
                        table_scores[table_full] = table_scores.get(table_full, 0) + weight
                        match_details.append(f"Weight: '{col_lower}' in '{table_full}' (weight={weight})")
                    for synonym in self.name_matches.get(col_lower, []):
                        if any(token == synonym.lower() for token in tokens):
                            table_scores[table_full] = table_scores.get(table_full, 0) + weight
                            match_details.append(f"Synonym: '{synonym}' -> '{col_lower}' in '{table_full}' (weight={weight})")

            # Name match lookup
            for col, synonyms in self.name_matches.items():
                col_lower = col.lower()
                for synonym in synonyms:
                    if synonym.lower() in query_lower or any(token == synonym.lower() for token in tokens):
                        for schema in self.schema_dict['columns']:
                            for table, cols in self.schema_dict['columns'][schema].items():
                                if col_lower in [c.lower() for c in cols]:
                                    table_full = f"{schema}.{table}"
                                    table_scores[table_full] = table_scores.get(table_full, 0) + 0.6
                                    match_details.append(f"Name match: '{synonym}' -> '{col_lower}' in '{table_full}' (+0.6)")

            # Column score-based matching
            for column_key, score in column_scores.items():
                if score > 0:
                    try:
                        schema, table, col = column_key.split('.')
                        table_full = f"{schema}.{table}"
                        if table_full in [f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                            table_scores[table_full] = table_scores.get(table_full, 0) + score * 0.5
                            match_details.append(f"Column: '{column_key}' (score={score}) -> '{table_full}' (+{score*0.5:.2f})")
                    except ValueError:
                        self.logger.debug(f"Invalid column key: '{column_key}'")

            # Custom rules for stock/availability
            if any(token in ['stock', 'availability', 'stocks', 'quantities'] for token in tokens):
                table_full = 'production.stocks'
                if table_full in [f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                    table_scores[table_full] = table_scores.get(table_full, 0) + 0.85
                    match_details.append("Custom: 'stock/availability' -> 'production.stocks' (+0.85)")
                table_full = 'sales.stores'
                if table_full in [f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                    table_scores[table_full] = table_scores.get(table_full, 0) + 0.75
                    match_details.append("Custom: 'stock/availability' -> 'sales.stores' (+0.75)")

            # S3-specific rules (Scenario 1)
            if any(token in ['store', 'stores'] for token in tokens):
                table_full = 'stores'  # S3 table (stores.csv)
                if table_full in [t for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                    table_scores[table_full] = table_scores.get(table_full, 0) + 0.85
                    match_details.append("S3 Custom: 'store/stores' -> 'stores' (+0.85)")

            # Prune to top tables
            ranked_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
            selected_tables = [table for table, score in ranked_tables if score > 0.7][:4]
            confidence = min(sum(score for _, score in ranked_tables[:4]) / 4, self.max_confidence) if ranked_tables else 0.0

            # Fallback to feedback history
            if not selected_tables and self.feedback_manager:
                feedback = self.feedback_manager.get_all_feedback()
                normalized_query = self.cache_synchronizer.normalize_query(query_lower)
                for entry in feedback:
                    if normalized_query == entry['query'].lower():
                        valid_tables, _ = self.validate_tables(entry['tables'])
                        if valid_tables:
                            selected_tables = valid_tables
                            confidence = max(confidence, 0.9)
                            match_details.append(f"Feedback: '{query}' -> {selected_tables} (+3.0)")
                            break

            for detail in match_details:
                self.logger.debug(detail)
            if selected_tables:
                self.logger.debug(f"Final tables: {selected_tables}, confidence={confidence:.2f}")
            else:
                self.logger.debug("No tables identified for query")
            return selected_tables, confidence
        except Exception as e:
            self.logger.error(f"Error identifying tables: {e}", exc_info=True)
            return [], 0.0

    async def update_weights_from_feedback(self, query: str, tables: List[str]):
        """
        Update weights based on feedback, skip or merge duplicates (Scenario 8).

        Args:
            query (str): User query.
            tables (List[str]): Feedback tables.
        """
        try:
            valid_tables, invalid_tables = self.validate_tables(tables)
            if invalid_tables:
                self.logger.debug(f"Invalid tables in feedback: {invalid_tables}, cleaning cache")
                await self.clean_feedback_cache()
                return

            is_duplicate, _, _ = await self.check_duplicate_feedback(query, valid_tables)
            if is_duplicate:
                self.logger.debug("Skipped weight update for duplicate feedback")
                return

            all_tables = {f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]}
            for table_full in all_tables:
                schema, table_name = table_full.split('.')
                weight_change = 0.02 if table_full in valid_tables else -0.15
                if table_full not in self.weights:
                    self.weights[table_full] = {}
                for col in self.schema_dict['columns'].get(schema, {}).get(table_name, []):
                    col_lower = col.lower()
                    self.weights[table_full][col_lower] = min(max(0.05, self.weights[table_full].get(col_lower, 0.05) + weight_change), 0.8)
                    self.logger.debug(f"{'Increased' if weight_change > 0 else 'Decreased'} weight for '{col_lower}' in '{table_full}' to {self.weights[table_full][col_lower]:.2f}")
            await self.cache_synchronizer.write_weights(self.weights)
            await self.cache_synchronizer.delete_ignored_query(query)
            self.logger.debug("Weights updated and query removed from ignored_queries")
        except Exception as e:
            self.logger.error(f"Error updating weights: {e}", exc_info=True)

    async def update_name_matches(self, column: str, synonyms: List[str]):
        """
        Update name matches with new synonyms asynchronously (Scenario 2).

        Args:
            column (str): Column name.
            synonyms (List[str]): List of synonyms.
        """
        try:
            column_lower = column.lower()
            if column_lower not in self.name_matches:
                self.name_matches[column_lower] = []
            self.name_matches[column_lower].extend(syn.lower() for syn in synonyms if syn.lower() not in self.name_matches[column_lower])
            self.name_matches[column_lower] = list(set(self.name_matches[column_lower]))
            await self.cache_synchronizer.write_name_matches(self.name_matches, 'dynamic')
            self.logger.debug(f"Updated name matches for '{column_lower}': {self.name_matches[column_lower]}")
        except Exception as e:
            self.logger.error(f"Error updating name matches: {e}", exc_info=True)

    async def save_name_matches(self):
        """
        Save name matches to cache asynchronously (Scenario 2).
        """
        try:
            await self.cache_synchronizer.write_name_matches(self.name_matches, 'dynamic')
            self.logger.debug("Saved name matches")
        except Exception as e:
            self.logger.error(f"Error saving name matches: {e}", exc_info=True)

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query by normalizing and lemmatizing (Scenario 1).

        Args:
            query (str): User query.

        Returns:
            str: Processed query.
        """
        try:
            query_clean = re.sub(r'\s+', ' ', query.strip().lower())
            doc = self.nlp(query_clean)
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            processed_query = ' '.join(tokens)
            self.logger.debug(f"Preprocessed query: {query} -> {processed_query}")
            return processed_query if processed_query else query_clean
        except Exception as e:
            self.logger.error(f"Error preprocessing query: {e}", exc_info=True)
            return query.lower()