"""
Manages column synonyms and name matches for query processing in TIA-1.2.
Uses spaCy for tokenization and sentence-transformers for embedding-based similarity.
Stores synonyms in cache.db via CacheSynchronizer, supporting SQL Server and S3 CSV.
Preserves TIA-1.1 functionality, fixes synonym quality and performance issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based paths (e.g., BIKE-STORES/BikeStores/cache.db)
- Async cache operations for <5s query processing
- Logging to codebase/logs/DISPLAY_NAME_app.log
- Non-hardcoding with config_name
- Higher synonym threshold (≥0.75) for quality
Addresses Scenarios 1 (table identification), 2 (synonym quality), 5 (performance),
and 8 (feedback deduplication).
"""
import os
import json
import logging
import asyncio
from typing import Dict, List, Optional
import spacy
import numpy as np
from sentence_transformers import util
from config.model_singleton import ModelSingleton
from config.cache_synchronizer import CacheSynchronizer
from config.logging_setup import setup_logging

class NameMatchManager:
    """Manages name matches and synonyms for query processing in TIA-1.2."""
    def __init__(self, display_name: str, db_name: str, config_name: str = "bikestores"):
        """
        Initialize NameMatchManager with spaCy, embeddings, and cache.

        Args:
            display_name (str): Display name (e.g., BIKE-STORES).
            db_name (str): Database name (e.g., BikeStores).
            config_name (str): Configuration name (e.g., bikestores).
        """
        setup_logging(display_name)
        self.logger = logging.getLogger("name_match_manager")
        self.display_name = display_name
        self.db_name = db_name
        self.config_name = config_name
        self.cache_synchronizer = CacheSynchronizer(display_name, db_name, config_name)
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self.logger.debug("Loaded spaCy model: en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {e}", exc_info=True)
            raise RuntimeError("Spacy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
        self.model = ModelSingleton(config_name).model
        self.default_matches = self.cache_synchronizer.read_name_matches('default') or {}
        self.dynamic_matches = self.cache_synchronizer.read_name_matches('dynamic') or {}
        self.synonym_threshold = 0.75  # Increased for quality (Scenario 2)
        self.embedding_cache = {}  # Cache embeddings with size limit
        self.embedding_cache_limit = 1000  # Limit cache size
        self.predefined_synonyms = {
            "stock": ["quantity", "inventory"],
            "availability": ["quantity", "inventory"],
            "names": ["first_name", "last_name"],
            "name": ["first_name", "last_name", "product_name", "store_name", "brand_name", "category_name"],
            "customer": ["customer_id"],
            "product": ["product_id"],
            "store": ["store_id", "store_name"],
            "category": ["category_name"],
            "order": ["order_id"],
            "brand": ["brand_name"]  # Added for better matching (Scenario 2)
        }
        self.suggested_synonyms = set()  # Track suggested synonyms to avoid repeat prompts
        self.logger.debug(f"Initialized NameMatchManager for {display_name}/{db_name} (config: {config_name})")

    async def process_query(self, query: str, schema_dict: Dict) -> Dict[str, float]:
        """
        Process query to identify column matches and return scores (Scenarios 1, 2, 5).

        Args:
            query (str): User query (e.g., "stores in USA").
            schema_dict (Dict): Schema dictionary with tables and columns.

        Returns:
            Dict[str, float]: Column paths (schema.table.column) with scores.
        """
        try:
            if not schema_dict or 'tables' not in schema_dict or not schema_dict['tables']:
                self.logger.error("Invalid or empty schema dictionary provided")
                return {}

            # Tokenize and extract relevant tokens
            doc = await asyncio.to_thread(self.nlp, query.lower())
            tokens = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'VERB', 'ADJ') and not token.is_stop]
            self.logger.debug(f"Extracted tokens: {tokens}")
            if not tokens:
                self.logger.debug("No relevant tokens found in query")
                return {}

            # Generate embeddings for tokens
            token_embeddings = [self._get_embedding(token) for token in tokens]
            self.logger.debug(f"Generated embeddings for {len(tokens)} tokens")

            # Initialize synonym dictionary and unmatched tokens set
            synonyms = {}
            unmatched_tokens = set(tokens)

            # Iterate through schema to find matches
            system_schemas = ['dbo', 'sys', 'information_schema']
            for schema in schema_dict['tables']:
                if schema.lower() in system_schemas:
                    self.logger.debug(f"Skipping system schema: {schema}")
                    continue
                for table in schema_dict['tables'][schema]:
                    for col in schema_dict['columns'][schema][table]:
                        col_lower = col.lower()
                        col_embedding = self._get_embedding(col_lower).reshape(1, -1)

                        # Check predefined and cached matches
                        for syn_key, syn_list in self.predefined_synonyms.items():
                            if syn_key in unmatched_tokens and col_lower in syn_list:
                                unmatched_tokens.discard(syn_key)
                                synonyms.setdefault(col_lower, []).append(syn_key)
                                self.logger.debug(f"Matched predefined synonym '{syn_key}' for '{col_lower}'")
                        if col_lower in self.default_matches:
                            for syn in self.default_matches[col_lower]:
                                if syn in unmatched_tokens:
                                    unmatched_tokens.discard(syn)
                                    synonyms.setdefault(col_lower, []).append(syn)
                                    self.logger.debug(f"Matched default synonym '{syn}' for '{col_lower}'")
                        if col_lower in self.dynamic_matches:
                            for syn in self.dynamic_matches[col_lower]:
                                if syn in unmatched_tokens:
                                    unmatched_tokens.discard(syn)
                                    synonyms.setdefault(col_lower, []).append(syn)
                                    self.logger.debug(f"Matched dynamic synonym '{syn}' for '{col_lower}'")

                        # Check token similarity for new synonyms
                        for token, token_emb in zip(tokens, token_embeddings):
                            if token not in unmatched_tokens:
                                continue
                            token_emb = token_emb.reshape(1, -1)
                            similarity = util.cos_sim(token_emb, col_embedding)[0][0]
                            if similarity > self.synonym_threshold:
                                suggested = await self._suggest_synonym(token, col, schema, table)
                                if suggested:
                                    synonyms.setdefault(col_lower, []).append(token)
                                    unmatched_tokens.discard(token)
                                    self.dynamic_matches.setdefault(col_lower, []).append(token)
                                    await self._save_dynamic_matches()
                                    self.logger.info(f"Added synonym '{token}' for '{col_lower}'")
                                else:
                                    conflict_col = self._find_conflict_column(
                                        schema_dict, schema, table, col_lower, token, token_emb
                                    )
                                    if conflict_col:
                                        self.logger.debug(f"Synonym conflict for '{token}' with '{conflict_col}'")
                                        synonyms.setdefault(conflict_col.lower(), []).append(token)
                                        unmatched_tokens.discard(token)
                                        self.dynamic_matches.setdefault(conflict_col.lower(), []).append(token)
                                        await self._save_dynamic_matches()

            # Log results
            for col, syn_list in synonyms.items():
                self.logger.debug(f"Synonyms for '{col}': {syn_list}")
            if unmatched_tokens:
                self.logger.debug(f"Unmatched tokens: {list(unmatched_tokens)}")

            # Score columns based on synonyms
            scores = self._score_columns(schema_dict, synonyms)
            return scores
        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {e}", exc_info=True)
            return {}

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get or compute embedding for text, using cache with size limit (Scenario 5).

        Args:
            text (str): Text to embed.

        Returns:
            np.ndarray: Embedding array.
        """
        try:
            if text not in self.embedding_cache:
                if len(self.embedding_cache) >= self.embedding_cache_limit:
                    self.embedding_cache.pop(next(iter(self.embedding_cache)))  # Remove oldest
                    self.logger.debug(f"Removed oldest embedding from cache (limit: {self.embedding_cache_limit})")
                self.embedding_cache[text] = self.model.encode(text, show_progress_bar=False)
                self.logger.debug(f"Cached embedding for '{text}' (cache size: {len(self.embedding_cache)})")
            else:
                self.logger.debug(f"Embedding cache hit for '{text}'")
            return self.embedding_cache[text]
        except Exception as e:
            self.logger.error(f"Error generating embedding for '{text}': {e}", exc_info=True)
            return np.zeros(384, dtype=np.float32)

    def compute_similarity(self, token: str, column: str) -> float:
        """
        Compute cosine similarity between token and column (Scenario 2).

        Args:
            token (str): Query token.
            column (str): Column name.

        Returns:
            float: Cosine similarity score.
        """
        try:
            token_emb = self._get_embedding(token)
            col_emb = self._get_embedding(column)
            similarity = float(util.cos_sim(token_emb, col_emb)[0][0])
            return similarity
        except Exception as e:
            self.logger.error(f"Error computing similarity between '{token}' and '{column}': {e}", exc_info=True)
            return 0.0

    async def _suggest_synonym(self, token: str, column: str, schema: str, table: str) -> bool:
        """
        Prompt user to confirm a synonym, only once per session (Scenario 2).

        Args:
            token (str): Query token.
            column (str): Column name.
            schema (str): Schema name.
            table (str): Table name.

        Returns:
            bool: True if synonym confirmed, False otherwise.
        """
        try:
            synonym_key = f"{token}:{schema}.{table}.{column}"
            if synonym_key in self.suggested_synonyms:
                self.logger.debug(f"Skipping repeated synonym prompt for '{token}' and '{column}'")
                return False
            # Simulate async user input (non-blocking)
            print(f"\nIs '{token}' a synonym for column '{column}' in table '{schema}.{table}'? (y/n)")
            response = await asyncio.to_thread(input)
            response = response.strip().lower()
            self.suggested_synonyms.add(synonym_key)
            if response == 'y':
                self.logger.debug(f"User confirmed synonym '{token}' for '{column}'")
                return True
            self.logger.debug(f"User rejected synonym '{token}' for '{column}'")
            return False
        except Exception as e:
            self.logger.error(f"Error prompting for synonym '{token}': {e}", exc_info=True)
            return False

    def _find_conflict_column(self, schema_dict: Dict, schema: str, table: str, current_col: str, token: str, token_emb: np.ndarray) -> Optional[str]:
        """
        Find a conflicting column with higher similarity to the token (Scenario 2).

        Args:
            schema_dict (Dict): Schema dictionary.
            schema (str): Schema name.
            table (str): Table name.
            current_col (str): Current column name.
            token (str): Query token.
            token_emb (np.ndarray): Token embedding.

        Returns:
            Optional[str]: Conflicting column name, or None.
        """
        try:
            max_similarity = self.synonym_threshold
            conflict_col = None
            for other_col in schema_dict['columns'][schema][table]:
                if other_col.lower() != current_col:
                    other_embedding = self._get_embedding(other_col.lower()).reshape(1, -1)
                    other_similarity = util.cos_sim(token_emb, other_embedding)[0][0]
                    if other_similarity > max_similarity:
                        max_similarity = other_similarity
                        conflict_col = other_col
            if conflict_col:
                self.logger.debug(f"Found conflict column '{conflict_col}' for token '{token}' (sim={max_similarity:.2f})")
            return conflict_col
        except Exception as e:
            self.logger.error(f"Error finding conflict column for token '{token}': {e}", exc_info=True)
            return None

    def _score_columns(self, schema_dict: Dict, synonyms: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Score columns based on matched synonyms (Scenarios 1, 2).

        Args:
            schema_dict (Dict): Schema dictionary.
            synonyms (Dict[str, List[str]]): Column-to-synonym mappings.

        Returns:
            Dict[str, float]: Column paths with scores.
        """
        try:
            scores = {}
            system_schemas = ['dbo', 'sys', 'information_schema']
            for schema in schema_dict['tables']:
                if schema.lower() in system_schemas:
                    continue
                for table in schema_dict['tables'][schema]:
                    for col in schema_dict['columns'][schema][table]:
                        col_lower = col.lower()
                        score = 0.0
                        if col_lower in synonyms:
                            score += 0.2 * len(synonyms[col_lower])  # Base score for synonyms
                            matched_synonyms = synonyms[col_lower]
                            if any(syn in self.predefined_synonyms for syn in matched_synonyms):
                                score += 0.1  # Boost for predefined synonyms
                            if any(syn in self.dynamic_matches.get(col_lower, []) for syn in matched_synonyms):
                                score += 0.15  # Boost for dynamic synonyms
                            if col_lower.endswith('_id'):
                                score += 0.05  # Boost for ID columns (Scenario 1)
                            if col_lower in schema_dict.get('primary_keys', {}).get(schema, {}).get(table, []):
                                score += 0.1  # Boost for primary keys (Scenario 1)
                            self.logger.debug(f"Scoring '{col_lower}': synonyms={matched_synonyms}, score={score:.2f}")
                        scores[f"{schema}.{table}.{col_lower}"] = score
                        self.logger.debug(f"Column score for '{col_lower}': {score:.2f}")
            return scores
        except Exception as e:
            self.logger.error(f"Error scoring columns: {e}", exc_info=True)
            return {}

    async def _save_dynamic_matches(self):
        """
        Save dynamic matches to SQLite asynchronously (Scenario 5).

        Deduplicates synonyms before saving.
        """
        try:
            for col in self.dynamic_matches:
                self.dynamic_matches[col] = list(set(self.dynamic_matches[col]))  # Deduplicate
            await asyncio.to_thread(self.cache_synchronizer.write_name_matches, self.dynamic_matches, 'dynamic')
            self.logger.debug("Saved dynamic matches to SQLite")
        except Exception as e:
            self.logger.error(f"Error saving dynamic matches: {e}", exc_info=True)
            raise

    async def save_matches(self):
        """
        Save all matches to SQLite asynchronously (Scenario 5).

        Deduplicates default and dynamic synonyms before saving.
        """
        try:
            for col in self.default_matches:
                self.default_matches[col] = list(set(self.default_matches[col]))  # Deduplicate
            await asyncio.to_thread(self.cache_synchronizer.write_name_matches, self.default_matches, 'default')
            await self._save_dynamic_matches()
            self.logger.debug("Saved all name matches to SQLite")
        except Exception as e:
            self.logger.error(f"Error saving name matches: {e}", exc_info=True)
            raise

    def suggest_synonyms(self, column: str, schema_dict: Dict) -> List[str]:
        """
        Suggest synonyms for a given column based on schema similarity (Scenario 2).

        Args:
            column (str): Column name.
            schema_dict (Dict): Schema dictionary.

        Returns:
            List[str]: Suggested synonyms.
        """
        try:
            col_lower = column.lower()
            col_embedding = self._get_embedding(col_lower).reshape(1, -1)
            suggestions = []
            for schema in schema_dict['tables']:
                for table in schema_dict['tables'][schema]:
                    table_embedding = self._get_embedding(table.lower()).reshape(1, -1)
                    similarity = util.cos_sim(col_embedding, table_embedding)[0][0]
                    if similarity > self.synonym_threshold - 0.1:
                        suggestions.append(table.lower())
                    for other_col in schema_dict['columns'][schema][table]:
                        if other_col.lower() != col_lower:
                            other_embedding = self._get_embedding(other_col.lower()).reshape(1, -1)
                            similarity = util.cos_sim(col_embedding, other_embedding)[0][0]
                            if similarity > self.synonym_threshold:
                                suggestions.append(other_col.lower())
            suggestions = list(set(suggestions))
            self.logger.debug(f"Suggested synonyms for '{col_lower}': {suggestions}")
            return suggestions
        except Exception as e:
            self.logger.error(f"Error suggesting synonyms for '{column}': {e}", exc_info=True)
            return []

    def extract_tokens(self, query: str) -> List[str]:
        """
        Extract relevant lemmatized tokens from query (Scenario 2).

        Args:
            query (str): User query.

        Returns:
            List[str]: Lemmatized tokens.
        """
        try:
            doc = self.nlp(query.lower())
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            self.logger.debug(f"Extracted tokens: {tokens}")
            return tokens
        except Exception as e:
            self.logger.error(f"Error extracting tokens from query '{query}': {e}", exc_info=True)
            return []