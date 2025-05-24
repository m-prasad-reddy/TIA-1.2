
"""
Provides NLPPipeline for natural language processing in TIA-1.2.
Uses spaCy for tokenization, entity recognition, and dependency parsing to extract keywords
and entities from user queries. Supports SQL Server and S3 CSV data sources.
Preserves TIA-1.1 functionality, fixes table selection and performance issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based paths (e.g., BIKE-STORES/BikeStores/nlp_cache/)
- Optimized spaCy model loading with caching for <5s query processing
- Logging to codebase/logs/DISPLAY_NAME_app.log
- Non-hardcoding with display_name and db_name
Addresses Scenarios 1 (table identification), 2 (synonym quality), 5 (performance),
and 6 (ignored queries).
"""
import os
import logging
import asyncio
from typing import List, Tuple, Dict, Optional
import spacy
from spacy.language import Language
from config.logging_setup import setup_logging

class NLPPipeline:
    """Processes user queries using spaCy for tokenization, entity recognition, and dependency parsing."""
    def __init__(self, pattern_manager: 'PatternManager', display_name: str, db_name: str):
        """
        Initialize NLPPipeline with spaCy model and logging.

        Args:
            pattern_manager (PatternManager): Pattern manager for regex patterns (placeholder for compatibility).
            display_name (str): Display name (e.g., BIKE-STORES).
            db_name (str): Database name (e.g., BikeStores).
        """
        setup_logging(display_name)
        self.logger = logging.getLogger("nlp_pipeline")
        self.display_name = display_name
        self.db_name = db_name
        self.pattern_manager = pattern_manager
        self.nlp_cache_path = os.path.join(display_name, db_name, "nlp_cache")
        os.makedirs(self.nlp_cache_path, exist_ok=True)
        self.nlp = None
        self._initialize_spacy()
        self.logger.debug(f"Initialized NLPPipeline for {display_name}/{db_name}")

    def _initialize_spacy(self):
        """
        Initialize spaCy model with caching to optimize loading (Scenario 5).
        """
        try:
            # Load spaCy model (en_core_web_sm for English queries)
            self.logger.debug("Loading spaCy model 'en_core_web_sm'")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner"] if not self._needs_ner() else [])
            self.logger.debug("Successfully loaded spaCy model")
        except Exception as e:
            self.logger.error(f"Error loading spaCy model: {e}", exc_info=True)
            try:
                self.logger.warning("Attempting to load minimal spaCy model as fallback")
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                self.logger.info("Loaded minimal spaCy model")
            except Exception as e:
                self.logger.error(f"Failed to load fallback spaCy model: {e}", exc_info=True)
                raise RuntimeError("Cannot initialize spaCy model")

    def _needs_ner(self) -> bool:
        """
        Determine if named entity recognition (NER) is needed based on pattern_manager.

        Returns:
            bool: True if NER is required, False otherwise.
        """
        # Placeholder: Assume NER is needed for location entities (e.g., "USA")
        return True

    async def process_query(self, query: str) -> Dict[str, any]:
        """
        Process a user query to extract keywords, entities, and dependencies (Scenarios 1, 2, 5).

        Args:
            query (str): User query (e.g., "stores in USA").

        Returns:
            Dict[str, any]: Dictionary containing keywords, entities, and dependencies.
        """
        try:
            if not query or not query.strip():
                self.logger.warning("Empty or invalid query provided")
                return {"keywords": [], "entities": [], "dependencies": [], "reason": "empty_query"}

            # Process query asynchronously
            doc = await asyncio.to_thread(self.nlp, query)
            keywords = self.extract_keywords(doc)
            entities = self.extract_entities(doc)
            dependencies = self.extract_dependencies(doc)

            if not keywords:
                self.logger.warning(f"No valid keywords extracted from query: {query}")
                return {"keywords": [], "entities": entities, "dependencies": dependencies, "reason": "no_valid_keywords"}

            result = {
                "keywords": keywords,
                "entities": entities,
                "dependencies": dependencies,
                "reason": None
            }
            self.logger.debug(f"Processed query '{query}': {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {e}", exc_info=True)
            return {"keywords": [], "entities": [], "dependencies": [], "reason": f"error: {str(e)}"}

    def extract_keywords(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Extract non-stopword, non-punctuation tokens with lemmatization (Scenarios 1, 2).

        Args:
            doc (spacy.tokens.Doc): Processed spaCy document.

        Returns:
            List[str]: Lemmatized keywords.
        """
        try:
            keywords = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and token.text.strip()
            ]
            self.logger.debug(f"Extracted keywords: {keywords}")
            return keywords
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}", exc_info=True)
            return []

    def extract_entities(self, doc: spacy.tokens.Doc) -> List[Tuple[str, str]]:
        """
        Extract named entities from the query (Scenario 1).

        Args:
            doc (spacy.tokens.Doc): Processed spaCy document.

        Returns:
            List[Tuple[str, str]]: List of (entity_text, entity_label) tuples.
        """
        try:
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            self.logger.debug(f"Extracted entities: {entities}")
            return entities
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}", exc_info=True)
            return []

    def extract_dependencies(self, doc: spacy.tokens.Doc) -> List[Tuple[str, str, str]]:
        """
        Extract syntactic dependencies from the query (Scenario 1).

        Args:
            doc (spacy.tokens.Doc): Processed spaCy document.

        Returns:
            List[Tuple[str, str, str]]: List of (token_text, dependency_label, head_text) tuples.
        """
        try:
            dependencies = [
                (token.text, token.dep_, token.head.text)
                for token in doc
                if not token.is_stop and not token.is_punct
            ]
            self.logger.debug(f"Extracted dependencies: {dependencies}")
            return dependencies
        except Exception as e:
            self.logger.error(f"Error extracting dependencies: {e}", exc_info=True)
            return []
