"""
CLI for TIA-1.2, providing database interaction and query processing.
Supports feedback deduplication, synonym caching, schema validation, and S3 tables.
Preserves TIA-1.1 functionality, fixes table selection and performance issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based paths (e.g., BIKE-STORES/BikeStores/)
- Async cache and feedback operations for <5s query processing
- Logging to logs/DISPLAY_NAME_app.log
- Non-hardcoding with config_name from app-configs/db_configurations.json
- S3 table support (e.g., stores.csv)
- Training data integration (table/column synonyms, query patterns)
Addresses Scenarios 1 (table identification), 2 (synonym quality), 4 (training data), 5 (performance), 8 (feedback deduplication).
"""
import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from langdetect import detect_langs, LangDetectException
from datetime import datetime
import re
import numpy as np
import spacy
import pyodbc
import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from config.cache_synchronizer import CacheSynchronizer
from feedback.manager import FeedbackManager
from analysis.name_match_manager import NameMatchManager
from analysis.table_identifier import TableIdentifier
from config.patterns import PatternManager
from schema.manager import SchemaManager
from config.model_singleton import ModelSingleton
from config.logging_setup import setup_logging
from config.utils import load_db_config, list_configurations
from generate_training_data import generate_training_data

class DatabaseAnalyzerCLI:
    """CLI for TIA-1.2, handling database connections, queries, feedback, and training data."""
    def __init__(self, config_name: str = "bikestores"):
        """
        Initialize CLI with configuration name.

        Args:
            config_name (str): Configuration name (e.g., 'bikestores', 'bikestores-csv').
        """
        try:
            self.config = load_db_config(config_name)
            self.config_name = self.config.get("name", "bikestores")
            self.display_name = self.config.get("display_name", "BIKE-STORES")
            self.db_name = self.config.get("database", "BikeStores")
            self.schemas = self.config.get("schemas", [])
            self.tables = self.config.get("tables", [])

            setup_logging(self.display_name, self.db_name)
            self.logger = logging.getLogger("interface")

            # Initialize NLP
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self.logger.debug("Loaded spaCy model: en_core_web_sm")

            # Initialize core components
            self.model = ModelSingleton(self.config_name).model
            self.cache_synchronizer = CacheSynchronizer(self.display_name, self.db_name, self.config_name)
            self.schema_manager = None
            self.feedback_manager = None
            self.schema_dict = {}
            self.pattern_manager = None
            self.name_match_manager = None
            self.table_identifier = None
            self.connection = None

            self.training_data_dir = os.path.join(self.display_name, self.db_name, "training_data")
            self.table_synonyms_file = os.path.join(self.training_data_dir, "table_synonyms.json")
            self.column_synonyms_file = os.path.join(self.training_data_dir, "column_synonyms.json")
            self.config_path = os.path.join("app-configs", "db_configurations.json")
            self.query_history = []
            self.max_history = 50

            self.example_queries = [
                {"query": "Show me store details", "tables": ["sales.stores"]},
                {"query": "List products in stock", "tables": ["production.stocks", "production.products"]},
                {"query": "Get customer orders", "tables": ["sales.orders", "sales.customers"]},
                {"query": "Show staff in each store", "tables": ["sales.staffs", "sales.stores"]},
                {"query": "Find products by category", "tables": ["production.products", "production.categories"]}
            ]

            self.logger.info(f"Initialized DatabaseAnalyzerCLI for {self.display_name}/{self.db_name} (config: {self.config_name})")
        except Exception as e:
            self.logger.error(f"Failed to initialize CLI: {e}", exc_info=True)
            raise RuntimeError(f"CLI initialization failed: {e}")

    async def run(self):
        """Run the CLI with main menu (Scenario 5)."""
        self.logger.info(f"Started CLI for {self.display_name}/{self.db_name}")
        try:
            while True:
                print(f"\n=== {self.display_name} Schema Analyzer ===\n")
                print("Main Menu:")
                print("1. Connect to Database")
                print("2. Query Mode")
                print("3. Reload Configurations")
                print("4. Manage Feedback")
                print("5. Manage Ignored Queries")
                print("6. View Schema")
                print("7. View Query History")
                print("8. Generate Synthetic Feedback")
                print("9. Generate Training Data")
                print("10. Exit")
                choice = input("Select option: ").strip()
                self.logger.debug(f"User selected option: {choice}")

                try:
                    if choice == '1':
                        await self._connect_to_database()
                    elif choice == '2':
                        await self._query_mode()
                    elif choice == '3':
                        await self._reload_configurations()
                    elif choice == '4':
                        await self._manage_feedback()
                    elif choice == '5':
                        await self._manage_ignored_queries()
                    elif choice == '6':
                        await self._view_schema()
                    elif choice == '7':
                        await self._view_query_history()
                    elif choice == '8':
                        await self._generate_synthetic_feedback()
                    elif choice == '9':
                        await self._generate_training_data()
                    elif choice == '10':
                        print("Exiting...")
                        await self._cleanup()
                        break
                    else:
                        print("Invalid option. Please try again.")
                except Exception as e:
                    self.logger.error(f"Error processing option {choice}: {e}", exc_info=True)
                    print(f"Error: {e}")
        except KeyboardInterrupt:
            await self._cleanup()
            print("\nExiting...")
        except Exception as e:
            self.logger.error(f"CLI startup error: {e}", exc_info=True)
            print(f"Failed to start CLI: {e}")

    async def _connect_to_database(self, max_attempts: int = 3):
        """Connect to database with configuration and S3 support (Scenario 1)."""
        for attempt in range(1, max_attempts + 1):
            try:
                configs = list_configurations()
                self.logger.debug(f"Loaded {len(configs)} configurations")

                if not configs:
                    print("No database configurations available.")
                    self.logger.warning("No configurations found")
                    return

                print("\nAvailable Configurations:")
                for i, (name, display_name) in enumerate(configs, 1):
                    print(f"{i}. {display_name} ({name})")
                print(f"{len(configs) + 1}. Cancel")

                choice = input("Select configuration: ").strip()
                self.logger.debug(f"Configuration choice: {choice}")

                if not choice.isdigit() or int(choice) < 1 or int(choice) > len(configs) + 1:
                    print("Invalid choice.")
                    return

                if int(choice) == len(configs) + 1:
                    print("Connection cancelled.")
                    return

                config_name = configs[int(choice) - 1][0]
                self.config = load_db_config(config_name)
                self.config_name = self.config.get("name", config_name)
                self.display_name = self.config.get("display_name", "BIKE-STORES")
                self.db_name = self.config.get("database", "BikeStores")
                self.schemas = self.config.get("schemas", [])
                self.tables = self.config.get("tables", [])

                # Initialize managers
                self.schema_manager = SchemaManager(self.config_name, self.schemas, self.tables)
                if self.config["type"].lower() == "s3":
                    bucket_name = self.config.get("bucket_name", "")
                    if not bucket_name or not re.match(r"^[a-zA-Z0-9.\-_]{1,255}$", bucket_name):
                        raise ValueError(f"Invalid S3 bucket name: {bucket_name}")
                    try:
                        self.connection = boto3.client(
                            's3',
                            aws_access_key_id=self.config.get("access_key"),
                            aws_secret_access_key=self.config.get("secret_key"),
                            endpoint_url=self.config.get("s3_endpoint")
                        )
                        # Verify bucket access
                        self.connection.head_bucket(Bucket=bucket_name)
                    except ClientError as e:
                        raise ValueError(f"Failed to access S3 bucket {bucket_name}: {e}")
                else:
                    dsn = (
                        f"DRIVER={{{self.config.get('driver', 'ODBC Driver 17 for SQL Server')}}};"
                        f"SERVER={self.config.get('server', 'localhost')};"
                        f"DATABASE={self.db_name};"
                        f"UID={self.config.get('username', 'sa')};"
                        f"PWD={self.config.get('password')}"
                    )
                    self.connection = pyodbc.connect(dsn)

                self.schema_dict = await self.schema_manager.build_schema_dictionary(self.connection)
                if not self.schema_dict or 'tables' not in self.schema_dict:
                    print("Failed to build schema dictionary. Check configuration or schema availability.")
                    self.logger.error("Failed to build schema dictionary")
                    self.connection = None
                    self.schema_manager = None
                    self.schema_dict = {}
                    return

                self.feedback_manager = FeedbackManager(self.config_name, self.cache_synchronizer, is_s3_config=self.config["type"].lower() == "s3")
                self.name_match_manager = NameMatchManager(self.display_name, self.db_name, self.config_name)
                self.pattern_manager = PatternManager(self.schema_dict, self.display_name, self.db_name, self.schemas, self.tables)
                self.table_identifier = TableIdentifier(
                    self.schema_dict, self.feedback_manager, self.pattern_manager, self.cache_synchronizer, self.display_name, self.config_name
                )

                print(f"Connected to {self.display_name}/{self.db_name}")
                self.logger.info(f"Connected to configuration: {self.config_name}")
                break
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt} failed: {e}", exc_info=True)
                print(f"Connection failed: {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(0.1 * (2 ** (attempt - 1)))
                else:
                    self.connection = None
                    self.schema_manager = None
                    self.schema_dict = {}
                    print("Max connection attempts reached.")

    async def _query_mode(self):
        """Handle query mode with async processing and deduplication (Scenarios 1, 2, 5, 8)."""
        try:
            if not self.schema_dict or not self.feedback_manager:
                print("Please connect to a database first.")
                self.logger.warning("Schema or feedback manager not initialized")
                return

            example_queries = await self.feedback_manager.get_top_queries(limit=5) or self.example_queries
            print("\nExample Queries:")
            for i, q in enumerate(example_queries, 1):
                print(f"{i}. {q['query']} -> {', '.join(q['tables'])}")

            while True:
                query = input("\nEnter query (or 'back'): ").strip()
                self.logger.debug(f"Received query: {query}")
                if query.lower() == 'back':
                    break
                if not query:
                    print("Query cannot be empty.")
                    continue

                is_valid, reason = await self._validate_query(query)
                if not is_valid:
                    print(f"Query rejected: {reason}")
                    print("\nExample Queries:")
                    for i, q in enumerate(example_queries, 1):
                        print(f"{i}. {q['query']} -> {', '.join(q['tables'])}")
                    continue

                processed_query = await self._expand_query_with_synonyms(query)
                column_scores = await self.name_match_manager.process_query(processed_query, self.schema_dict)
                tables, confidence = await self.table_identifier.identify_tables(processed_query, column_scores)

                similar_feedback = await self.cache_synchronizer.find_similar_feedback(query)
                feedback_tables = []
                if similar_feedback:
                    print("\nSimilar queries found in feedback:")
                    for fb_query, fb_tables, sim in similar_feedback[:3]:
                        print(f"- {fb_query} -> {', '.join(fb_tables)} (similarity: {sim:.2f})")
                        feedback_tables.extend(fb_tables)
                    feedback_tables = list(set(feedback_tables))

                if not tables and feedback_tables:
                    tables = feedback_tables
                    confidence = max(confidence, 0.9)

                if not tables:
                    print("No tables identified for the query.")
                    if similar_feedback:
                        print("Using feedback tables:", ', '.join(feedback_tables))
                    embedding = await asyncio.to_thread(self.model.encode, query, show_progress_bar=False) if self.model else None
                    await self.cache_synchronizer.insert_ignored_query(query, embedding, "no_tables_identified")
                    self.query_history.append({
                        'query': query,
                        'tables': [],
                        'confidence': 0.0,
                        'timestamp': datetime.now()
                    })
                    continue

                print(f"\nIdentified Tables: {', '.join(tables)}")
                print(f"Confidence: {'High' if confidence > 0.5 else 'Low'}")

                valid_tables = []
                for schema in self.schema_dict.get('tables', {}):
                    for table in self.schema_dict['tables'][schema]:
                        valid_tables.append(f"{schema}.{table}" if schema != 's3' else table)
                valid_tables = sorted(valid_tables)
                print("\nAvailable Tables:")
                for i, table in enumerate(valid_tables, 1):
                    print(f"{i}. {table}")

                feedback_ok = input("\nAre these tables correct? (y/n): ").strip().lower()
                embedding = await asyncio.to_thread(self.model.encode, query, show_progress_bar=False) if self.model else None
                is_duplicate, feedback_id, existing_tables = await self.table_identifier.check_duplicate_feedback(query, tables)

                if feedback_ok == 'y':
                    try:
                        if is_duplicate:
                            if set(tables) != set(existing_tables):
                                await self.feedback_manager.update_feedback(feedback_id, query, tables, confidence)
                                self.logger.debug(f"Updated feedback {feedback_id} for {query} -> {tables}")
                            else:
                                self.logger.debug(f"Duplicate feedback for {query}, no update needed")
                        else:
                            await self.feedback_manager.store_feedback(query, tables, self.schema_dict, confidence)
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                            await self.cache_synchronizer.insert_feedback(timestamp, query, tables, embedding)
                            self.logger.debug(f"Stored feedback for {query} -> {tables}")
                        await self.table_identifier.update_weights_from_feedback(query, tables)
                        await self.table_identifier.save_name_matches()
                        self.query_history.append({
                            'query': query,
                            'tables': tables,
                            'confidence': confidence,
                            'timestamp': datetime.now()
                        })
                    except Exception as e:
                        self.logger.error(f"Error storing feedback: {e}", exc_info=True)
                        print("Failed to store feedback.")
                else:
                    print("\nEnter correct tables (e.g., sales.stores,stores or 1,2) or 'skip':")
                    while True:
                        correct_tables_input = input().strip()
                        if correct_tables_input.lower() == 'skip':
                            await self.cache_synchronizer.insert_ignored_query(query, embedding, "user_rejected")
                            self.query_history.append({
                                'query': query,
                                'tables': tables,
                                'confidence': confidence,
                                'timestamp': datetime.now(),
                                'rejected': True
                            })
                            break

                        correct_tables_list = []
                        input_parts = [t.strip() for t in correct_tables_input.split(',') if t.strip()]
                        for part in input_parts:
                            if re.match(r'^\d+$', part) and 1 <= int(part) <= len(valid_tables):
                                correct_tables_list.append(valid_tables[int(part) - 1])
                            elif re.match(r'^(\w+\.\w+|\w+)$', part):
                                correct_tables_list.append(part)
                            else:
                                print(f"Invalid input: {part}")
                                correct_tables_list = []
                                break

                        valid_tables_selected, invalid_tables = self.table_identifier.validate_tables(correct_tables_list)
                        if invalid_tables:
                            print(f"\nInvalid tables: {', '.join(invalid_tables)}")
                            self.logger.debug(f"Invalid tables provided: {invalid_tables}")
                            print("\nAvailable Tables:")
                            for i, table in enumerate(valid_tables, 1):
                                print(f"{i}. {table}")
                            print("\nPlease enter valid tables or 'skip':")
                            continue

                        if valid_tables_selected:
                            try:
                                if is_duplicate:
                                    await self.feedback_manager.update_feedback(feedback_id, query, valid_tables_selected, confidence)
                                    self.logger.debug(f"Updated feedback {feedback_id} for {query} -> {valid_tables_selected}")
                                else:
                                    await self.feedback_manager.store_feedback(query, valid_tables_selected, self.schema_dict, confidence)
                                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                                    await self.cache_synchronizer.insert_feedback(timestamp, query, valid_tables_selected, embedding)
                                    self.logger.debug(f"Stored feedback for {query} -> {valid_tables_selected}")
                                await self.table_identifier.update_weights_from_feedback(query, valid_tables_selected)
                                await self.table_identifier.save_name_matches()
                                query_tokens = self.name_match_manager.extract_tokens(query)
                                for table in valid_tables_selected:
                                    schema = 's3' if '.' not in table else table.split('.')[0]
                                    table_name = table if schema == 's3' else table.split('.')[1]
                                    columns = self.schema_dict['columns'].get(schema, {}).get(table_name, {})
                                    for token in query_tokens:
                                        for col in columns:
                                            col_lower = col.lower()
                                            similarity = self.name_match_manager.compute_similarity(token, col_lower)
                                            if similarity > 0.75:
                                                self.name_match_manager.dynamic_matches.setdefault(col_lower, []).append(token)
                                                await self.name_match_manager._save_dynamic_matches()
                                                self.logger.debug(f"Cached synonym '{token}' for '{col_lower}' in '{table}' (sim={similarity:.2f})")
                                self.query_history.append({
                                    'query': query,
                                    'tables': valid_tables_selected,
                                    'confidence': confidence,
                                    'timestamp': datetime.now()
                                })
                                break
                            except Exception as e:
                                self.logger.error(f"Error processing correct tables: {e}", exc_info=True)
                                print(f"Error processing correct tables: {e}")
                                await self.cache_synchronizer.insert_ignored_query(query, embedding, "user_rejected_error")
                                break
                        else:
                            await self.cache_synchronizer.insert_ignored_query(query, embedding, "user_rejected_no_valid_tables")
                            self.query_history.append({
                                'query': query,
                                'tables': tables,
                                'confidence': confidence,
                                'timestamp': datetime.now(),
                                'rejected': True
                            })
                            break

                if len(self.query_history) > self.max_history:
                    self.query_history = self.query_history[-self.max_history:]
        except Exception as e:
            self.logger.error(f"Error in query mode: {e}", exc_info=True)
            print("An error occurred in query mode.")

    async def _validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate query for length, language, format, and relevance (Scenarios 1, 2)."""
        try:
            if not query or len(query.strip()) < 3:
                self.logger.debug("Query too short or empty")
                return False, "Query too short or empty"

            query_normalized = re.sub(r'\s+', ' ', query.strip().lower())
            query_lower = query_normalized
            query_clean = re.sub(r'\b[jJ]an\b-(\d{1,2})-(\d{4})\b', r'January \1, \2', query_lower)
            query_clean = re.sub(r'\b(\d+)\s*(items?|products?|orders?)\b', r'\1 \2', query_clean)
            doc = await asyncio.to_thread(self.nlp, query_clean)
            cleaned_query = ' '.join(token.text for token in doc if token.ent_type_ not in ('PERSON', 'ORG', 'GPE'))
            if not cleaned_query.strip():
                cleaned_query = query_lower

            if self.feedback_manager:
                feedback = await self.feedback_manager.get_all_feedback()
                for entry in feedback:
                    if query_normalized == entry['query'].lower():
                        self.logger.debug("Query matched feedback")
                        return True, ""

            try:
                lang_result = detect_langs(cleaned_query)
                lang, confidence = lang_result[0].lang, lang_result[0].prob
                self.logger.debug(f"Language: lang={lang}, confidence={confidence:.2f}")
                if lang != 'en' and confidence > 0.9:
                    tokens = self.name_match_manager.extract_tokens(cleaned_query)
                    schema_match = False
                    for token in tokens:
                        for schema in self.schema_dict.get('tables', {}):
                            for table in self.schema_dict['tables'][schema]:
                                if token.lower() in table.lower():
                                    schema_match = True
                                    self.logger.debug(f"Non-English schema match: table '{token}'")
                                    break
                                for column in self.schema_dict['columns'].get(schema, {}).get(table, {}):
                                    if token.lower() in column.lower():
                                        schema_match = True
                                        self.logger.debug(f"Non-English schema match: column '{token}'")
                                        break
                                for col, synonyms in {**self.name_match_manager.default_matches, **self.name_match_manager.dynamic_matches}.items():
                                    if token.lower() in synonyms:
                                        schema_match = True
                                        self.logger.debug(f"Non-English synonym match: '{token}' for '{col}'")
                                        break
                            if schema_match:
                                break
                    if schema_match:
                        example_queries = await self.feedback_manager.get_top_queries(limit=5) or self.example_queries
                        print("\nNon-English query detected. Example English queries:")
                        for i, q in enumerate(example_queries, 1):
                            print(f"{i}. {q['query']} -> {', '.join(q['tables'])}")
                        print(f"\nNon-English query (language: {lang}, confidence: {confidence:.2f})")
                        override = input("Force English processing? (y/n): ").strip().lower()
                        if override == 'y':
                            self.logger.debug("User forced English")
                            return True, ""
                        await self.cache_synchronizer.insert_ignored_query(query, None, f"non_english_lang_{lang}")
                        return False, f"Non-English query (detected: {lang})"
                    else:
                        await self.cache_synchronizer.insert_ignored_query(query, None, f"non_english_irrelevant_{lang}")
                        return False, f"Non-English query unrelated to database (detected: {lang})"
            except LangDetectException:
                self.logger.warning("Language detection failed")
                return True, ""

            ignored_queries = self.cache_synchronizer.read_ignored_queries()
            for iq, info in ignored_queries.items():
                if query_normalized == iq.lower():
                    self.logger.debug("Ignored query matched")
                    return False, f"Ignored query (reason: {info['reason']})"

            intent_patterns = [
                r'^(how\s+many|what|which|list|show|get|find|count|sum|select)\b',
                r'\b(names|categories|products|orders|sales|customers|stores)\b'
            ]
            has_intent = any(re.search(pattern, query_lower) for pattern in intent_patterns)
            if not has_intent:
                self.logger.debug("No actionable intent")
                return False, "Query lacks actionable intent"

            if not self.schema_dict or 'tables' not in self.schema_dict:
                self.logger.warning("Schema dictionary empty")
                return False, "Schema not initialized"

            schema_match = False
            tokens = self.name_match_manager.extract_tokens(cleaned_query)
            for token in tokens:
                for schema in self.schema_dict.get('tables', {}):
                    for table in self.schema_dict['tables'][schema]:
                        if token.lower() in table.lower():
                            schema_match = True
                            self.logger.debug(f"Schema match: table '{token}'")
                            break
                        for column in self.schema_dict['columns'].get(schema, {}).get(table, {}):
                            if token.lower() in column.lower():
                                schema_match = True
                                self.logger.debug(f"Schema match: column '{token}'")
                                break
                    if schema_match:
                        break
                for col, synonyms in {**self.name_match_manager.default_matches, **self.name_match_manager.dynamic_matches}.items():
                    if token.lower() in synonyms:
                        schema_match = True
                        self.logger.debug(f"Synonym match: '{token}' for '{col}'")
                        break
                if not schema_match:
                    token_embedding = await asyncio.to_thread(self.model.encode, token, show_progress_bar=False)
                    for schema in self.schema_dict.get('tables', {}):
                        for table in self.schema_dict['tables'][schema]:
                            for column in self.schema_dict['columns'].get(schema, {}).get(table, {}):
                                col_embedding = await asyncio.to_thread(self.model.encode, column.lower(), show_progress_bar=False)
                                similarity = np.dot(token_embedding, col_embedding) / (
                                    np.linalg.norm(token_embedding) * np.linalg.norm(col_embedding)
                                )
                                if similarity > 0.75:
                                    schema_match = True
                                    self.logger.debug(f"Embedding match: '{token}' similar to '{column}' (sim={similarity:.2f})")
                                    break
                        if schema_match:
                            break
                if schema_match:
                    break

            if schema_match or has_intent:
                self.logger.debug(f"Query validated: schema_match={schema_match}, has_intent={has_intent}")
                return True, ""

            self.logger.debug("No relevant tables, columns, or synonyms")
            await self.cache_synchronizer.insert_ignored_query(query, None, "irrelevant")
            return False, "No relevant tables or columns found"
        except Exception as e:
            self.logger.error(f"Error validating query: {e}", exc_info=True)
            return False, f"Validation error: {e}"

    async def _expand_query_with_synonyms(self, query: str) -> str:
        """Expand query with synonyms from training data and NameMatchManager (Scenario 2)."""
        try:
            synonyms_data = {}
            for file in [self.table_synonyms_file, self.column_synonyms_file]:
                if os.path.exists(file):
                    with open(file, 'r') as f:
                        synonyms_data.update(json.load(f))
                    self.logger.debug(f"Loaded synonyms from {file}")

            synonyms = {}
            for key, syn_list in synonyms_data.items():
                if isinstance(syn_list, list):
                    synonyms[key.lower()] = syn_list
                elif isinstance(syn_list, dict):
                    for col, col_syns in syn_list.items():
                        synonyms[col.lower()] = col_syns

            query_words = query.lower().split()
            expanded_words = []
            for word in query_words:
                for key, syn_list in synonyms.items():
                    if word == key:
                        expanded_words.extend(syn_list)
                        self.logger.debug(f"Expanded '{word}' to {syn_list} from training data")
                        break
                else:
                    for col, syn_list in {**self.name_match_manager.default_matches, **self.name_match_manager.dynamic_matches}.items():
                        if word in syn_list:
                            expanded_words.append(col)
                            self.logger.debug(f"Expanded '{word}' to '{col}' via NameMatchManager")
                            break
                    else:
                        expanded_words.append(word)

            expanded_query = ' '.join(expanded_words)
            self.logger.debug(f"Expanded query: {query} -> {expanded_query}")
            return expanded_query
        except Exception as e:
            self.logger.error(f"Error expanding query: {e}", exc_info=True)
            return query

    async def _reload_configurations(self, max_attempts: int = 3):
        """Reload configurations and reinitialize managers asynchronously (Scenario 5)."""
        for attempt in range(1, max_attempts + 1):
            try:
                configs = list_configurations()
                self.logger.debug(f"Reloaded {len(configs)} configurations")

                if not configs:
                    print("No configurations reloaded.")
                    self.logger.warning("No configurations found")
                    return

                print("\nAvailable Configurations:")
                for i, (name, display_name) in enumerate(configs, 1):
                    print(f"{i}. {display_name} ({name})")
                print(f"{len(configs) + 1}. Cancel")

                choice = input("Select configuration to reload: ").strip()
                self.logger.debug(f"Reload configuration choice: {choice}")

                if not choice.isdigit() or int(choice) < 1 or int(choice) > len(configs) + 1:
                    print("Invalid choice.")
                    return

                if int(choice) == len(configs) + 1:
                    print("Reload cancelled.")
                    return

                config_name = configs[int(choice) - 1][0]
                self.config = load_db_config(config_name)
                self.config_name = self.config.get("name", config_name)
                self.display_name = self.config.get("display_name", "BIKE-STORES")
                self.db_name = self.config.get("database", "BikeStores")
                self.schemas = self.config.get("schemas", [])
                self.tables = self.config.get("tables", [])

                self.schema_manager = SchemaManager(self.config_name, self.schemas, self.tables)
                if self.config["type"].lower() == "s3":
                    bucket_name = self.config.get("bucket_name", "")
                    if not bucket_name or not re.match(r"^[a-zA-Z0-9.\-_]{1,255}$", bucket_name):
                        raise ValueError(f"Invalid S3 bucket name: {bucket_name}")
                    try:
                        self.connection = boto3.client(
                            's3',
                            aws_access_key_id=self.config.get("access_key"),
                            aws_secret_access_key=self.config.get("secret_key"),
                            endpoint_url=self.config.get("s3_endpoint")
                        )
                        self.connection.head_bucket(Bucket=bucket_name)
                    except ClientError as e:
                        raise ValueError(f"Failed to access S3 bucket {bucket_name}: {e}")
                else:
                    dsn = (
                        f"DRIVER={{{self.config.get('driver', 'ODBC Driver 17 for SQL Server')}}};"
                        f"SERVER={self.config.get('server', 'localhost')};"
                        f"DATABASE={self.db_name};"
                        f"UID={self.config.get('username', 'sa')};"
                        f"PWD={self.config.get('password')}"
                    )
                    self.connection = pyodbc.connect(dsn)

                self.schema_dict = await self.schema_manager.build_schema_dictionary(self.connection)
                if not self.schema_dict or 'tables' not in self.schema_dict:
                    print("Failed to reload schema dictionary. Check configuration or schema availability.")
                    self.logger.error("Failed to reload schema dictionary")
                    self.connection = None
                    self.schema_manager = None
                    self.schema_dict = {}
                    return

                self.feedback_manager = FeedbackManager(self.config_name, self.cache_synchronizer, is_s3_config=self.config["type"].lower() == "s3")
                self.name_match_manager = NameMatchManager(self.display_name, self.db_name, self.config_name)
                self.pattern_manager = PatternManager(self.schema_dict, self.display_name, self.db_name, self.schemas, self.tables)
                self.table_identifier = TableIdentifier(
                    self.schema_dict, self.feedback_manager, self.pattern_manager, self.cache_synchronizer, self.display_name, self.config_name
                )

                print(f"Configurations reloaded for {self.display_name}/{self.db_name}")
                self.logger.info(f"Reloaded configurations for {self.config_name}")
                break
            except Exception as e:
                self.logger.error(f"Reload attempt {attempt} failed: {e}", exc_info=True)
                print(f"Error reloading configurations: {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(0.1 * (2 ** (attempt - 1)))
                else:
                    self.connection = None
                    self.schema_manager = None
                    self.schema_dict = {}
                    print("Max reload attempts reached.")

    async def _manage_feedback(self):
        """Manage feedback entries with async similarity search (Scenario 8)."""
        try:
            if not self.feedback_manager:
                print("Feedback manager not initialized. Please connect to a database.")
                self.logger.warning("Feedback manager not initialized")
                return

            while True:
                print("\nManage Feedback:")
                print("1. List Feedback")
                print("2. Add Feedback")
                print("3. Remove Feedback")
                print("4. Clear All Feedback")
                print("5. Find Similar Feedback")
                print("6. Back")
                choice = input("Select option: ").strip()
                self.logger.debug(f"Feedback option: {choice}")

                if choice == '1':
                    feedback = await self.feedback_manager.get_all_feedback()
                    if not feedback:
                        print("No feedback found.")
                    else:
                        print("\nFeedback Entries:")
                        for entry in feedback:
                            print(f"- Query: {entry['query']}, Tables: {', '.join(entry['tables'])}, Count: {entry['count']}, Confidence: {entry['confidence']:.2f}")
                elif choice == '2':
                    query = input("Enter query: ").strip()
                    print("\nAvailable Tables:")
                    valid_tables = []
                    for schema in self.schema_dict.get('tables', {}):
                        for table in self.schema_dict['tables'][schema]:
                            valid_tables.append(f"{schema}.{table}" if schema != 's3' else table)
                    for i, table in enumerate(sorted(valid_tables), 1):
                        print(f"{i}. {table}")
                    tables = input("Enter tables (comma-separated): ").strip().split(',')
                    tables = [t.strip() for t in tables if t.strip()]
                    if query and tables:
                        valid_tables, invalid_tables = self.table_identifier.validate_tables(tables)
                        if invalid_tables:
                            print(f"Invalid tables: {', '.join(invalid_tables)}")
                            continue
                        await self.feedback_manager.store_feedback(query, valid_tables, self.schema_dict)
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        embedding = await asyncio.to_thread(self.model.encode, query, show_progress_bar=False) if self.model else None
                        await self.cache_synchronizer.insert_feedback(timestamp, query, valid_tables, embedding)
                        await self.table_identifier.update_weights_from_feedback(query, valid_tables)
                        await self.table_identifier.save_name_matches()
                        print(f"Feedback added: {query} -> {valid_tables}")
                        self.logger.debug(f"Added feedback: {query}")
                    else:
                        print("Query and tables cannot be empty.")
                elif choice == '3':
                    query = input("Enter query to remove: ").strip()
                    feedback = await self.feedback_manager.get_all_feedback()
                    if any(entry['query'].lower() == query.lower() for entry in feedback):
                        await self.feedback_manager.delete_feedback(query)
                        await self.cache_synchronizer.delete_feedback(query)
                        print(f"Feedback removed: {query}")
                        self.logger.debug(f"Removed feedback: {query}")
                    else:
                        print("Feedback not found.")
                elif choice == '4':
                    confirm = input("Clear all feedback? (y/n): ").strip().lower()
                    if confirm == 'y':
                        await self.feedback_manager.clear_feedback()
                        await self.cache_synchronizer.clear_cache(table='feedback')
                        print("All feedback cleared.")
                        self.logger.debug("Cleared all feedback")
                    else:
                        print("Clear cancelled.")
                elif choice == '5':
                    query = input("Enter query to find similar feedback: ").strip()
                    similar = await self.cache_synchronizer.find_similar_feedback(query)
                    if not similar:
                        print("No similar feedback found.")
                    else:
                        print("\nSimilar Feedback:")
                        for fb_query, tables, sim in similar:
                            print(f"- {fb_query} -> {', '.join(tables)} (similarity: {sim:.2f})")
                elif choice == '6':
                    break
                else:
                    print("Invalid option.")
        except Exception as e:
            self.logger.error(f"Error managing feedback: {e}", exc_info=True)
            print("An error occurred while managing feedback.")

    async def _manage_ignored_queries(self):
        """Manage ignored queries with async validation (Scenario 8)."""
        try:
            while True:
                print("\nManage Ignored Queries:")
                print("1. List Ignored Queries")
                print("2. Add Ignored Query")
                print("3. Remove Ignored Query")
                print("4. Clear All Ignored Queries")
                print("5. Back")
                choice = input("Select option: ").strip()
                self.logger.debug(f"Ignored queries option: {choice}")

                if choice == '1':
                    ignored_queries = self.cache_synchronizer.read_ignored_queries()
                    if not ignored_queries:
                        print("No ignored queries found.")
                    else:
                        print("\nIgnored Queries:")
                        for query, info in ignored_queries.items():
                            print(f"- {query} (Reason: {info['reason']})")
                elif choice == '2':
                    query = input("Enter query to ignore: ").strip()
                    if not query or len(query.strip()) < 3:
                        print("Query must be at least 3 characters long.")
                        continue
                    ignored_queries = self.cache_synchronizer.read_ignored_queries()
                    if query.lower() in [q.lower() for q in ignored_queries]:
                        print(f"Query '{query}' is already ignored.")
                        continue
                    reason = input("Enter reason: ").strip()
                    if not reason:
                        print("Reason cannot be empty.")
                        continue
                    embedding = await asyncio.to_thread(self.model.encode, query, show_progress_bar=False) if self.model else None
                    await self.cache_synchronizer.insert_ignored_query(query, embedding, reason)
                    print(f"Ignored query added: {query}")
                    self.logger.debug(f"Added ignored query: {query}")
                elif choice == '3':
                    query = input("Enter query to remove: ").strip()
                    ignored_queries = self.cache_synchronizer.read_ignored_queries()
                    if query.lower() in [q.lower() for q in ignored_queries]:
                        await self.cache_synchronizer.delete_ignored_query(query)
                        print(f"Ignored query removed: {query}")
                        self.logger.debug(f"Removed ignored query: {query}")
                    else:
                        print("Query not found in ignored list.")
                elif choice == '4':
                    confirm = input("Clear all ignored queries? (y/n): ").strip().lower()
                    if confirm == 'y':
                        await self.cache_synchronizer.clear_ignored_queries()
                        print("All ignored queries cleared.")
                        self.logger.debug("Cleared all ignored queries")
                    else:
                        print("Clear cancelled.")
                elif choice == '5':
                    break
                else:
                    print("Invalid option.")
        except Exception as e:
            self.logger.error(f"Error managing ignored queries: {e}", exc_info=True)
            print("An error occurred while managing ignored queries.")

    async def _view_schema(self):
        """Display schema details with S3 support (Scenario 1)."""
        try:
            if not self.schema_dict or 'tables' not in self.schema_dict or not any(self.schema_dict['tables'].values()):
                print("Schema not initialized. Please connect to a database.")
                self.logger.warning("Schema not initialized")
                return

            print("\nDatabase Schema:")
            for schema in self.schema_dict['tables']:
                if not self.schema_dict['tables'][schema]:
                    continue
                print(f"\nSchema: {schema}")
                for table in self.schema_dict['tables'][schema]:
                    table_display = f"{schema}.{table}" if schema != 's3' else table
                    print(f"  Table: {table_display}")
                    columns = self.schema_dict['columns'].get(schema, {}).get(table, {})
                    if columns:
                        print("    Columns:")
                        for col, col_info in columns.items():
                            print(f"      - {col} ({col_info['type']}, Nullable: {col_info['nullable']})")
                    primary_keys = self.schema_dict['primary_keys'].get(schema, {}).get(table, [])
                    if primary_keys:
                        print("    Primary Keys:")
                        for pk in primary_keys:
                            print(f"      - {pk}")
                    foreign_keys = self.schema_dict['foreign_keys'].get(schema, {}).get(table, [])
                    if foreign_keys:
                        print("    Foreign Keys:")
                        for fk in foreign_keys:
                            try:
                                ref_table = fk.get('referenced_table', 'unknown')
                                ref_column = fk.get('referenced_column', 'unknown')
                                print(f"      - {fk['column']} -> {ref_table}.{ref_column}")
                            except KeyError as e:
                                self.logger.error(f"Invalid foreign key structure for {schema}.{table}: {fk}")
                                print(f"      - {fk['column']} -> [invalid: missing {e}]")
            self.logger.debug("Displayed schema details")
        except Exception as e:
            self.logger.error(f"Error displaying schema: {e}", exc_info=True)
            print(f"An error occurred while displaying schema: {e}")

    async def _view_query_history(self):
        """Display query history with detailed status (Scenario 5)."""
        try:
            if not self.query_history:
                print("No query history available.")
                self.logger.debug("No query history")
                return

            print("\nQuery History:")
            for entry in self.query_history:
                status = "Rejected" if entry.get('rejected') else "Accepted"
                print(f"- Query: {entry['query']}")
                print(f"  Tables: {', '.join(entry['tables']) if entry['tables'] else 'None'}")
                print(f"  Status: {status}")
                print(f"  Confidence: {entry['confidence']:.2f}")
                print(f"  Timestamp: {entry['timestamp']}")
            self.logger.debug("Displayed query history")
        except Exception as e:
            self.logger.error(f"Error displaying query history: {e}", exc_info=True)
            print("An error occurred while displaying query history.")

    async def _generate_synthetic_feedback(self):
        """Generate synthetic feedback for training asynchronously (Scenario 8)."""
        try:
            if not self.schema_dict or not self.feedback_manager:
                print("Schema or feedback manager not initialized. Please connect to a database.")
                self.logger.warning("Schema or feedback manager not initialized")
                return

            print("\nGenerating synthetic feedback...")
            count = 0
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    table_full = f"{schema}.{table}" if schema != 's3' else table
                    table_name = table.lower()
                    queries = [
                        f"List all {table_name}",
                        f"Show {table_name} details",
                        f"Count {table_name}"
                    ]
                    for col in self.schema_dict['columns'][schema][table]:
                        col_lower = col.lower()
                        queries.extend([
                            f"Find {table_name} by {col_lower}",
                            f"List {table_name} with {col_lower}",
                            f"Sum {col_lower} in {table_name}"
                        ])
                    for query in queries:
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        embedding = await asyncio.to_thread(self.model.encode, query, show_progress_bar=False) if self.model else None
                        await self.feedback_manager.store_feedback(query, [table_full], self.schema_dict)
                        await self.cache_synchronizer.insert_feedback(timestamp, query, [table_full], embedding)
                        count += 1
            print(f"Generated {count} synthetic feedback entries.")
            self.logger.debug(f"Generated {count} synthetic feedback entries")
        except Exception as e:
            self.logger.error(f"Error generating synthetic feedback: {e}", exc_info=True)
            print("An error occurred while generating synthetic feedback.")

    async def _generate_training_data(self):
        """Generate training data using generate_training_data.py (Scenario 4)."""
        try:
            if not self.schema_manager:
                print("Schema manager not initialized. Please connect to a database.")
                self.logger.warning("Schema manager not initialized")
                return

            print(f"\nGenerating training data for {self.config_name}...")
            success = await generate_training_data(self.config_name, self.schemas, self.tables)
            if success:
                print(f"Training data generated successfully in {self.training_data_dir}")
                self.logger.info(f"Training data generated for {self.config_name}")
            else:
                print("Failed to generate training data. Check logs for details.")
                self.logger.error(f"Failed to generate training data for {self.config_name}")
        except Exception as e:
            self.logger.error(f"Error generating training data: {e}", exc_info=True)
            print("An error occurred while generating training data.")

    async def _cleanup(self):
        """Clean up resources before exiting asynchronously (Scenario 5)."""
        try:
            if self.connection:
                if isinstance(self.connection, pyodbc.Connection):
                    self.connection.close()
                    self.logger.debug("Closed SQL Server connection")
                elif isinstance(self.connection, BaseClient):
                    self.connection.close()
                    self.logger.debug("Closed S3 client connection")
                self.connection = None
            if hasattr(self, 'cache_synchronizer') and self.cache_synchronizer:
                await self.cache_synchronizer.close()
                self.logger.debug("Closed cache synchronizer")
                self.cache_synchronizer = None
            self.query_history = []
            self.schema_dict = {}
            self.schema_manager = None
            self.feedback_manager = None
            self.pattern_manager = None
            self.name_match_manager = None
            self.table_identifier = None
            self.logger.debug("Cleared query history and managers")
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
            print(f"Cleanup error: {e}")

if __name__ == "__main__":
    cli = DatabaseAnalyzerCLI(config_name="bikestores")
    asyncio.run(cli.run())