"""
Manages SQLite cache for feedback, weights, name matches, ignored queries, and query results in TIA-1.2.
Supports SQL Server and S3 CSV configurations with thread-safe and async operations.
Preserves all TIA-1.1 functionality, fixes deduplication and performance issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based paths (e.g., BIKE-STORES/BikeStores/cache.db)
- Async SQLite operations for <5s query processing
- Logging to codebase/logs/DISPLAY_NAME_app.log
- S3 table support in feedback
- Non-hardcoding with config_name
Addresses Scenarios 1 (table identification), 2 (synonym quality), 3 (feedback usage),
5 (performance), 6 (ignored queries), and 8 (deduplication).
"""
import os
import sqlite3
import logging
import asyncio
import json
import threading
import time
import re
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sentence_transformers import util
from datetime import datetime
from config.model_singleton import ModelSingleton
from config.logging_setup import setup_logging

class CacheSynchronizer:
    """Manages SQLite cache for feedback, weights, name matches, ignored queries, and query results."""
    def __init__(self, display_name: str, db_name: str, config_name: str = "bikestores"):
        """
        Initialize CacheSynchronizer with SQLite database for a specific database.

        Args:
            display_name (str): Display name (e.g., BIKE-STORES).
            db_name (str): Database name (e.g., BikeStores).
            config_name (str): Configuration name (e.g., bikestores).
        """
        setup_logging(display_name)
        self.display_name = display_name
        self.db_name = db_name
        self.config_name = config_name
        self.db_path = os.path.join(display_name, db_name, "cache.db")
        self.model = ModelSingleton(config_name).model
        self.lock = threading.Lock()
        self.deduplication_lock = threading.Lock()  # Separate lock for deduplication
        self._conn = None  # Persistent connection
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.logger = logging.getLogger("cache_synchronizer")
        try:
            self.logger.debug(f"Starting SQLite database initialization at {self.db_path}")
            self._validate_db_file()
            self._initialize_database()
            self.logger.debug(f"Completed SQLite database initialization at {self.db_path}")
            self.logger.info(f"Initialized CacheSynchronizer for {display_name}/{db_name} (config: {config_name})")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite database: {e}", exc_info=True)
            self._fallback_to_memory_db()
            self.logger.warning(f"Using in-memory SQLite database as fallback for {display_name}/{db_name}")

    def _validate_db_file(self):
        """
        Validate that the database file is accessible and not corrupted.
        """
        try:
            if os.path.exists(self.db_path):
                if not os.access(self.db_path, os.R_OK | os.W_OK):
                    self.logger.error(f"Database file {self.db_path} is not readable/writable")
                    raise PermissionError(f"Insufficient permissions for {self.db_path}")
                conn = sqlite3.connect(self.db_path, timeout=5)
                conn.execute("PRAGMA integrity_check")
                conn.close()
                self.logger.debug(f"Validated database file {self.db_path}")
            else:
                self.logger.debug(f"Database file {self.db_path} does not exist, will be created")
        except Exception as e:
            self.logger.error(f"Database file validation failed: {e}", exc_info=True)
            if os.path.exists(self.db_path):
                backup_path = f"{self.db_path}.bak"
                os.rename(self.db_path, backup_path)
                self.logger.info(f"Backed up corrupted database to {backup_path}")
            raise

    def _fallback_to_memory_db(self):
        """
        Switch to an in-memory SQLite database if file-based database fails.
        """
        try:
            self.db_path = ":memory:"
            self._conn = sqlite3.connect(self.db_path, timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=15000")
            self._initialize_database()
            self.logger.info(f"Initialized in-memory SQLite database for {self.display_name}/{self.db_name}")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to initialize in-memory database: {e}", exc_info=True)
            raise RuntimeError("Cannot initialize SQLite database")

    def _get_connection(self):
        """
        Get or create a persistent SQLite connection with optimized settings.

        Returns:
            SQLite connection object.
        """
        with self.lock:
            if self._conn is None or getattr(self._conn, 'closed', True):
                try:
                    start_time = time.time()
                    self._conn = sqlite3.connect(self.db_path, timeout=10)
                    if time.time() - start_time > 10:
                        self.logger.warning("SQLite connection took longer than expected")
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA synchronous=NORMAL")
                    self._conn.execute("PRAGMA busy_timeout=15000")
                    self.logger.debug(f"Created and configured SQLite connection to {self.db_path}")
                except sqlite3.Error as e:
                    self.logger.error(f"Error creating SQLite connection: {e}", exc_info=True)
                    raise
            return self._conn

    def _initialize_database(self):
        """
        Create necessary tables sequentially with immediate commits and PRIMARY KEY.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Weights table
            self.logger.debug("Creating weights table")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weights (
                    table_name TEXT,
                    column TEXT,
                    weight REAL,
                    PRIMARY KEY (table_name, column)
                )
            """)
            conn.commit()

            # Name matches table
            self.logger.debug("Creating name_matches table")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS name_matches (
                    column_name TEXT,
                    synonym TEXT,
                    PRIMARY KEY (column_name, synonym)
                )
            """)
            conn.commit()

            # Feedback table
            self.logger.debug("Creating feedback table")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    timestamp TEXT,
                    query TEXT,
                    tables TEXT,
                    embedding BLOB,
                    count INTEGER,
                    PRIMARY KEY (timestamp, query)
                )
            """)
            conn.commit()

            # Ignored queries table
            self.logger.debug("Creating ignored_queries table")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ignored_queries (
                    query TEXT PRIMARY KEY,
                    embedding BLOB,
                    reason TEXT
                )
            """)
            conn.commit()

            # Query cache table (for QueryProcessor)
            self.logger.debug("Creating query_cache table")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    tables TEXT,
                    columns TEXT,
                    confidence REAL
                )
            """)
            conn.commit()

            self._create_feedback_indexes()
            self._checkpoint_wal(conn)
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing database tables: {e}", exc_info=True)
            raise
        finally:
            cursor.close()

    def _create_feedback_indexes(self):
        """
        Create indexes on feedback.query and feedback.timestamp with fallback.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            self.logger.debug("Attempting to create index idx_feedback_query")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query)")
            conn.commit()
            self.logger.debug("Successfully created index idx_feedback_query")

            self.logger.debug("Attempting to create index idx_feedback_timestamp")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")
            conn.commit()
            self.logger.debug("Successfully created index idx_feedback_timestamp")
        except sqlite3.Error as e:
            self.logger.warning(f"Failed to create feedback indexes: {e}", exc_info=True)
            self.logger.info("Proceeding without indexes; performance may be affected")
        finally:
            cursor.close()

    def _checkpoint_wal(self, conn, mode: str = 'TRUNCATE'):
        """
        Perform a WAL checkpoint to manage WAL file size.

        Args:
            conn: SQLite connection object.
            mode: Checkpoint mode ('PASSIVE', 'TRUNCATE', 'FULL').
        """
        try:
            self.logger.debug(f"Performing {mode} WAL checkpoint")
            conn.execute(f"PRAGMA wal_checkpoint({mode})")
            self.logger.debug(f"{mode} WAL checkpoint completed")
        except sqlite3.Error as e:
            self.logger.warning(f"Error performing WAL checkpoint: {e}", exc_info=True)

    def normalize_query(self, query: str) -> str:
        """
        Normalize query for consistent storage and comparison.

        Args:
            query (str): Input query.

        Returns:
            str: Normalized query.
        """
        try:
            query = re.sub(r'\s+', ' ', query.strip().lower())
            query = re.sub(r'[^\w\s]', '', query)
            return query
        except Exception as e:
            self.logger.error(f"Error normalizing query '{query}': {e}", exc_info=True)
            return query.lower()

    def _embedding_to_blob(self, embedding: Optional[np.ndarray]) -> Optional[bytes]:
        """
        Convert numpy embedding to SQLite BLOB.

        Args:
            embedding (Optional[np.ndarray]): Query embedding.

        Returns:
            Optional[bytes]: BLOB representation.
        """
        try:
            if embedding is None:
                return None
            return embedding.tobytes()
        except Exception as e:
            self.logger.error(f"Error converting embedding to blob: {e}", exc_info=True)
            return None

    def _blob_to_embedding(self, blob: bytes, dim: int = 384) -> np.ndarray:
        """
        Convert SQLite BLOB to numpy embedding with dimension validation.

        Args:
            blob (bytes): BLOB data.
            dim (int): Expected embedding dimension (default: 384).

        Returns:
            np.ndarray: Embedding array.
        """
        try:
            embedding = np.frombuffer(blob, dtype=np.float32)
            if len(embedding) != dim:
                self.logger.error(f"Invalid embedding dimension: {len(embedding)}, expected {dim}")
                return np.zeros(dim, dtype=np.float32)
            return embedding.reshape(-1)
        except Exception as e:
            self.logger.error(f"Error converting blob to embedding: {e}", exc_info=True)
            return np.zeros(dim, dtype=np.float32)

    async def execute_with_retry(self, query: str, params: tuple = (), max_attempts: int = 5) -> None:
        """
        Execute a SQL query asynchronously with retry logic to handle database lock (Scenario 5).

        Args:
            query (str): SQL query.
            params (tuple): Query parameters.
            max_attempts (int): Maximum retry attempts.
        """
        attempt = 1
        while attempt <= max_attempts:
            try:
                async def execute():
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("BEGIN TRANSACTION")
                    if isinstance(params, list):
                        cursor.executemany(query, params)
                    else:
                        cursor.execute(query, params)
                    conn.commit()
                    cursor.close()
                await asyncio.to_thread(execute)
                self.logger.debug(f"Successfully executed query: {query}")
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_attempts:
                    self.logger.warning(f"Database locked on attempt {attempt}, retrying in {0.5 * attempt}s...")
                    await asyncio.sleep(0.5 * attempt)
                    attempt += 1
                else:
                    self.logger.error(f"Error executing query '{query}': {e}", exc_info=True)
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error executing query '{query}': {e}", exc_info=True)
                raise

    async def deduplicate_feedback(self, timeout: float = 60.0) -> int:
        """
        Remove duplicate feedback entries for the same normalized query in a single transaction (Scenario 8).

        Args:
            timeout (float): Maximum time (seconds) to wait for deduplication lock.

        Returns:
            int: Number of duplicate entries removed.
        """
        try:
            start_time = time.time()
            if not self.deduplication_lock.acquire(timeout=timeout):
                self.logger.error("Failed to acquire deduplication lock within timeout")
                raise TimeoutError("Deduplication lock timeout")

            async def deduplicate():
                conn = self._get_connection()
                cursor = conn.cursor()
                try:
                    self.logger.debug("Starting feedback deduplication")
                    self._checkpoint_wal(conn, mode='TRUNCATE')

                    # Count total feedback entries
                    cursor.execute("SELECT COUNT(*) FROM feedback")
                    total_entries = cursor.fetchone()[0]
                    self.logger.debug(f"Found {total_entries} feedback entries")

                    if total_entries > 200:
                        self.logger.warning(f"Large feedback table ({total_entries} entries), consider clearing")

                    # Create temporary table
                    cursor.execute("""
                        CREATE TEMPORARY TABLE temp_feedback (
                            timestamp TEXT,
                            query TEXT,
                            tables TEXT,
                            embedding BLOB,
                            count INTEGER,
                            PRIMARY KEY (timestamp, query)
                        )
                    """)
                    self.logger.debug("Created temporary table for deduplication")

                    # Insert deduplicated entries
                    cursor.execute("""
                        INSERT INTO temp_feedback (timestamp, query, tables, embedding, count)
                        SELECT MAX(timestamp) as timestamp, 
                               query, 
                               tables, 
                               embedding, 
                               SUM(count) as count
                        FROM feedback
                        GROUP BY query
                    """)
                    cursor.execute("SELECT COUNT(*) FROM temp_feedback")
                    deduplicated_count = cursor.fetchone()[0]
                    self.logger.debug(f"Inserted {deduplicated_count} deduplicated entries into temp_feedback")

                    # Calculate removed count
                    removed_count = total_entries - deduplicated_count
                    self.logger.debug(f"Will remove {removed_count} duplicate entries")

                    # Replace feedback table
                    cursor.execute("DELETE FROM feedback")
                    self.logger.debug("Cleared original feedback table")

                    cursor.execute("""
                        INSERT INTO feedback (timestamp, query, tables, embedding, count)
                        SELECT timestamp, query, tables, embedding, count
                        FROM temp_feedback
                    """)
                    self.logger.debug("Copied deduplicated entries back to feedback table")

                    # Drop temporary table
                    cursor.execute("DROP TABLE temp_feedback")
                    self.logger.debug("Dropped temporary table")

                    conn.commit()
                    self._checkpoint_wal(conn, mode='TRUNCATE')
                    self.logger.info(f"Removed {removed_count} duplicate feedback entries in {time.time() - start_time:.2f}s")
                    return removed_count
                except Exception as e:
                    conn.rollback()
                    self.logger.error(f"Error deduplicating feedback: {e}", exc_info=True)
                    raise
                finally:
                    cursor.close()

            return await asyncio.to_thread(deduplicate)
        except Exception as e:
            self.logger.error(f"Error deduplicating feedback: {e}", exc_info=True)
            raise
        finally:
            if self.deduplication_lock.locked():
                self.deduplication_lock.release()

    async def write_weights(self, weights: Dict[str, Dict[str, float]], batch_size: int = 100):
        """
        Write weights to SQLite database in batches (Scenario 5).

        Args:
            weights (Dict[str, Dict[str, float]]): Table-column weight mappings.
            batch_size (int): Batch size for inserts.
        """
        try:
            all_entries = [(table, col, weight) for table, cols in weights.items() for col, weight in cols.items()]
            total_entries = len(all_entries)
            self.logger.debug(f"Writing {total_entries} weight entries to SQLite")

            for i in range(0, total_entries, batch_size):
                batch = all_entries[i:i + batch_size]
                self.logger.debug(f"Writing weights batch {i//batch_size + 1} ({len(batch)} entries)")
                await self.execute_with_retry(
                    "INSERT OR REPLACE INTO weights (table_name, column, weight) VALUES (?, ?, ?)",
                    batch
                )
            conn = self._get_connection()
            self._checkpoint_wal(conn)
        except sqlite3.Error as e:
            self.logger.error(f"Error writing weights: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error writing weights: {e}", exc_info=True)
            raise

    def load_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Load weights from SQLite database (Scenario 5).

        Returns:
            Dict[str, Dict[str, float]]: Table-column weights.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT table_name, column, weight FROM weights")
            weights = {}
            for table, col, weight in cursor.fetchall():
                if table not in weights:
                    weights[table] = {}
                weights[table][col] = weight
            self.logger.debug(f"Loaded {sum(len(cols) for cols in weights.values())} weight entries from SQLite")
            return weights
        except sqlite3.Error as e:
            self.logger.error(f"Error loading weights: {e}", exc_info=True)
            return {}
        finally:
            cursor.close()

    def read_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Alias for load_weights (Scenario 5).

        Returns:
            Dict[str, Dict[str, float]]: Table-column weights.
        """
        return self.load_weights()

    async def write_name_matches(self, name_matches: Dict[str, List[str]], source: str = 'default', batch_size: int = 100):
        """
        Write name matches to SQLite database in batches (Scenario 2).

        Args:
            name_matches (Dict[str, List[str]]): Column-synonym mappings.
            source (str): Source identifier (default: 'default').
            batch_size (int): Batch size for inserts.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            self.logger.debug("Clearing existing name_matches")
            cursor.execute("DELETE FROM name_matches")
            conn.commit()

            all_entries = [(col, syn) for col, synonyms in name_matches.items() for syn in synonyms]
            total_entries = len(all_entries)
            self.logger.debug(f"Writing {total_entries} name match entries to SQLite")

            for i in range(0, total_entries, batch_size):
                batch = all_entries[i:i + batch_size]
                self.logger.debug(f"Writing name matches batch {i//batch_size + 1} ({len(batch)} entries)")
                await self.execute_with_retry(
                    "INSERT OR REPLACE INTO name_matches (column_name, synonym) VALUES (?, ?)",
                    batch
                )
            self._checkpoint_wal(conn)
            self.logger.debug(f"Wrote {total_entries} name match entries to SQLite")
        except sqlite3.Error as e:
            self.logger.error(f"Error writing name matches: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error writing name matches: {e}", exc_info=True)
            raise
        finally:
            cursor.close()

    def load_name_matches(self) -> Dict[str, List[str]]:
        """
        Load all name matches from SQLite database (Scenario 2).

        Returns:
            Dict[str, List[str]]: Column-synonym mappings.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT column_name, synonym FROM name_matches")
            name_matches = {}
            for col, syn in cursor.fetchall():
                if col not in name_matches:
                    name_matches[col] = []
                name_matches[col].append(syn)
            for col in name_matches:
                name_matches[col] = list(set(name_matches[col]))
            self.logger.debug(f"Loaded {sum(len(syns) for syns in name_matches.values())} name match entries from SQLite")
            return name_matches
        except sqlite3.Error as e:
            self.logger.error(f"Error loading name matches: {e}", exc_info=True)
            return {}
        finally:
            cursor.close()

    def read_name_matches(self, source: str = 'default') -> Dict[str, List[str]]:
        """
        Read name matches from SQLite database (Scenario 2).

        Args:
            source (str): Source identifier (default: 'default').

        Returns:
            Dict[str, List[str]]: Column-synonym mappings.
        """
        return self.load_name_matches()

    async def write_feedback(self, timestamp: str, query: str, tables: List[str], embedding: np.ndarray, count: int = 1):
        """
        Write feedback to SQLite database, checking for near-duplicates (Scenarios 3, 8).

        Args:
            timestamp (str): Feedback timestamp.
            query (str): User query.
            tables (List[str]): Matched tables (SQL Server or S3).
            embedding (np.ndarray): Query embedding.
            count (int): Feedback count (default: 1).
        """
        try:
            normalized_query = self.normalize_query(query)
            similar_feedback = await self.find_similar_feedback(query, threshold=0.90)
            for fb_query, fb_tables, sim in similar_feedback:
                if sim > 0.90 and set(fb_tables) == set(tables):
                    await self.update_feedback_count(fb_query, increment=count)
                    self.logger.debug(f"Updated existing feedback for query '{fb_query}' (sim={sim:.2f})")
                    return

            embedding_blob = self._embedding_to_blob(embedding)
            tables_json = json.dumps(tables)
            await self.execute_with_retry(
                "INSERT OR REPLACE INTO feedback (timestamp, query, tables, embedding, count) VALUES (?, ?, ?, ?, ?)",
                (timestamp, normalized_query, tables_json, embedding_blob, count)
            )
            self.logger.debug(f"Wrote feedback for query: {normalized_query}, tables: {tables}")
        except sqlite3.Error as e:
            self.logger.error(f"Error writing feedback for query '{query}': {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error writing feedback for query '{query}': {e}", exc_info=True)
            raise

    async def delete_feedback(self, query: str):
        """
        Delete feedback entry from SQLite database (Scenario 3).

        Args:
            query (str): Query to delete.
        """
        try:
            normalized_query = self.normalize_query(query)
            await self.execute_with_retry("DELETE FROM feedback WHERE query = ?", (normalized_query,))
            self.logger.debug(f"Deleted feedback for query: {normalized_query}")
        except sqlite3.Error as e:
            self.logger.error(f"Error deleting feedback for query '{query}': {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error deleting feedback for query '{query}': {e}", exc_info=True)
            raise

    def get_feedback(self, query: Optional[str] = None) -> List[Tuple[str, str, List[str], np.ndarray, int]]:
        """
        Retrieve feedback entries from SQLite database (Scenario 3).

        Args:
            query (Optional[str]): Specific query to retrieve (default: None, retrieves all).

        Returns:
            List[Tuple[str, str, List[str], np.ndarray, int]]: Feedback data
                (timestamp, query, tables, embedding, count).
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if query:
                normalized_query = self.normalize_query(query)
                cursor.execute(
                    "SELECT timestamp, query, tables, embedding, count FROM feedback WHERE query = ?",
                    (normalized_query,)
                )
            else:
                cursor.execute("SELECT timestamp, query, tables, embedding, count FROM feedback")
            feedback = []
            for timestamp, query, tables_json, embedding_blob, count in cursor.fetchall():
                try:
                    tables = json.loads(tables_json)
                    embedding = self._blob_to_embedding(embedding_blob) if embedding_blob else np.zeros(384, dtype=np.float32)
                    feedback.append((timestamp, query, tables, embedding, count))
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error decoding tables JSON for query '{query}': {e}", exc_info=True)
                    continue
            self.logger.debug(f"Retrieved {len(feedback)} feedback entries")
            return feedback
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving feedback: {e}", exc_info=True)
            return []
        finally:
            cursor.close()

    def read_feedback(self) -> Dict[str, Dict]:
        """
        Read feedback entries from SQLite database (Scenario 3).

        Returns:
            Dict[str, Dict]: Feedback entries with timestamp as key.
        """
        try:
            feedback = self.get_feedback()
            feedback_dict = {}
            for timestamp, query, tables, _, count in feedback:
                feedback_dict[timestamp] = {
                    'query': query,
                    'tables': tables,
                    'count': count
                }
            self.logger.debug(f"Loaded {len(feedback_dict)} feedback entries from SQLite")
            return feedback_dict
        except Exception as e:
            self.logger.error(f"Error reading feedback: {e}", exc_info=True)
            return {}

    async def update_feedback_count(self, query: str, increment: int = 1):
        """
        Increment the count for a feedback entry (Scenario 8).

        Args:
            query (str): Query to update.
            increment (int): Count increment (default: 1).
        """
        try:
            normalized_query = self.normalize_query(query)
            await self.execute_with_retry(
                "UPDATE feedback SET count = count + ? WHERE query = ?",
                (increment, normalized_query)
            )
            self.logger.debug(f"Updated feedback count for query: {normalized_query}")
        except sqlite3.Error as e:
            self.logger.error(f"Error updating feedback count for query '{query}': {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error updating feedback count for query '{query}': {e}", exc_info=True)
            raise

    async def write_ignored_query(self, query: str, embedding: Optional[np.ndarray], reason: str):
        """
        Write ignored query to SQLite database, skipping short queries (Scenario 6).

        Args:
            query (str): Ignored query.
            embedding (Optional[np.ndarray]): Query embedding.
            reason (str): Reason for ignoring.
        """
        try:
            if not query or len(query.strip()) < 3:
                self.logger.debug(f"Skipping invalid query: '{query}' (too short or empty)")
                return
            normalized_query = self.normalize_query(query)
            if embedding is None and self.model:
                try:
                    embedding = self.model.encode(normalized_query, show_progress_bar=False)
                except Exception as e:
                    self.logger.error(f"Error generating embedding for query '{normalized_query}': {e}", exc_info=True)
                    embedding = None
            embedding_blob = self._embedding_to_blob(embedding)
            await self.execute_with_retry(
                "INSERT OR REPLACE INTO ignored_queries (query, embedding, reason) VALUES (?, ?, ?)",
                (normalized_query, embedding_blob, reason)
            )
            self.logger.debug(f"Wrote ignored query: {normalized_query}")
        except sqlite3.Error as e:
            self.logger.error(f"Error writing ignored query '{query}': {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error writing ignored query '{query}': {e}", exc_info=True)
            raise

    def get_ignored_queries(self) -> List[Tuple[str, np.ndarray, str]]:
        """
        Retrieve ignored queries from SQLite database (Scenario 6).

        Returns:
            List[Tuple[str, np.ndarray, str]]: Ignored queries, embeddings, and reasons.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT query, embedding, reason FROM ignored_queries")
            ignored = []
            for query, embedding_blob, reason in cursor.fetchall():
                embedding = self._blob_to_embedding(embedding_blob) if embedding_blob else np.zeros(384, dtype=np.float32)
                ignored.append((query, embedding, reason))
            self.logger.debug(f"Retrieved {len(ignored)} ignored queries")
            return ignored
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving ignored queries: {e}", exc_info=True)
            return []
        finally:
            cursor.close()

    async def is_query_ignored(self, query: str, threshold: float = 0.85) -> Optional[str]:
        """
        Check if a query matches an ignored query based on embedding similarity (Scenario 6).

        Args:
            query (str): Query to check.
            threshold (float): Similarity threshold (default: 0.85).

        Returns:
            Optional[str]: Reason for ignoring if matched, else None.
        """
        try:
            if not self.model:
                self.logger.warning("No model available for ignored query check")
                return None
            normalized_query = self.normalize_query(query)
            query_embedding = self.model.encode(normalized_query, show_progress_bar=False)
            ignored_queries = self.get_ignored_queries()
            for ignored_query, ignored_embedding, reason in ignored_queries:
                similarity = float(util.cos_sim(query_embedding, ignored_embedding)[0][0])
                if similarity > threshold:
                    self.logger.debug(f"Query '{normalized_query}' matches ignored query '{ignored_query}' (sim={similarity:.2f}, reason={reason})")
                    return reason
            self.logger.debug(f"No ignored query match for '{normalized_query}'")
            return None
        except Exception as e:
            self.logger.error(f"Error checking ignored query '{query}': {e}", exc_info=True)
            return None

    async def delete_ignored_query(self, query: str):
        """
        Delete an ignored query from SQLite database (Scenario 6).

        Args:
            query (str): Query to delete.
        """
        try:
            normalized_query = self.normalize_query(query)
            await self.execute_with_retry("DELETE FROM ignored_queries WHERE query = ?", (normalized_query,))
            self.logger.debug(f"Deleted ignored query: {normalized_query}")
        except sqlite3.Error as e:
            self.logger.error(f"Error deleting ignored query '{query}': {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error deleting ignored query '{query}': {e}", exc_info=True)
            raise

    async def clear_ignored_queries(self):
        """
        Clear all ignored queries from SQLite database (Scenario 6).
        """
        try:
            await self.execute_with_retry("DELETE FROM ignored_queries")
            self.logger.debug("Cleared all ignored queries")
        except sqlite3.Error as e:
            self.logger.error(f"Error clearing ignored queries: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error clearing ignored queries: {e}", exc_info=True)
            raise

    def read_ignored_queries(self) -> Dict[str, Dict[str, str]]:
        """
        Read ignored queries from SQLite database (Scenario 6).

        Returns:
            Dict[str, Dict[str, str]]: Ignored queries with metadata.
        """
        try:
            ignored_queries = self.get_ignored_queries()
            ignored_dict = {}
            for query, _, reason in ignored_queries:
                ignored_dict[query] = {
                    'schema_name': None,
                    'reason': reason
                }
            self.logger.debug(f"Loaded {len(ignored_dict)} ignored queries from SQLite")
            return ignored_dict
        except Exception as e:
            self.logger.error(f"Error reading ignored queries: {e}", exc_info=True)
            return {}

    async def migrate_file_caches(self):
        """
        Migrate legacy file-based caches to SQLite (Scenario 5).
        """
        try:
            config_dir = os.path.join(self.display_name, self.db_name)
            weights_file = os.path.join(config_dir, "weights.json")
            name_matches_file = os.path.join(config_dir, "name_matches.json")
            feedback_file = os.path.join(config_dir, "feedback.json")
            ignored_file = os.path.join(config_dir, "ignored_queries.json")

            if os.path.exists(weights_file):
                try:
                    with open(weights_file) as f:
                        weights = json.load(f)
                    await self.write_weights(weights)
                    os.rename(weights_file, f"{weights_file}.bak")
                    self.logger.debug(f"Migrated weights from {weights_file}")
                except Exception as e:
                    self.logger.warning(f"Error migrating weights: {e}", exc_info=True)

            if os.path.exists(name_matches_file):
                try:
                    with open(name_matches_file) as f:
                        name_matches = json.load(f)
                    await self.write_name_matches(name_matches)
                    os.rename(name_matches_file, f"{name_matches_file}.bak")
                    self.logger.debug(f"Migrated name matches from {name_matches_file}")
                except Exception as e:
                    self.logger.warning(f"Error migrating name matches: {e}", exc_info=True)

            if os.path.exists(feedback_file):
                try:
                    with open(feedback_file) as f:
                        feedback = json.load(f)
                    for timestamp, entry in feedback.items():
                        normalized_query = self.normalize_query(entry['query'])
                        embedding = np.zeros(384, dtype=np.float32)
                        await self.write_feedback(timestamp, normalized_query, entry['tables'], embedding, entry.get('count', 1))
                    os.rename(feedback_file, f"{feedback_file}.bak")
                    self.logger.debug(f"Migrated feedback from {feedback_file}")
                except Exception as e:
                    self.logger.warning(f"Error migrating feedback: {e}", exc_info=True)

            if os.path.exists(ignored_file):
                try:
                    with open(ignored_file) as f:
                        ignored = json.load(f)
                    for query, info in ignored.items():
                        normalized_query = self.normalize_query(query)
                        embedding = np.zeros(384, dtype=np.float32)
                        await self.write_ignored_query(normalized_query, embedding, info.get('reason', 'unknown'))
                    os.rename(ignored_file, f"{ignored_file}.bak")
                    self.logger.debug(f"Migrated ignored queries from {ignored_file}")
                except Exception as e:
                    self.logger.warning(f"Error migrating ignored queries: {e}", exc_info=True)

            self.logger.info("Completed file cache migration to SQLite")
        except Exception as e:
            self.logger.error(f"Error migrating file caches: {e}", exc_info=True)
            raise

    async def reload_caches(self, schema_manager, feedback_manager, name_match_manager):
        """
        Reload caches for schema, feedback, and name matches (Scenario 5).

        Args:
            schema_manager: SchemaManager instance.
            feedback_manager: FeedbackManager instance.
            name_match_manager: NameMatchManager instance.
        """
        try:
            self.logger.debug("Reloading caches")
            await self.clear_cache(table='weights')
            await self.clear_cache(table='name_matches')
            await self.clear_cache(table='ignored_queries')
            self.logger.debug("Preserving feedback during cache reload")
            self.logger.info("Caches reloaded successfully")
        except Exception as e:
            self.logger.error(f"Error reloading caches: {e}", exc_info=True)
            raise

    async def clear_cache(self, table: Optional[str] = None):
        """
        Clear specific cache table or all tables except feedback (Scenario 5).

        Args:
            table (Optional[str]): Specific table to clear (default: None, clears weights, name_matches, ignored_queries).
        """
        try:
            tables = ['weights', 'name_matches', 'ignored_queries']
            if table and table in tables:
                await self.execute_with_retry(f"DELETE FROM {table}")
                self.logger.debug(f"Cleared cache table: {table}")
            else:
                for t in tables:
                    await self.execute_with_retry(f"DELETE FROM {t}")
                self.logger.debug("Cleared weights, name_matches, and ignored_queries tables")
        except sqlite3.Error as e:
            self.logger.error(f"Error clearing cache: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error clearing cache: {e}", exc_info=True)
            raise

    def validate_cache(self) -> bool:
        """
        Validate cache integrity by checking table existence and data consistency (Scenario 5).

        Returns:
            bool: True if cache is valid, False otherwise.
        """
        try:
            tables = ['weights', 'name_matches', 'feedback', 'ignored_queries', 'query_cache']
            conn = self._get_connection()
            cursor = conn.cursor()
            for table in tables:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if not cursor.fetchone():
                    self.logger.error(f"Cache table missing: {table}")
                    return False
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                self.logger.debug(f"Table {table} has {count} entries")
            self.logger.debug("Cache validation successful")
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Error validating cache: {e}", exc_info=True)
            return False
        finally:
            cursor.close()

    def count_feedback(self) -> int:
        """
        Count the number of feedback entries in the cache (Scenario 8).

        Returns:
            int: Number of feedback entries.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feedback")
            count = cursor.fetchone()[0]
            self.logger.debug(f"Counted {count} feedback entries")
            return count
        except sqlite3.Error as e:
            self.logger.error(f"Error counting feedback: {e}", exc_info=True)
            return 0
        finally:
            cursor.close()

    async def find_similar_feedback(self, query: str, threshold: float = 0.90) -> List[Tuple[str, List[str], float]]:
        """
        Find feedback entries similar to the given query based on embedding similarity (Scenarios 2, 3).

        Args:
            query (str): Query to compare.
            threshold (float): Similarity threshold (default: 0.90).

        Returns:
            List[Tuple[str, List[str], float]]: Similar queries, tables, and similarity scores.
        """
        try:
            if not self.model:
                self.logger.warning("No model available for similarity comparison")
                return []
            normalized_query = self.normalize_query(query)
            query_embedding = self.model.encode(normalized_query, show_progress_bar=False)
            feedback = self.get_feedback()
            similar = []
            for _, fb_query, tables, fb_embedding, _ in feedback:
                similarity = float(util.cos_sim(query_embedding, fb_embedding)[0][0])
                if similarity > threshold:
                    similar.append((fb_query, tables, similarity))
            similar.sort(key=lambda x: x[2], reverse=True)
            self.logger.debug(f"Found {len(similar)} similar feedback entries for query: {normalized_query}")
            return similar
        except Exception as e:
            self.logger.error(f"Error finding similar feedback for query '{query}': {e}", exc_info=True)
            return []

    async def cache_result(self, cache_key: str, result: Tuple[List[str], List[str], float]):
        """
        Cache query processing result (Scenario 5).

        Args:
            cache_key (str): Unique key for query.
            result (Tuple[List[str], List[str], float]): Tables, columns, confidence.
        """
        try:
            tables, columns, confidence = result
            tables_json = json.dumps(tables)
            columns_json = json.dumps(columns)
            await self.execute_with_retry(
                "INSERT OR REPLACE INTO query_cache (cache_key, tables, columns, confidence) VALUES (?, ?, ?, ?)",
                (cache_key, tables_json, columns_json, confidence)
            )
            self.logger.debug(f"Cached result for key: {cache_key}")
        except sqlite3.Error as e:
            self.logger.error(f"Error caching result for key {cache_key}: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error caching result for key {cache_key}: {e}", exc_info=True)
            raise

    def get_cached_result(self, cache_key: str) -> Optional[Tuple[List[str], List[str], float]]:
        """
        Retrieve cached query result (Scenario 5).

        Args:
            cache_key (str): Unique key for query.

        Returns:
            Optional[Tuple[List[str], List[str], float]]: Cached tables, columns, confidence, or None.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT tables, columns, confidence FROM query_cache WHERE cache_key = ?", (cache_key,))
            result = cursor.fetchone()
            if result:
                tables, columns, confidence = result
                self.logger.debug(f"Retrieved cached result for key: {cache_key}")
                return json.loads(tables), json.loads(columns), confidence
            return None
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving cached result for key {cache_key}: {e}", exc_info=True)
            return None
        finally:
            cursor.close()

    def close(self):
        """
        Close SQLite database connection and deduplicate feedback, idempotent.
        """
        try:
            with self.lock:
                if self._conn is not None and not getattr(self._conn, 'closed', True):
                    try:
                        asyncio.run(self.deduplicate_feedback(timeout=60.0))
                        self._checkpoint_wal(self._conn)
                        self._conn.commit()
                    finally:
                        self._conn.close()
                        self._conn = None
                        self.logger.debug(f"Closed SQLite connection and deduplicated feedback for {self.display_name}/{self.db_name}")
                else:
                    self.logger.debug("No open SQLite connection to close")
        except Exception as e:
            self.logger.error(f"Error closing CacheSynchronizer: {e}", exc_info=True)
            raise