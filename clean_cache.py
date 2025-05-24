"""
Cleans SQLite feedback cache and JSON cache files for TIA-1.2 to optimize performance and ensure consistency.
Deduplicates feedback entries, ensures indexes, and cleans schema/training data cache.
Preserves TIA-1.1 functionality, fixes performance and path issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based paths (e.g., BIKE-STORES/BikeStores/cache.db, BIKE-STORES/BikeStores/schema_cache/)
- Async SQLite and file operations for <5s processing
- Logging to codebase/logs/DISPLAY_NAME_app.log
- Non-hardcoding with display_name and db_name
- S3 cache validation
- Force, dry-run, and feedback-only modes
Addresses Scenarios 1 (S3 support), 5 (performance), and 8 (feedback deduplication).
"""
import os
import json
import sqlite3
import asyncio
import logging
import shutil
import pathlib
import argparse
import re
from datetime import datetime, timedelta
from typing import Optional
from schema.manager import SchemaManager
from config.logging_setup import setup_logging

async def normalize_query(query: str) -> str:
    """
    Normalize query for deduplication by removing extra spaces and punctuation (Scenario 8).

    Args:
        query (str): Input query.

    Returns:
        str: Normalized query.
    """
    query = re.sub(r'\s+', ' ', query.strip().lower())
    query = re.sub(r'[^\w\s]', '', query)
    return query

async def deduplicate_feedback(db_path: str, logger: logging.Logger, dry_run: bool = False) -> int:
    """
    Deduplicate feedback entries in SQLite database asynchronously (Scenario 8).

    Identifies duplicates by normalized query, retains the most recent entry,
    aggregates counts, and removes older entries.

    Args:
        db_path (str): Path to SQLite database (e.g., BIKE-STORES/BikeStores/cache.db).
        logger (logging.Logger): Logger instance.
        dry_run (bool): List actions without executing.

    Returns:
        int: Number of duplicates removed.

    Raises:
        sqlite3.Error: If database operations fail after retries.
    """
    async def execute_with_retry(conn: sqlite3.Connection, query: str, params: tuple = (), max_attempts: int = 5) -> None:
        """Execute SQLite query with retry on locking errors."""
        attempt = 1
        while attempt <= max_attempts:
            try:
                cursor = conn.cursor()
                cursor.execute("BEGIN TRANSACTION")
                cursor.execute(query, params)
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                conn.rollback()
                if "database is locked" in str(e) and attempt < max_attempts:
                    await asyncio.sleep(0.5 * attempt)
                    attempt += 1
                else:
                    raise
            finally:
                cursor.close()

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")  # Enable Write-Ahead Logging
        cursor = conn.cursor()

        # Find duplicate queries
        cursor.execute("""
            SELECT query, COUNT(*) as entry_count
            FROM feedback
            GROUP BY query
            HAVING entry_count > 1
        """)
        duplicate_queries = cursor.fetchall()
        removed_count = 0

        for query, count in duplicate_queries:
            logger.info(f"Found {count} duplicate entries for query: {query}")
            normalized_query = await normalize_query(query)
            
            # Get all entries for the query, sorted by timestamp
            cursor.execute("""
                SELECT timestamp, tables, embedding, count
                FROM feedback
                WHERE query = ?
                ORDER BY timestamp DESC
            """, (normalized_query,))
            entries = cursor.fetchall()

            if not entries:
                continue

            # Retain most recent entry
            most_recent = entries[0]
            recent_timestamp, recent_tables, recent_embedding, recent_count = most_recent
            total_count = sum(entry[3] for entry in entries)

            if dry_run:
                logger.info(f"[Dry Run] Would update query '{query}' (timestamp: {recent_timestamp}, count: {total_count}), remove {len(entries)-1} older entries")
                removed_count += len(entries) - 1
                continue

            # Update most recent entry with aggregated count
            await execute_with_retry(
                conn,
                "UPDATE feedback SET count = ? WHERE query = ? AND timestamp = ?",
                (total_count, normalized_query, recent_timestamp)
            )

            # Remove older entries
            for entry in entries[1:]:
                old_timestamp = entry[0]
                await execute_with_retry(
                    conn,
                    "DELETE FROM feedback WHERE query = ? AND timestamp = ?",
                    (normalized_query, old_timestamp)
                )
                removed_count += 1
            logger.debug(f"Retained entry for query '{query}' with timestamp {recent_timestamp}, total count={total_count}")

        cursor.close()
        logger.info(f"Removed {removed_count} duplicate feedback entries")
        return removed_count

    except sqlite3.Error as e:
        logger.error(f"Error deduplicating feedback: {e}", exc_info=True)
        raise
    finally:
        conn.close()

async def ensure_index(db_path: str, logger: logging.Logger, dry_run: bool = False) -> bool:
    """
    Ensure indexes exist on feedback table for query and timestamp columns (Scenario 5).

    Creates idx_feedback_query and idx_feedback_timestamp if missing.

    Args:
        db_path (str): Path to SQLite database.
        logger (logging.Logger): Logger instance.
        dry_run (bool): List actions without executing.

    Returns:
        bool: True if indexes exist or created, False on failure.

    Raises:
        sqlite3.Error: If index creation fails.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()

        # Check and create index on query
        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='index' AND name='idx_feedback_query'
        """)
        if cursor.fetchone():
            logger.debug("Index idx_feedback_query already exists")
        elif dry_run:
            logger.info("[Dry Run] Would create index idx_feedback_query")
        else:
            cursor.execute("CREATE INDEX idx_feedback_query ON feedback(query)")
            conn.commit()
            logger.info("Created index idx_feedback_query")

        # Check and create index on timestamp
        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='index' AND name='idx_feedback_timestamp'
        """)
        if cursor.fetchone():
            logger.debug("Index idx_feedback_timestamp already exists")
        elif dry_run:
            logger.info("[Dry Run] Would create index idx_feedback_timestamp")
        else:
            cursor.execute("CREATE INDEX idx_feedback_timestamp ON feedback(timestamp)")
            conn.commit()
            logger.info("Created index idx_feedback_timestamp")

        cursor.close()
        return True

    except sqlite3.Error as e:
        logger.error(f"Error creating indexes: {e}", exc_info=True)
        return False
    finally:
        conn.close()

async def clean_cache(
    display_name: str,
    db_name: str,
    connection: Optional[object] = None,
    force: bool = False,
    dry_run: bool = False,
    backup: bool = True,
    feedback_only: bool = False
) -> bool:
    """
    Clean SQLite feedback cache and JSON cache files (Scenarios 1, 5, 8).

    Deduplicates feedback entries, ensures indexes, and cleans schema/training data cache.

    Args:
        display_name (str): Display name (e.g., BIKE-STORES).
        db_name (str): Database name (e.g., BikeStores).
        connection (Optional[object]): SQL Server connection or S3 client for JSON cache validation.
        force (bool): Delete all cache files without validation.
        dry_run (bool): List actions without executing.
        backup (bool): Back up JSON files before deletion.
        feedback_only (bool): Clean SQLite feedback cache only.

    Returns:
        bool: True if cleanup succeeds, False otherwise.
    """
    setup_logging(display_name, db_name)
    logger = logging.getLogger("clean_cache")
    logger.info(f"Starting cache cleanup for {display_name}/{db_name}, force={force}, dry_run={dry_run}, backup={backup}, feedback_only={feedback_only}")

    try:
        # Initialize SchemaManager for JSON cache validation
        schema_manager = SchemaManager(display_name, db_name) if connection and not feedback_only else None

        # SQLite feedback cache
        db_path = pathlib.Path(display_name, db_name, "cache.db")
        feedback_cleaned = False
        if db_path.exists():
            if not os.access(db_path, os.R_OK | os.W_OK):
                logger.error(f"Insufficient permissions for cache database: {db_path}")
                return False

            # Deduplicate feedback
            removed = await deduplicate_feedback(db_path, logger, dry_run)
            logger.info(f"Deduplicated feedback, removed {removed} entries")

            # Ensure indexes
            if await ensure_index(db_path, logger, dry_run):
                logger.info("Feedback index setup complete")
            else:
                logger.error("Failed to set up feedback indexes")
                return False

            feedback_cleaned = True
        elif feedback_only:
            logger.error(f"Cache database not found: {db_path}")
            return False
        else:
            logger.debug(f"Cache database not found, skipping feedback cleanup: {db_path}")

        if feedback_only:
            return feedback_cleaned

        # JSON cache directories
        schema_cache_dir = pathlib.Path(display_name, db_name, "schema_cache")
        training_data_dir = pathlib.Path(display_name, db_name, "training_data")
        backup_dir = pathlib.Path(display_name, db_name, "cache_backup", datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Collect JSON cache files
        cache_files = []
        for cache_dir in [schema_cache_dir, training_data_dir]:
            if cache_dir.exists():
                cache_files.extend([f for f in cache_dir.rglob("*.json") if f.is_file()])

        if not cache_files and feedback_cleaned:
            logger.info("No JSON cache files found, feedback cache cleaned")
            return True
        elif not cache_files:
            logger.info("No cache files found")
            return True

        logger.debug(f"Found {len(cache_files)} JSON cache files: {[str(f) for f in cache_files]}")

        # Process JSON cache files
        deleted_files = []
        for cache_file in cache_files:
            if force or await validate_cache_file(cache_file, schema_manager, connection, logger):
                if dry_run:
                    logger.info(f"[Dry Run] Would delete JSON cache file: {cache_file}")
                    continue

                if backup:
                    await backup_cache(cache_file, backup_dir, logger)

                await asyncio.to_thread(cache_file.unlink, missing_ok=True)
                deleted_files.append(cache_file)
                logger.info(f"Deleted JSON cache file: {cache_file}")
            else:
                logger.debug(f"Retained valid JSON cache file: {cache_file}")

        # Clean empty JSON directories
        for cache_dir in [schema_cache_dir, training_data_dir]:
            if cache_dir.exists() and not any(cache_dir.iterdir()):
                await asyncio.to_thread(shutil.rmtree, cache_dir, ignore_errors=True)
                logger.info(f"Removed empty JSON cache directory: {cache_dir}")

        logger.info(f"Cache cleanup completed: {len(deleted_files)} JSON files deleted, feedback cache cleaned")
        return True
    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}", exc_info=True)
        return False

async def validate_cache_file(
    cache_file: pathlib.Path,
    schema_manager: Optional[SchemaManager],
    connection: Optional[object],
    logger: logging.Logger
) -> bool:
    """
    Validate JSON cache file by age and content (Scenario 5).

    Args:
        cache_file (pathlib.Path): Cache file to validate.
        schema_manager (Optional[SchemaManager]): SchemaManager for schema validation.
        connection (Optional[object]): Connection for schema validation.
        logger (logging.Logger): Logger instance.

    Returns:
        bool: True if file should be deleted, False if valid.
    """
    try:
        # Check file age (>7 days)
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(days=7):
            logger.debug(f"JSON cache file {cache_file} is older than 7 days")
            return True

        # Check JSON integrity
        async with asyncio.Lock():
            with cache_file.open('r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in cache file: {cache_file}")
                    return True

        # Validate schema cache if SchemaManager and connection are provided
        if schema_manager and connection and 'schema.json' in cache_file.name:
            is_valid = await schema_manager.validate_cache_consistency(connection)
            if not is_valid:
                logger.warning(f"Schema cache inconsistent: {cache_file}")
                return True

        return False
    except Exception as e:
        logger.error(f"Error validating JSON cache file {cache_file}: {e}", exc_info=True)
        return True  # Delete on error to avoid stale cache

async def backup_cache(cache_file: pathlib.Path, backup_dir: pathlib.Path, logger: logging.Logger):
    """
    Back up JSON cache file before deletion (Scenario 5).

    Args:
        cache_file (pathlib.Path): Cache file to back up.
        backup_dir (pathlib.Path): Backup directory.
        logger (logging.Logger): Logger instance.
    """
    try:
        backup_path = backup_dir / cache_file.relative_to(cache_file.parents[2])
        await asyncio.to_thread(backup_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, cache_file, backup_path)
        logger.debug(f"Backed up JSON cache file to: {backup_path}")
    except Exception as e:
        logger.error(f"Error backing up JSON cache file {cache_file}: {e}", exc_info=True)

if __name__ == "__main__":
    async def main():
        parser = argparse.ArgumentParser(description="Clean TIA-1.2 cache (SQLite feedback and JSON files)")
        parser.add_argument("--display-name", default="BIKE-STORES", help="Display name (e.g., BIKE-STORES)")
        parser.add_argument("--db-name", default="BikeStores", help="Database name (e.g., BikeStores)")
        parser.add_argument("--force", action="store_true", help="Delete all JSON cache files without validation")
        parser.add_argument("--dry-run", action="store_true", help="List actions without executing")
        parser.add_argument("--no-backup", action="store_false", dest="backup", help="Skip JSON cache backup")
        parser.add_argument("--feedback-only", action="store_true", help="Clean SQLite feedback cache only")
        args = parser.parse_args()

        success = await clean_cache(
            args.display_name,
            args.db_name,
            force=args.force,
            dry_run=args.dry_run,
            backup=args.backup,
            feedback_only=args.feedback_only
        )
        print(f"Cache cleanup {'succeeded' if success else 'failed'}")

    asyncio.run(main())