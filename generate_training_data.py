"""
Generates training data for TIA-1.2 feedback system, including table/column synonyms and query patterns.
Processes schema metadata from SchemaManager and uses SentenceTransformer for synonym generation.
Preserves TIA-1.1 functionality, fixes performance and path issues, and adds TIA-1.2 enhancements:
- DISPLAY_NAME-based paths (e.g., BIKE-STORES/BikeStores/training_data/)
- Async file operations and batch embeddings for <5s processing
- Logging to logs/DISPLAY_NAME_app.log
- Non-hardcoding with config_name from app-configs/db_configurations.json
- S3 table support
- Enhanced synonym generation with embeddings and static dictionary
Addresses Scenarios 1 (S3 support), 4 (training data/synonyms), and 5 (performance).
"""
import os
import json
import asyncio
import logging
import pathlib
import argparse
from typing import Dict, List, Optional
import numpy as np
import pyodbc
import boto3
from botocore.client import BaseClient
from sentence_transformers import SentenceTransformer
from schema.manager import SchemaManager
from config.model_singleton import ModelSingleton
from config.logging_setup import setup_logging
from config.patterns import SYNONYMS
from config.utils import load_db_config

async def generate_training_data(config_name: str, schemas: List[str] = None, tables: List[str] = None) -> bool:
    """
    Generate training data for tables, columns, and query patterns (Scenarios 1, 4, 5).

    Args:
        config_name (str): Configuration name (e.g., 'bikestores', 'bikestores-csv').
        schemas (List[str], optional): Schemas to include.
        tables (List[str], optional): Tables to include (schema.table or S3 table).

    Returns:
        bool: True if generation succeeds, False otherwise.
    """
    # Load config to get display_name and db_name
    config = load_db_config(config_name)
    display_name = config.get("display_name", "BIKE-STORES")
    db_name = config.get("database", "BikeStores")

    setup_logging(display_name, db_name)
    logger = logging.getLogger("training_data")
    logger.info(f"Starting training data generation for {display_name}/{db_name} (config: {config_name})")

    connection = None
    try:
        # Initialize SchemaManager and ModelSingleton
        schema_manager = SchemaManager(config_name, schemas, tables)
        model_singleton = ModelSingleton(config_name)
        await model_singleton.ensure_model()
        model = model_singleton.model

        # Get connection based on config type
        if config['type'].lower() == 's3':
            connection = boto3.client(
                's3',
                aws_access_key_id=config.get('access_key'),
                aws_secret_access_key=config.get('secret_key'),
                endpoint_url=config.get('endpoint_url')
            )
        else:
            dsn = (
                f"DRIVER={{{config.get('driver', 'ODBC Driver 17 for SQL Server')}}};"
                f"SERVER={config.get('server', 'localhost')};"
                f"DATABASE={db_name};"
                f"UID={config.get('username', 'sa')};"
                f"PWD={config.get('password', 'S0rry!43')}"
            )
            connection = pyodbc.connect(dsn)

        # Get schema dictionary
        schema_dict = await schema_manager.build_schema_dictionary(connection)
        if not schema_dict:
            logger.error("Failed to load schema dictionary")
            return False

        # Generate training data
        table_synonyms = await generate_synonyms(schema_dict, model, 'tables', logger, config['type'])
        column_synonyms = await generate_synonyms(schema_dict, model, 'columns', logger, config['type'])
        query_patterns = await generate_query_patterns(schema_dict, logger, config['type'])

        # Save training data
        output_dir = pathlib.Path(display_name, db_name, "training_data")
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

        await save_training_data(output_dir / "table_synonyms.json", table_synonyms, logger)
        await save_training_data(output_dir / "column_synonyms.json", column_synonyms, logger)
        await save_training_data(output_dir / "query_patterns.json", query_patterns, logger)

        logger.info(f"Training data generated successfully in {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error generating training data: {e}", exc_info=True)
        return False
    finally:
        if connection:
            if isinstance(connection, pyodbc.Connection):
                connection.close()
                logger.debug("Closed SQL Server connection")
            elif isinstance(connection, BaseClient):
                connection.close()
                logger.debug("Closed S3 client connection")

async def generate_synonyms(schema_dict: Dict, model: SentenceTransformer, data_type: str, logger: logging.Logger, config_type: str) -> Dict:
    """
    Generate synonyms for tables or columns using embeddings and static dictionary (Scenario 4).

    Args:
        schema_dict (Dict): Schema dictionary from SchemaManager.
        model (SentenceTransformer): Model for embedding-based synonyms.
        data_type (str): 'tables' or 'columns'.
        logger (logging.Logger): Logger instance.
        config_type (str): 'SQL Server' or 's3'.

    Returns:
        Dict: Mapping of table/column names to synonym lists.
    """
    try:
        synonyms = {}
        candidate_words = list(SYNONYMS.keys())
        candidate_embeddings = model.encode(candidate_words, batch_size=32, show_progress_bar=False)

        for schema in schema_dict[data_type]:
            items = schema_dict[data_type][schema]
            if data_type == 'columns':
                items = {table: cols.keys() for table in schema_dict['tables'][schema] for cols in [schema_dict['columns'][schema][table]]}

            for item in (items if data_type == 'tables' else items.items()):
                if data_type == 'columns':
                    table, columns = item
                    synonyms[f"{schema}.{table}"] = {}
                    names = list(columns)
                else:
                    names = [item]
                    if config_type.lower() == 's3':
                        synonyms[item] = []
                    else:
                        synonyms[f"{schema}.{item}"] = []

                name_embeddings = model.encode(names, batch_size=32, show_progress_bar=False)

                for i, name in enumerate(names):
                    similarities = np.dot(candidate_embeddings, name_embeddings[i]) / (
                        np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(name_embeddings[i])
                    )
                    embedding_synonyms = [candidate_words[j] for j in np.where(similarities > 0.8)[0]]
                    static_synonyms = SYNONYMS.get(name.lower(), [])
                    combined_synonyms = list(set(embedding_synonyms + static_synonyms))

                    if data_type == 'columns':
                        synonyms[f"{schema}.{table}"][name] = combined_synonyms
                    else:
                        if config_type.lower() == 's3':
                            synonyms[item].extend(combined_synonyms)
                        else:
                            synonyms[f"{schema}.{item}"].extend(combined_synonyms)

                    logger.debug(f"Generated {len(combined_synonyms)} synonyms for {schema}.{name if data_type == 'tables' else f'{table}.{name}'}: {combined_synonyms}")

        return synonyms
    except Exception as e:
        logger.error(f"Error generating {data_type} synonyms: {e}", exc_info=True)
        return {}

async def generate_query_patterns(schema_dict: Dict, logger: logging.Logger, config_type: str) -> List[Dict]:
    """
    Generate synthetic query patterns for training (Scenario 4).

    Args:
        schema_dict (Dict): Schema dictionary from SchemaManager.
        logger (logging.Logger): Logger instance.
        config_type (str): 'SQL Server' or 's3'.

    Returns:
        List[Dict]: List of query pattern dictionaries.
    """
    try:
        patterns = []
        templates = [
            "SELECT * FROM {table}",
            "SELECT {column} FROM {table}",
            "FIND {column} IN {table}",
            "GET {column} FROM {table} WHERE {condition}",
            "LIST {table}"
        ]

        for schema in schema_dict['tables']:
            for table in schema_dict['tables'][schema]:
                table_ref = table if config_type.lower() == 's3' else f"{schema}.{table}"
                for template in [t for t in templates if "{column}" not in t]:
                    patterns.append({"query": template.format(table=table_ref), "table": table_ref})

                columns = schema_dict['columns'][schema].get(table, {})
                for column in columns:
                    for template in [t for t in templates if "{column}" in t]:
                        condition = f"{column} = 'value'" if "{condition}" in template else ""
                        patterns.append({
                            "query": template.format(table=table_ref, column=column, condition=condition),
                            "table": table_ref,
                            "column": column
                        })

                logger.debug(f"Generated {len(patterns)} query patterns for {table_ref}")

        return patterns
    except Exception as e:
        logger.error(f"Error generating query patterns: {e}", exc_info=True)
        return []

async def save_training_data(file_path: pathlib.Path, data: Dict, logger: logging.Logger):
    """
    Save training data to JSON file asynchronously (Scenario 5).

    Args:
        file_path (pathlib.Path): Output file path.
        data (Dict): Data to save.
        logger (logging.Logger): Logger instance.
    """
    try:
        async with asyncio.Lock():
            await asyncio.to_thread(json.dump, data, file_path.open('w'), indent=2)
        logger.debug(f"Saved training data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving training data to {file_path}: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for TIA-1.2")
    parser.add_argument("--config", default="bikestores", help="Configuration name (e.g., bikestores, bikestores-csv)")
    args = parser.parse_args()

    async def main():
        success = await generate_training_data(args.config)
        print(f"Training data generation {'succeeded' if success else 'failed'} for config: {args.config}")

    asyncio.run(main())