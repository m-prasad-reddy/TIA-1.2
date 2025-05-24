"""
Profiles performance of cache operations for weights and name matches in TIA-1.2.
Improves TIA-1.1 and TIA-1.2 versions with async profiling, S3 support, and structured output.
Uses DISPLAY_NAME-based paths and logging for consistency.
Addresses Scenarios 1 (S3 support), 5 (performance optimization), and 8 (weight profiling).
"""
import cProfile
import pstats
import logging
import asyncio
import pathlib
import argparse
from typing import Optional
from config.cache_synchronizer import CacheSynchronizer
from config.logging_setup import setup_logging

async def profile_operations(
    display_name: str,
    db_name: str,
    output_dir: Optional[str] = None
) -> bool:
    """
    Profile cache write operations for weights and name matches asynchronously (Scenarios 1, 5, 8).

    Args:
        display_name (str): Display name (e.g., BIKE-STORES).
        db_name (str): Database name (e.g., BikeStores).
        output_dir (Optional[str]): Directory for profile output files (default: DISPLAY_NAME/db-name/profiles/).

    Returns:
        bool: True if profiling succeeds, False otherwise.
    """
    setup_logging(display_name, db_name)
    logger = logging.getLogger("profile_weights")
    logger.info(f"Starting profile_operations for {display_name}/{db_name}")

    try:
        # Set up output directory
        profile_dir = pathlib.Path(output_dir or f"{display_name}/{db_name}/profiles")
        await asyncio.to_thread(profile_dir.mkdir, parents=True, exist_ok=True)

        # Initialize CacheSynchronizer
        cache_synchronizer = CacheSynchronizer(display_name, db_name)

        # Sample data including S3 table
        weights = {
            "sales.orders": {
                "order_id": 0.1,
                "order_date": 0.05
            },
            "s3.stores": {
                "store_id": 0.15,
                "store_name": 0.08
            }
        }
        name_matches = {
            "order_id": ["order id", "orderid"],
            "order_date": ["order date", "date"],
            "store_id": ["store id", "storeid"],
            "store_name": ["store name", "name"]
        }

        # Profile write_weights with timeout
        logger.debug("Profiling write_weights")
        write_weights_file = profile_dir / "write_weights_profile.out"
        async with asyncio.timeout(120):  # 2-minute timeout
            await asyncio.to_thread(
                cProfile.runctx,
                "cache_synchronizer.write_weights(weights, batch_size=10)",
                globals(),
                locals(),
                str(write_weights_file)
            )

        # Profile write_name_matches with timeout
        logger.debug("Profiling write_name_matches")
        write_name_matches_file = profile_dir / "write_name_matches_profile.out"
        async with asyncio.timeout(120):  # 2-minute timeout
            await asyncio.to_thread(
                cProfile.runctx,
                "cache_synchronizer.write_name_matches(name_matches, batch_size=50)",
                globals(),
                locals(),
                str(write_name_matches_file)
            )

        # Print profiling results
        logger.info("write_weights profile:")
        p = pstats.Stats(str(write_weights_file))
        p.sort_stats("cumulative").print_stats(10)

        logger.info("write_name_matches profile:")
        p = pstats.Stats(str(write_name_matches_file))
        p.sort_stats("cumulative").print_stats(10)

        logger.info(f"Completed profile_operations, profiles saved to {profile_dir}")
        return True
    except TimeoutError:
        logger.error("Profiling operation timed out after 120 seconds")
        return False
    except Exception as e:
        logger.error(f"Error during profiling: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    async def main():
        parser = argparse.ArgumentParser(description="Profile TIA-1.2 cache operations")
        parser.add_argument("--display-name", default="BIKE-STORES", help="Display name (e.g., BIKE-STORES)")
        parser.add_argument("--db-name", default="BikeStores", help="Database name (e.g., BikeStores)")
        parser.add_argument("--output-dir", help="Directory for profile output files")
        args = parser.parse_args()

        success = await profile_operations(args.display_name, args.db_name, args.output_dir)
        print(f"Profiling {'succeeded' if success else 'failed'}")

    asyncio.run(main())