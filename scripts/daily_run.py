#!/usr/bin/env python
"""
Script to run the arXiv opportunity finder pipeline daily.
Designed to be run as a cron job.
"""
import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline import run_pipeline
from src.config import config

# Configure logging
log_dir = Path(project_root) / "data" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"daily_run_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """
    Main entry point.
    """
    logger.info("Starting daily run of arXiv opportunity finder")
    
    # Default categories for daily run
    categories = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE"]
    
    # Limits for daily run
    scrape_limit = config.get_int("ARXIV_MAX_RESULTS", 50)
    analyze_limit = config.get_int("BATCH_SIZE", 5)
    max_workers = config.get_int("MAX_WORKERS", 5)
    
    try:
        # Run the pipeline
        await run_pipeline(
            categories=categories,
            scrape_limit=scrape_limit,
            analyze_limit=analyze_limit,
            max_workers=max_workers,
            download_pdfs=True
        )
        
        logger.info("Daily run completed successfully")
    
    except Exception as e:
        logger.error(f"Daily run failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 