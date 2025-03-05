"""
Main script for the paper analyzer.
Analyzes papers with Claude and OpenAI and stores the results in the database.
"""
import os
import sys
import logging
import asyncio
import argparse
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.analyzer.orchestrator import AnalyzerOrchestrator
from src.database.db import Database
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("data/logs/analyzer.log"))
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def analyze_papers(
    paper_ids: Optional[List[str]] = None,
    limit: int = 10,
    max_workers: int = 5
) -> None:
    """
    Analyze papers with Claude and OpenAI.
    
    Args:
        paper_ids: Optional list of paper IDs to analyze.
                  If None, gets unprocessed papers from the database.
        limit: Maximum number of papers to process if paper_ids is None
        max_workers: Maximum number of concurrent workers
    """
    logger.info(f"Starting paper analysis with limit: {limit}, max_workers: {max_workers}")
    
    # Initialize the database
    db = Database()
    
    # Initialize the orchestrator
    orchestrator = AnalyzerOrchestrator(db=db, max_workers=max_workers)
    
    try:
        # Analyze papers
        results = await orchestrator.analyze_papers(paper_ids=paper_ids, limit=limit)
        
        # Log the results
        if results:
            logger.info(f"Successfully analyzed {len(results)} papers")
            
            # Log the top papers
            sorted_results = sorted(results, key=lambda x: x["combined_score"], reverse=True)
            
            logger.info("Top paper opportunities:")
            for i, result in enumerate(sorted_results[:5]):
                paper_id = result["paper_id"]
                score = result["combined_score"]
                paper = db.get_paper(paper_id)
                title = paper["title"] if paper else "Unknown"
                
                logger.info(f"{i+1}. {title} (ID: {paper_id}, Score: {score:.2f})")
        else:
            logger.info("No papers were analyzed")
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Analyze papers with Claude and OpenAI")
    
    parser.add_argument(
        "--paper-ids",
        type=str,
        help="Comma-separated list of paper IDs to analyze"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=config.get_int("BATCH_SIZE", 10),
        help=f"Maximum number of papers to process (default: {config.get_int('BATCH_SIZE', 10)})"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=config.get_int("MAX_WORKERS", 5),
        help=f"Maximum number of concurrent workers (default: {config.get_int('MAX_WORKERS', 5)})"
    )
    
    return parser.parse_args()


async def main():
    """
    Main entry point.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Parse paper IDs if provided
    paper_ids = None
    if args.paper_ids:
        paper_ids = [pid.strip() for pid in args.paper_ids.split(",")]
    
    # Create log directory
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze papers
    await analyze_papers(
        paper_ids=paper_ids,
        limit=args.limit,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    asyncio.run(main()) 