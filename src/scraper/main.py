"""
Main script for the arXiv scraper.
Combines the ArxivClient and PaperQueue to scrape papers from arXiv and prepare them for processing.
"""
import os
import sys
import logging
import asyncio
import argparse
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.scraper.arxiv_client import ArxivClient
from src.scraper.paper_queue import PaperQueue


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("data/logs/scraper.log"))
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def scrape_papers(
    categories: List[str],
    limit: int,
    download_pdfs: bool = True,
    wait_time: int = 3
) -> None:
    """
    Scrape papers from arXiv and add them to the queue.
    
    Args:
        categories: List of categories to scrape
        limit: Maximum number of papers to scrape
        download_pdfs: Whether to download PDFs
        wait_time: Time to wait between API calls
    """
    logger.info(f"Starting paper scraping with categories: {categories}, limit: {limit}")
    
    # Initialize the arXiv client
    papers_dir = os.getenv("PAPERS_DIR", "data/pdfs")
    client = ArxivClient(papers_dir=papers_dir, wait_time=wait_time)
    
    # Initialize the paper queue
    queue = PaperQueue()
    
    # Search for papers
    papers = await client.search_papers(
        categories=categories,
        max_results=limit,
        sort_by="submittedDate",
        sort_order="descending"
    )
    
    if not papers:
        logger.warning("No papers found.")
        return
    
    logger.info(f"Found {len(papers)} papers")
    
    # Add papers to the queue
    added_count = await queue.add_papers(papers)
    logger.info(f"Added {added_count} new papers to the queue")
    
    # Download PDFs if requested
    if download_pdfs and added_count > 0:
        logger.info("Downloading PDFs...")
        
        # Get the papers that were just added
        batch = await queue.get_next_batch(added_count)
        
        # Download PDFs for each paper
        for i, paper in enumerate(batch):
            paper_id = paper["paper_id"]
            logger.info(f"Downloading PDF {i+1}/{len(batch)}: {paper_id}")
            
            try:
                pdf_path = await client.download_pdf(paper_id)
                logger.info(f"Downloaded PDF for {paper_id} to {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to download PDF for {paper_id}: {e}")
    
    # Log queue status
    queue_size = await queue.get_queue_size()
    processed_count = await queue.get_processed_count()
    logger.info(f"Current queue status: {queue_size} papers in queue, {processed_count} papers processed")


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Scrape papers from arXiv")
    
    parser.add_argument(
        "--categories",
        type=str,
        default="cs.AI,cs.LG",
        help="Comma-separated list of categories to scrape (default: cs.AI,cs.LG)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of papers to scrape (default: 100)"
    )
    
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download PDFs"
    )
    
    parser.add_argument(
        "--wait-time",
        type=int,
        default=3,
        help="Time to wait between API calls (default: 3 seconds)"
    )
    
    return parser.parse_args()


async def main():
    """
    Main entry point.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Parse categories
    categories = args.categories.split(",")
    
    # Create log directory
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Scrape papers
        await scrape_papers(
            categories=categories,
            limit=args.limit,
            download_pdfs=not args.no_download,
            wait_time=args.wait_time
        )
    except Exception as e:
        logger.error(f"Error during scraping: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 