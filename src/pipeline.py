"""
Main pipeline script that ties together the scraper and analyzer.
Orchestrates the full process of scraping, analyzing, and storing results.
"""
import os
import sys
import logging
import asyncio
import argparse
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.scraper.arxiv_client import ArxivClient
from src.scraper.paper_queue import PaperQueue
from src.analyzer.orchestrator import AnalyzerOrchestrator
from src.database.db import Database
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("data/logs/pipeline.log"))
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class Pipeline:
    """
    Main pipeline for orchestrating the scraping and analysis of papers.
    """
    
    def __init__(
        self,
        db: Optional[Database] = None,
        arxiv_client: Optional[ArxivClient] = None,
        paper_queue: Optional[PaperQueue] = None,
        analyzer: Optional[AnalyzerOrchestrator] = None,
        max_workers: int = 5
    ):
        """
        Initialize the pipeline.
        
        Args:
            db: Database instance
            arxiv_client: ArxivClient instance
            paper_queue: PaperQueue instance
            analyzer: AnalyzerOrchestrator instance
            max_workers: Maximum number of concurrent workers
        """
        # Initialize the database
        self.db = db or Database()
        
        # Initialize the arXiv client
        self.arxiv_client = arxiv_client or ArxivClient()
        
        # Initialize the paper queue
        self.paper_queue = paper_queue or PaperQueue()
        
        # Initialize the analyzer orchestrator
        # Use OpenAI-only mode to ensure layman's explanations are generated
        self.analyzer = analyzer or AnalyzerOrchestrator(db=self.db, max_workers=max_workers, openai_only=True)
        
        # Maximum number of concurrent workers
        self.max_workers = max_workers
        
        # Statistics
        self.stats = {
            "arxiv_papers_found": 0,
            "papers_queued": 0,
            "papers_failed": 0,
            "papers_duplicates": 0,
            "papers_analyzed": 0,
            "papers_processed": 0,
            "analysis_failed": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None
        }
        
        logger.info(f"Pipeline initialized with max_workers: {max_workers}")
    
    async def run(
        self,
        categories: List[str],
        scrape_limit: int = 100,
        analyze_limit: int = 10,
        download_pdfs: bool = True
    ) -> None:
        """
        Run the full pipeline.
        
        Args:
            categories: List of arXiv categories to scrape
            scrape_limit: Maximum number of papers to scrape
            analyze_limit: Maximum number of papers to analyze
            download_pdfs: Whether to download PDFs during scraping
        """
        logger.info(
            f"Starting pipeline with categories: {categories}, "
            f"scrape_limit: {scrape_limit}, analyze_limit: {analyze_limit}"
        )
        
        try:
            # Step 1: Scrape papers from arXiv
            await self._scrape_papers(categories, scrape_limit, download_pdfs)
            
            # Step 2: Store papers in the database
            await self._store_papers()
            
            # Step 3: Analyze papers
            await self._analyze_papers(analyze_limit)
            
            # Step 4: Log results
            self._log_results()
            
            logger.info("Pipeline completed successfully")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    async def _scrape_papers(self, categories: List[str], limit: int, download_pdfs: bool) -> None:
        """
        Scrape papers from arXiv.
        
        Args:
            categories: List of categories to scrape
            limit: Maximum number of papers to scrape
            download_pdfs: Whether to download PDFs
        """
        logger.info(f"Scraping papers from arXiv: categories={categories}, limit={limit}")
        
        try:
            # Search for papers
            papers = await self.arxiv_client.search_papers(
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
            added_count = await self.paper_queue.add_papers(papers)
            logger.info(f"Added {added_count} new papers to the queue")
            
            # Download PDFs if requested
            if download_pdfs and added_count > 0:
                logger.info("Downloading PDFs...")
                
                # Get the papers that were just added
                batch = await self.paper_queue.get_next_batch(added_count)
                
                # Download PDFs for each paper
                for i, paper in enumerate(batch):
                    paper_id = paper["paper_id"]
                    logger.info(f"Downloading PDF {i+1}/{len(batch)}: {paper_id}")
                    
                    try:
                        pdf_path = await self.arxiv_client.download_pdf(paper_id)
                        paper["pdf_path"] = pdf_path
                        logger.info(f"Downloaded PDF for {paper_id} to {pdf_path}")
                    except Exception as e:
                        logger.error(f"Failed to download PDF for {paper_id}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to scrape papers: {e}", exc_info=True)
            raise
    
    async def _store_papers(self) -> None:
        """
        Store papers from the queue in the database.
        """
        logger.info("Storing papers in the database")
        
        try:
            # Get all papers from the queue
            queue_size = await self.paper_queue.get_queue_size()
            papers = await self.paper_queue.get_next_batch(queue_size)
            
            if not papers:
                logger.info("No papers to store")
                return
            
            # Add papers to the database
            self.db.add_papers(papers)
            
            # Mark papers as processed in the queue
            paper_ids = [paper["paper_id"] for paper in papers]
            await self.paper_queue.mark_processed(paper_ids)
            
            logger.info(f"Stored {len(papers)} papers in the database")
        
        except Exception as e:
            logger.error(f"Failed to store papers: {e}", exc_info=True)
            raise
    
    async def _analyze_papers(self, limit: int) -> None:
        """
        Analyze papers from the database.
        
        Args:
            limit: Maximum number of papers to analyze
        """
        logger.info(f"Analyzing papers: limit={limit}")
        
        try:
            # Get unprocessed papers from the database
            papers = self.db.get_unprocessed_papers(limit=limit)
            
            if not papers:
                logger.info("No papers to analyze")
                return
            
            logger.info(f"Analyzing {len(papers)} papers")
            
            # Get the paper IDs
            paper_ids = [paper["paper_id"] for paper in papers]
            
            # Analyze the papers
            results = await self.analyzer.analyze_papers(paper_ids=paper_ids)
            
            logger.info(f"Analyzed {len(results)} papers")
        
        except Exception as e:
            logger.error(f"Failed to analyze papers: {e}", exc_info=True)
            raise
    
    def _log_results(self) -> None:
        """
        Log the results of the pipeline.
        """
        try:
            # Get opportunity counts
            total_papers, processed_papers, top_opportunities = self.db.get_opportunity_count()
            
            logger.info(f"Pipeline results:")
            logger.info(f"Total papers: {total_papers}")
            logger.info(f"Processed papers: {processed_papers}")
            logger.info(f"Top opportunities: {top_opportunities}")
            
            # Get top opportunities
            opportunities = self.db.get_top_opportunities(limit=5)
            
            if opportunities:
                logger.info("Top paper opportunities:")
                for i, opp in enumerate(opportunities):
                    logger.info(f"{i+1}. {opp['title']} (ID: {opp['paper_id']}, Score: {opp['combined_score']:.2f})")
        
        except Exception as e:
            logger.error(f"Failed to log results: {e}", exc_info=True)


async def run_pipeline(
    categories: Optional[List[str]] = None,
    scrape_limit: int = 100,
    analyze_limit: int = 10,
    max_workers: int = 5,
    download_pdfs: bool = True,
    openai_only: bool = True
) -> None:
    """
    Run the pipeline end-to-end.
    
    Args:
        categories: List of arXiv categories to scrape. If None, uses default categories.
        scrape_limit: Maximum number of papers to scrape
        analyze_limit: Maximum number of papers to analyze
        max_workers: Maximum number of concurrent workers
        download_pdfs: Whether to download PDFs
        openai_only: Whether to use OpenAI only mode (skips Claude)
    """
    # Set up categories
    if categories is None:
        categories = config.DEFAULT_CATEGORIES
    
    logger.info(f"Running pipeline with categories: {categories}")
    logger.info(f"Limits: scrape={scrape_limit}, analyze={analyze_limit}, workers={max_workers}")
    logger.info(f"Using OpenAI-only mode: {openai_only}")
    
    # Initialize the database
    db = Database()
    
    # Ensure tables exist
    db.init_db()
    
    # Initialize the arXiv client
    arxiv_client = ArxivClient()
    
    # Initialize the paper queue
    paper_queue = PaperQueue()
    
    # Initialize the analyzer orchestrator
    analyzer = AnalyzerOrchestrator(db=db, max_workers=max_workers, openai_only=openai_only)
    
    # Create and run the pipeline
    pipeline = Pipeline(
        db=db,
        arxiv_client=arxiv_client,
        paper_queue=paper_queue,
        analyzer=analyzer,
        max_workers=max_workers
    )
    
    # Run the pipeline
    await pipeline.run(
        categories=categories,
        scrape_limit=scrape_limit,
        analyze_limit=analyze_limit,
        download_pdfs=download_pdfs
    )


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the ArXiv opportunity finder pipeline")
    
    parser.add_argument(
        "--categories", 
        nargs="+", 
        help="ArXiv categories to scrape (space-separated)"
    )
    
    parser.add_argument(
        "--scrape-limit", 
        type=int, 
        default=100, 
        help="Maximum number of papers to scrape"
    )
    
    parser.add_argument(
        "--analyze-limit", 
        type=int, 
        default=10, 
        help="Maximum number of papers to analyze"
    )
    
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=5,
        help="Maximum number of concurrent workers"
    )
    
    parser.add_argument(
        "--skip-pdf-download", 
        action="store_true",
        help="Skip downloading PDFs during scraping"
    )
    
    parser.add_argument(
        "--use-claude",
        action="store_true",
        help="Use Claude for analysis (default is OpenAI-only mode)"
    )
    
    return parser.parse_args()


async def main():
    """
    Main entry point for the pipeline.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up categories
    categories = args.categories
    
    # Set up limits
    scrape_limit = args.scrape_limit
    analyze_limit = args.analyze_limit
    max_workers = args.max_workers
    
    # Determine whether to download PDFs
    download_pdfs = not args.skip_pdf_download
    
    # Determine whether to use OpenAI-only mode
    openai_only = not args.use_claude
    
    # Run the pipeline
    await run_pipeline(
        categories=categories,
        scrape_limit=scrape_limit,
        analyze_limit=analyze_limit,
        max_workers=max_workers,
        download_pdfs=download_pdfs,
        openai_only=openai_only
    )


if __name__ == "__main__":
    asyncio.run(main()) 