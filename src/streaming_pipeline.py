"""
Streaming pipeline that processes papers one by one as they are scraped.
More efficient than the batch pipeline as it immediately processes each paper.
"""
import os
import sys
import logging
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.scraper.arxiv_client import ArxivClient
from src.analyzer.orchestrator import AnalyzerOrchestrator
from src.database.db import Database
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("data/logs/streaming_pipeline.log"))
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class StreamingPipeline:
    """
    Streaming pipeline that processes papers one by one as they are scraped.
    Advantages:
    - Immediate processing without waiting for all papers to be scraped
    - Better error isolation (one paper failure doesn't affect others)
    - Lower memory usage as papers are processed immediately
    - Natural deduplication as papers are checked against DB before processing
    """
    
    def __init__(
        self,
        db: Optional[Database] = None,
        arxiv_client: Optional[ArxivClient] = None,
        analyzer: Optional[AnalyzerOrchestrator] = None,
        max_workers: int = 5
    ):
        """
        Initialize the streaming pipeline.
        
        Args:
            db: Database instance
            arxiv_client: ArxivClient instance
            analyzer: AnalyzerOrchestrator instance
            max_workers: Maximum number of concurrent workers
        """
        # Initialize the database
        self.db = db or Database()
        
        # Initialize the arXiv client
        self.arxiv_client = arxiv_client or ArxivClient()
        
        # Initialize the analyzer orchestrator with OpenAI only mode
        # This skips Claude analysis which might have rate limiting issues
        self.analyzer = analyzer or AnalyzerOrchestrator(db=self.db, max_workers=max_workers, openai_only=True)
        
        # Maximum number of concurrent workers
        self.max_workers = max_workers
        
        # Semaphore for limiting concurrent processing
        self.semaphore = asyncio.Semaphore(max_workers)
        
        logger.info(f"StreamingPipeline initialized with max_workers: {max_workers}")
    
    async def run(
        self,
        categories: List[str],
        max_papers: int = 100,
        min_score_threshold: float = 0.7,
        retention_days: Optional[int] = None,
        force_reanalysis: bool = False
    ) -> Dict[str, Any]:
        """
        Run the streaming pipeline.
        
        Args:
            categories: List of arXiv categories to process
            max_papers: Maximum number of papers to process
            min_score_threshold: Minimum score for an opportunity to be considered "top"
            retention_days: If specified, removes processed papers older than this many days
            force_reanalysis: If True, re-analyzes papers that have already been processed
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(
            f"Starting streaming pipeline with categories: {categories}, max_papers: {max_papers}"
        )
        
        stats = {
            "total_papers_found": 0,
            "new_papers": 0,
            "papers_processed": 0,
            "papers_failed": 0,
            "top_opportunities": 0,
            "duplicates_skipped": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None
        }
        
        try:
            # Optionally clean up old papers
            if retention_days is not None:
                await self._cleanup_old_papers(retention_days)
            
            # Search for papers
            papers = await self.arxiv_client.search_papers(
                categories=categories,
                max_results=max_papers,
                sort_by="submittedDate",
                sort_order="descending"
            )
            
            stats["total_papers_found"] = len(papers)
            
            if not papers:
                logger.warning("No papers found.")
                return stats
            
            logger.info(f"Found {len(papers)} papers")
            
            # Get existing paper titles to check for duplicates based on content
            existing_titles = {p["title"].lower(): p["paper_id"] for p in self.db.get_all_papers()}
            
            # Process papers in parallel with limited concurrency
            tasks = []
            for paper in papers:
                # Check if the paper already exists and is processed
                existing_paper = self.db.get_paper(paper["paper_id"])
                if existing_paper and existing_paper.get("processed", 0) == 1 and not force_reanalysis:
                    logger.debug(f"Paper {paper['paper_id']} already processed, skipping")
                    continue
                
                # Check for title similarity to detect duplicates even with version differences
                # This catches papers that might have been updated or have slightly different IDs
                paper_title = paper["title"].lower()
                is_duplicate = False
                
                # Skip exact title matches (different versions of same paper)
                if paper_title in existing_titles and existing_titles[paper_title] != paper["paper_id"]:
                    logger.info(f"Skipping paper with duplicate title: {paper['title']} (ID: {paper['paper_id']})")
                    stats["duplicates_skipped"] += 1
                    continue
                
                # Check for high similarity with existing titles
                # This is a simple check for very similar titles (e.g. with minor changes)
                for existing_title in existing_titles:
                    # Calculate Jaccard similarity between titles (as sets of words)
                    title_words = set(paper_title.split())
                    existing_words = set(existing_title.split())
                    
                    if len(title_words) > 0 and len(existing_words) > 0:
                        intersection = len(title_words.intersection(existing_words))
                        union = len(title_words.union(existing_words))
                        similarity = intersection / union
                        
                        # If titles are very similar (>85% word overlap), consider it a duplicate
                        if similarity > 0.85:
                            logger.info(f"Skipping paper with similar title ({similarity:.2f}): {paper['title']} (ID: {paper['paper_id']})")
                            stats["duplicates_skipped"] += 1
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    continue
                
                # If paper exists but we're forcing reanalysis, mark it as new
                if existing_paper and force_reanalysis:
                    logger.info(f"Re-analyzing paper {paper['paper_id']}")
                    # Keep the PDF path if it exists
                    if "pdf_path" in existing_paper and existing_paper["pdf_path"]:
                        paper["pdf_path"] = existing_paper["pdf_path"]
                
                # Paper is new or needs reanalysis
                stats["new_papers"] += 1
                task = asyncio.create_task(self._process_paper(paper, force_reanalysis))
                tasks.append(task)
            
            # Wait for all tasks to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count successes and failures
                for result in results:
                    if isinstance(result, Exception):
                        stats["papers_failed"] += 1
                    elif result:
                        stats["papers_processed"] += 1
                        if result.get("is_top_opportunity", False):
                            stats["top_opportunities"] += 1
            
            # Update stats
            stats["end_time"] = datetime.now().isoformat()
            
            # Log results
            self._log_results(stats)
            
            return stats
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            stats["end_time"] = datetime.now().isoformat()
            return stats
    
    async def _process_paper(self, paper: Dict[str, Any], force_reanalysis: bool = False) -> Optional[Dict[str, Any]]:
        """
        Process a single paper from scraping to analysis.
        
        Args:
            paper: Paper metadata dictionary
            force_reanalysis: If True, re-analyzes papers that have already been processed
            
        Returns:
            Dictionary with processing results, or None on failure
        """
        paper_id = paper["paper_id"]
        logger.info(f"Processing paper: {paper_id} {'(reanalysis)' if force_reanalysis else ''}")
        
        # Use semaphore to limit concurrent processing
        async with self.semaphore:
            try:
                # Step 1: Add paper to database
                self.db.add_paper(paper)
                
                # Step 2: Download PDF if needed
                if "pdf_path" not in paper or not paper["pdf_path"]:
                    try:
                        pdf_path = await self.arxiv_client.download_pdf(paper_id)
                        paper["pdf_path"] = pdf_path
                        
                        # Update paper in database with PDF path
                        self.db.add_paper(paper)
                        logger.debug(f"Downloaded PDF for {paper_id} to {pdf_path}")
                    except Exception as e:
                        logger.error(f"Failed to download PDF for {paper_id}: {e}")
                        return None
                
                # Step 3: Analyze the paper
                results = await self.analyzer.analyze_papers(paper_ids=[paper_id])
                
                if not results:
                    logger.warning(f"No analysis results for paper {paper_id}")
                    return None
                
                analysis = results[0]
                
                # Determine if this is a top opportunity based on score
                is_top = analysis.get("combined_score", 0) >= 0.7
                
                return {
                    "paper_id": paper_id,
                    "combined_score": analysis.get("combined_score", 0),
                    "is_top_opportunity": is_top
                }
                
            except Exception as e:
                logger.error(f"Failed to process paper {paper_id}: {e}", exc_info=True)
                return None
    
    async def _cleanup_old_papers(self, days: int) -> None:
        """
        Clean up processed papers older than the specified number of days.
        
        Args:
            days: Number of days to keep papers
        """
        logger.info(f"Cleaning up processed papers older than {days} days")
        
        try:
            # Calculate cutoff date
            from datetime import timedelta
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Get papers older than cutoff date
            conn = self.db._get_connection()
            cursor = conn.execute(
                """
                SELECT paper_id, pdf_path FROM papers 
                WHERE processed = 1 AND scrape_date < ? 
                """,
                (cutoff_date,)
            )
            old_papers = cursor.fetchall()
            
            if not old_papers:
                logger.info("No old papers to clean up")
                conn.close()
                return
                
            logger.info(f"Found {len(old_papers)} papers older than {days} days")
            
            # Delete PDFs
            deleted_pdfs = 0
            for paper in old_papers:
                if paper["pdf_path"]:
                    pdf_path = Path(paper["pdf_path"])
                    if pdf_path.exists():
                        pdf_path.unlink()  # Delete the file
                        deleted_pdfs += 1
            
            # Delete papers from database
            cursor = conn.execute(
                """
                DELETE FROM papers
                WHERE processed = 1 AND scrape_date < ?
                """,
                (cutoff_date,)
            )
            deleted_papers = cursor.rowcount
            
            # Delete analyses
            paper_ids = [paper["paper_id"] for paper in old_papers]
            if paper_ids:
                placeholders = ",".join(["?" for _ in paper_ids])
                cursor = conn.execute(
                    f"""
                    DELETE FROM analyses
                    WHERE paper_id IN ({placeholders})
                    """,
                    paper_ids
                )
                deleted_analyses = cursor.rowcount
            else:
                deleted_analyses = 0
                
            conn.commit()
            conn.close()
            
            logger.info(f"Cleanup completed: Deleted {deleted_pdfs} PDFs, {deleted_papers} paper records, and {deleted_analyses} analyses")
        except Exception as e:
            logger.error(f"Failed to clean up old papers: {e}", exc_info=True)
    
    def _log_results(self, stats: Dict[str, Any]) -> None:
        """
        Log the results of the pipeline.
        
        Args:
            stats: Statistics dictionary
        """
        logger.info(f"Pipeline results:")
        logger.info(f"Total papers found: {stats['total_papers_found']}")
        logger.info(f"New/unprocessed papers: {stats['new_papers']}")
        logger.info(f"Papers successfully processed: {stats['papers_processed']}")
        logger.info(f"Papers failed: {stats['papers_failed']}")
        logger.info(f"Top opportunities: {stats['top_opportunities']}")
        logger.info(f"Duplicates skipped: {stats.get('duplicates_skipped', 0)}")
        
        # Get opportunity counts from database for verification
        total_papers, processed_papers, top_opportunities = self.db.get_opportunity_count()
        logger.info(f"Database stats - Total: {total_papers}, Processed: {processed_papers}, Top: {top_opportunities}")


async def main():
    """
    Main entry point for the streaming pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the streaming pipeline")
    parser.add_argument(
        "--categories", 
        nargs="+", 
        default=["cs.AI", "cs.LG", "cs.CV"], 
        help="arXiv categories to process"
    )
    parser.add_argument(
        "--max-papers", 
        type=int, 
        default=100, 
        help="Maximum number of papers to process"
    )
    parser.add_argument(
        "--min-score", 
        type=float, 
        default=0.7, 
        help="Minimum score for top opportunities"
    )
    parser.add_argument(
        "--retention-days", 
        type=int, 
        help="Delete processed papers older than this many days"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=5, 
        help="Maximum number of concurrent workers"
    )
    parser.add_argument(
        "--force-reanalysis",
        action="store_true",
        help="Force re-analysis of already processed papers"
    )
    
    args = parser.parse_args()
    
    # Initialize and run the pipeline
    pipeline = StreamingPipeline(max_workers=args.max_workers)
    await pipeline.run(
        categories=args.categories,
        max_papers=args.max_papers,
        min_score_threshold=args.min_score,
        retention_days=args.retention_days,
        force_reanalysis=args.force_reanalysis
    )


if __name__ == "__main__":
    asyncio.run(main()) 