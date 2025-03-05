#!/usr/bin/env python
"""
Script to run the arxiv-opportunity-finder using only OpenAI's o1 model.
"""
import os
import sys
import logging
import asyncio
import argparse
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.analyzer.openai_client import OpenAIClient
from src.scraper.arxiv_client import ArxivClient
from src.scraper.paper_queue import PaperQueue
from src.database.db import Database
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/logs/openai_run.log")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class OpenAIOnlyOrchestrator:
    """
    Modified orchestrator that only uses OpenAI for analysis.
    """
    
    def __init__(
        self,
        db: Optional[Database] = None,
        openai_client: Optional[OpenAIClient] = None,
        max_workers: int = 8
    ):
        """
        Initialize the OpenAI-only orchestrator.
        
        Args:
            db: Database instance
            openai_client: OpenAI client instance
            max_workers: Maximum number of concurrent workers
        """
        # Initialize the database
        self.db = db or Database()
        
        # Initialize the OpenAI client with o1 model
        self.openai_client = openai_client or OpenAIClient(model="o1-2024-12-17")
        
        # Maximum number of concurrent workers
        self.max_workers = max_workers
        
        # Semaphore for limiting concurrent API calls
        self.semaphore = asyncio.Semaphore(max_workers)
        
        # PDF directory for checking file existence
        self.papers_dir = os.getenv("PAPERS_DIR", "data/pdfs")
        
        logger.info(f"OpenAIOnlyOrchestrator initialized with max_workers: {max_workers} and model: {self.openai_client.model}")
        logger.info("Running with unrestricted OpenAI access - performance optimized for high throughput")
    
    async def analyze_papers(self, paper_ids: Optional[List[str]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Analyze a batch of papers using only OpenAI.
        
        Args:
            paper_ids: Optional list of paper IDs to analyze.
                      If None, gets unprocessed papers from the database.
            limit: Maximum number of papers to process if paper_ids is None
            
        Returns:
            List of analysis results
        """
        logger.info(f"Starting OpenAI-only analysis of papers: {paper_ids if paper_ids else f'(up to {limit} unprocessed papers)'}")
        
        # Get the papers to analyze
        if paper_ids:
            papers = [self.db.get_paper(paper_id) for paper_id in paper_ids]
            papers = [p for p in papers if p is not None]
        else:
            papers = self.db.get_unprocessed_papers(limit=limit)
        
        if not papers:
            logger.info("No papers to analyze")
            return []
        
        # Filter papers to only include those with PDFs
        papers_with_pdfs = []
        for paper in papers:
            paper_id = paper["paper_id"]
            pdf_path = Path(self.papers_dir) / f"{paper_id}.pdf"
            if pdf_path.exists():
                papers_with_pdfs.append(paper)
            else:
                logger.warning(f"Skipping paper {paper_id} - PDF not found at {pdf_path}")
        
        if not papers_with_pdfs:
            logger.warning(f"None of the {len(papers)} papers have PDFs. No analysis will be performed.")
            return []
            
        logger.info(f"Analyzing {len(papers_with_pdfs)} papers with OpenAI o1 (filtered from {len(papers)} total papers)")
        
        # Process the papers in parallel with limited concurrency
        results = []
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(paper):
            async with semaphore:
                try:
                    result = await self._process_paper(paper)
                    if result is not None:
                        return result
                    # If result is None (e.g., PDF not found), skip this paper
                    return None
                except Exception as e:
                    logger.error(f"Error processing paper {paper.get('paper_id')}: {e}")
                    return None
        
        tasks = [process_with_semaphore(paper) for paper in papers_with_pdfs]
        task_results = await asyncio.gather(*tasks)
        
        # Filter out None results and exceptions
        for result in task_results:
            if result is not None:
                results.append(result)
                
                # Log successful analysis
                paper_id = result.get("paper_id")
                score = result.get("combined_score", 0)
                title = result.get("title", "Unknown")
                logger.info(f"Successfully analyzed paper: {title} (ID: {paper_id}, Score: {score:.2f})")
        
        return results
    
    async def _process_paper(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single paper with OpenAI.
        
        Args:
            paper: Paper data
            
        Returns:
            Analysis results or None if processing failed
        """
        paper_id = paper["paper_id"]
        logger.info(f"Processing paper: {paper_id}")
        
        # Get the paper metadata
        paper_metadata = {
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "categories": paper.get("categories", ""),
            "abstract": paper.get("abstract", "")
        }
        
        # Get the PDF path
        papers_dir = os.getenv("PAPERS_DIR", "data/pdfs")
        pdf_path = str(Path(papers_dir) / f"{paper_id}.pdf")
        
        if not Path(pdf_path).exists():
            logger.warning(f"PDF not found for paper {paper_id}: {pdf_path}")
            return None
        
        # Analyze with OpenAI directly
        try:
            # Create a simple mock of the Claude analysis with just enough info for OpenAI
            mock_claude_analysis = {
                "innovation_score": 7,
                "poc_potential_score": 8,
                "wow_factor_score": 7,
                "implementation_complexity_score": 5,
                "raw_analysis": f"Paper Title: {paper_metadata['title']}\n\nAbstract: {paper_metadata['abstract']}"
            }
            
            # Plan implementation with OpenAI
            openai_analysis = await self.openai_client.plan_implementation(
                paper_metadata=paper_metadata,
                pdf_path=pdf_path,
                claude_analysis=mock_claude_analysis
            )
            
            # Combine results
            combined_results = self._combine_results(paper, mock_claude_analysis, openai_analysis)
            
            # Store the results
            self.db.add_analysis(combined_results)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error analyzing paper {paper_id}: {e}", exc_info=True)
            return None  # Return None on error
    
    def _combine_results(self, paper: Dict[str, Any], mock_claude_analysis: Dict[str, Any], openai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine paper data with analysis results.
        
        Args:
            paper: Paper data
            mock_claude_analysis: Placeholder for Claude analysis
            openai_analysis: OpenAI analysis results
            
        Returns:
            Combined results
        """
        # Extract scores from raw plan
        raw_plan = openai_analysis.get("raw_plan", "")
        
        # Initialize default scores
        innovation_score = 0
        technical_feasibility_score = 0
        market_potential_score = 0
        impact_score = 0
        opportunity_score = 0
        
        # Try to extract innovation score
        if "innovation" in raw_plan.lower() or "novel" in raw_plan.lower():
            innovation_matches = re.findall(r"innovation[^:]*:\s*(\d+)(?:/10)?|innovation[^:]*score[^:]*:\s*(\d+)(?:/10)?", raw_plan.lower())
            if innovation_matches:
                for match in innovation_matches:
                    for group in match:
                        if group and group.isdigit():
                            innovation_score = int(group)
                            break
        
        # Try to extract technical feasibility score
        if "technical feasibility" in raw_plan.lower() or "feasibility" in raw_plan.lower():
            feasibility_matches = re.findall(r"technical feasibility[^:]*:\s*(\d+)(?:/10)?|feasibility[^:]*score[^:]*:\s*(\d+)(?:/10)?", raw_plan.lower())
            if feasibility_matches:
                for match in feasibility_matches:
                    for group in match:
                        if group and group.isdigit():
                            technical_feasibility_score = int(group)
                            break
        
        # Try to extract market potential score
        if "market potential" in raw_plan.lower():
            market_matches = re.findall(r"market potential[^:]*:\s*(\d+)(?:/10)?|market[^:]*score[^:]*:\s*(\d+)(?:/10)?", raw_plan.lower())
            if market_matches:
                for match in market_matches:
                    for group in match:
                        if group and group.isdigit():
                            market_potential_score = int(group)
                            break
        
        # Try to extract impact score
        if "impact" in raw_plan.lower():
            impact_matches = re.findall(r"impact[^:]*:\s*(\d+)(?:/10)?|impact[^:]*score[^:]*:\s*(\d+)(?:/10)?", raw_plan.lower())
            if impact_matches:
                for match in impact_matches:
                    for group in match:
                        if group and group.isdigit():
                            impact_score = int(group)
                            break
        
        # Try to extract the overall opportunity score
        if "opportunity score" in raw_plan.lower():
            opportunity_matches = re.findall(r"opportunity score[^:]*:\s*(\d+)(?:/10)?", raw_plan.lower())
            if opportunity_matches:
                for match in opportunity_matches:
                    if match and match.isdigit():
                        opportunity_score = int(match)
                        break
        
        # If we couldn't extract scores, use some heuristics
        if innovation_score == 0:
            innovation_score = mock_claude_analysis.get("innovation_score", 7)
        
        if technical_feasibility_score == 0:
            # Use inverse of implementation complexity for feasibility
            complexity = mock_claude_analysis.get("implementation_complexity_score", 5)
            technical_feasibility_score = 11 - complexity  # Convert complexity to feasibility (10 - complexity + 1)
        
        if market_potential_score == 0:
            market_potential_score = mock_claude_analysis.get("poc_potential_score", 7)
        
        if impact_score == 0:
            impact_score = mock_claude_analysis.get("wow_factor_score", 6)
        
        # Calculate combined score if not extracted directly
        if opportunity_score == 0:
            # Scale scores from 0-10 to 0-1
            innovation = innovation_score / 10
            feasibility = technical_feasibility_score / 10
            market = market_potential_score / 10
            impact = impact_score / 10
            
            # Calculate weighted opportunity score
            opportunity_score = (innovation * 0.25 + feasibility * 0.3 + market * 0.3 + impact * 0.15) * 10
        
        # Extract time to market estimation
        time_to_market = "Unknown"
        time_patterns = [
            r"time[- ]to[- ]market[^:]*:\s*([^\.]+)",
            r"could be implemented in\s*([^\.]+)",
            r"estimated timeline[^:]*:\s*([^\.]+)"
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, raw_plan.lower())
            if matches:
                time_to_market = matches[0].strip()
                break
        
        # Extract target markets or industries
        target_markets = []
        market_section = re.search(r"target industries[^:]*:(.+?)(?:\n\n|\n[A-Z])", raw_plan, re.DOTALL | re.IGNORECASE)
        if market_section:
            market_text = market_section.group(1)
            # Extract bullet points or lines
            for line in market_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*')):
                    target_markets.append(line[1:].strip())
                elif line and not line.endswith(':'):
                    target_markets.append(line)
        
        if not target_markets:
            # Try to extract using AI/ML common markets
            common_markets = ["healthcare", "finance", "retail", "manufacturing", "education", 
                             "transportation", "logistics", "security", "energy", "entertainment"]
            for market in common_markets:
                if market in raw_plan.lower():
                    surrounding_text = re.search(r"[^.]*" + market + "[^.]*", raw_plan.lower())
                    if surrounding_text:
                        target_markets.append(surrounding_text.group(0).strip())
        
        # Include the extracted scores in the openai_analysis dictionary
        openai_analysis["innovation_score"] = innovation_score
        openai_analysis["technical_feasibility_score"] = technical_feasibility_score
        openai_analysis["market_potential_score"] = market_potential_score
        openai_analysis["impact_score"] = impact_score
        openai_analysis["opportunity_score"] = round(opportunity_score, 2)
        openai_analysis["time_to_market"] = time_to_market
        openai_analysis["target_markets"] = target_markets
        
        # Combine everything
        return {
            "paper_id": paper["paper_id"],
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "categories": paper.get("categories", ""),
            "abstract": paper.get("abstract", ""),
            "published": paper.get("published", ""),
            "updated": paper.get("updated", ""),
            "innovation_score": innovation_score,
            "poc_potential_score": market_potential_score,
            "wow_factor_score": impact_score,
            "implementation_complexity": mock_claude_analysis.get("implementation_complexity_score", 0),
            "technical_feasibility_score": technical_feasibility_score, 
            "market_potential_score": market_potential_score,
            "impact_score": impact_score,
            "time_to_market": time_to_market,
            "target_markets": target_markets,
            "combined_score": opportunity_score / 10,  # Scale to 0-1 for database compatibility
            "implementation_plan": openai_analysis.get("plan", ""),
            "steps": openai_analysis.get("steps", []),
            "resources": openai_analysis.get("resources", []),
            "challenges": openai_analysis.get("challenges", []),
            "time_estimate": openai_analysis.get("time_estimate", ""),
            "analyzed_at": datetime.now().isoformat(),
            "claude_analysis": mock_claude_analysis.get("raw_analysis", ""),
            "openai_analysis": openai_analysis.get("raw_plan", "")
        }


async def scrape_papers(
    categories: List[str],
    limit: int,
    download_pdfs: bool = True
) -> List[str]:
    """
    Scrape papers from arXiv and add them to the queue.
    
    Args:
        categories: List of categories to scrape
        limit: Maximum number of papers to scrape
        download_pdfs: Whether to download PDFs
        
    Returns:
        List of paper IDs that were successfully downloaded
    """
    logger.info(f"Starting paper scraping with categories: {categories}, limit: {limit}")
    
    # Initialize the arXiv client with a conservative wait time
    papers_dir = os.getenv("PAPERS_DIR", "data/pdfs")
    client = ArxivClient(papers_dir=papers_dir, wait_time=5)
    
    # Initialize the paper queue
    queue = PaperQueue()
    
    # Initialize the database
    db = Database()
    
    # Search for papers
    papers = await client.search_papers(
        categories=categories,
        max_results=limit,
        sort_by="submittedDate",
        sort_order="descending"
    )
    
    if not papers:
        logger.warning("No papers found.")
        return []
    
    logger.info(f"Found {len(papers)} papers")
    
    # Add papers to the queue
    added_count = await queue.add_papers(papers)
    logger.info(f"Added {added_count} new papers to the queue")
    
    # Also add papers directly to the database
    papers_added_to_db = 0
    for paper in papers:
        # Only add if the paper doesn't already exist in the database
        if not db.get_paper(paper["paper_id"]):
            db.add_paper(paper)
            papers_added_to_db += 1
    
    logger.info(f"Added {papers_added_to_db} new papers to the database")
    
    # Download PDFs if requested
    downloaded_papers = []
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
                downloaded_papers.append(paper_id)
            except Exception as e:
                logger.error(f"Failed to download PDF for {paper_id}: {e}")
    
    # Log queue status
    queue_size = await queue.get_queue_size()
    processed_count = await queue.get_processed_count()
    logger.info(f"Current queue status: {queue_size} papers in queue, {processed_count} papers processed")
    
    return downloaded_papers


async def download_missing_pdfs(paper_ids: List[str]) -> List[str]:
    """
    Download PDFs for papers that don't have them yet.
    
    Args:
        paper_ids: List of paper IDs to download PDFs for
        
    Returns:
        List of paper IDs that were successfully downloaded
    """
    logger.info(f"Checking PDFs for {len(paper_ids)} papers...")
    
    papers_dir = os.getenv("PAPERS_DIR", "data/pdfs")
    client = ArxivClient(papers_dir=papers_dir, wait_time=3)
    
    downloaded_papers = []
    missing_pdfs = []
    
    # Check which papers are missing PDFs
    for paper_id in paper_ids:
        pdf_path = Path(papers_dir) / f"{paper_id}.pdf"
        if not pdf_path.exists():
            missing_pdfs.append(paper_id)
    
    if not missing_pdfs:
        logger.info("All papers already have PDFs")
        return []
    
    logger.info(f"Downloading {len(missing_pdfs)} missing PDFs...")
    
    # Download missing PDFs
    for i, paper_id in enumerate(missing_pdfs):
        logger.info(f"Downloading PDF {i+1}/{len(missing_pdfs)}: {paper_id}")
        
        try:
            pdf_path = await client.download_pdf(paper_id)
            logger.info(f"Downloaded PDF for {paper_id} to {pdf_path}")
            downloaded_papers.append(paper_id)
        except Exception as e:
            logger.error(f"Failed to download PDF for {paper_id}: {e}")
    
    return downloaded_papers


async def sync_queue_to_database():
    """
    Sync the paper queue with the database to ensure all queued papers
    are also in the database.
    
    Returns:
        Number of papers synced to the database
    """
    logger.info("Syncing paper queue to database...")
    
    # Initialize the paper queue and database
    queue = PaperQueue()
    db = Database()
    
    # Get all papers from the queue
    batch_size = 100
    queue_size = await queue.get_queue_size()
    
    if queue_size == 0:
        logger.info("Queue is empty, nothing to sync")
        return 0
    
    # Get papers in batches to avoid memory issues with large queues
    total_papers = 0
    synced_papers = 0
    
    # Get all unprocessed papers from the queue
    all_papers = await queue.get_next_batch(queue_size)
    total_papers = len(all_papers)
    
    # Add each paper to the database if it doesn't already exist
    for paper in all_papers:
        paper_id = paper["paper_id"]
        if not db.get_paper(paper_id):
            db.add_paper(paper)
            synced_papers += 1
    
    logger.info(f"Synced {synced_papers} papers from queue to database (out of {total_papers} total papers in queue)")
    return synced_papers


async def get_all_papers_from_db(db):
    """
    Get all papers from the database.
    
    Args:
        db: Database instance
        
    Returns:
        List of all papers in the database
    """
    try:
        # We need to use raw SQL since there's no convenient method for getting all papers
        conn = db._get_connection()
        cursor = conn.execute("SELECT * FROM papers")
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        papers = [dict(row) for row in rows]
        
        # Convert comma-separated strings back to lists
        for paper in papers:
            if "categories" in paper and isinstance(paper["categories"], str):
                paper["categories"] = paper["categories"].split(",")
        
        return papers
    except Exception as e:
        logger.error(f"Failed to get all papers from database: {e}")
        return []


async def analyze_papers(
    paper_ids: Optional[List[str]] = None,
    limit: int = 10,
    max_workers: int = 8
) -> None:
    """
    Analyze papers with OpenAI only.
    
    Args:
        paper_ids: Optional list of paper IDs to analyze.
                  If None, gets unprocessed papers from the database.
        limit: Maximum number of papers to process if paper_ids is None
        max_workers: Maximum number of concurrent workers
    """
    logger.info("Starting OpenAI-only paper analysis with high-throughput configuration")
    
    # First, ensure the paper queue is synced with the database
    await sync_queue_to_database()
    
    # Get papers to analyze
    db = Database()
    if paper_ids:
        papers_to_analyze = paper_ids
    else:
        # Get unprocessed papers
        unprocessed_papers = db.get_unprocessed_papers(limit=limit)
        papers_to_analyze = [p["paper_id"] for p in unprocessed_papers]
    
    if not papers_to_analyze:
        logger.info("No papers to analyze")
        return
    
    logger.info(f"Found {len(papers_to_analyze)} papers to analyze")
    
    # Download any missing PDFs before analysis
    await download_missing_pdfs(papers_to_analyze)
    
    # Initialize the orchestrator
    orchestrator = OpenAIOnlyOrchestrator(max_workers=max_workers)
    
    # Analyze papers
    results = await orchestrator.analyze_papers(paper_ids=papers_to_analyze, limit=limit)
    
    # Log the results
    if results:
        logger.info(f"Successfully analyzed {len(results)} papers with OpenAI o1")
        
        # Sort and log the top papers
        sorted_results = sorted(results, key=lambda x: x["combined_score"], reverse=True)
        
        logger.info("Top paper opportunities:")
        for i, result in enumerate(sorted_results[:5]):
            paper_id = result["paper_id"]
            score = result["combined_score"]
            title = result["title"]
            
            logger.info(f"{i+1}. {title} (ID: {paper_id}, Score: {score:.2f})")
    else:
        logger.info("No papers were analyzed")
        
        # Check the database to see if there are any papers at all
        all_papers = await get_all_papers_from_db(db)
        
        if not all_papers:
            logger.warning("Database is empty. Try running the scrape command first.")
        else:
            # Get analyses by directly querying the database
            conn = db._get_connection()
            cursor = conn.execute("SELECT paper_id FROM analyses")
            analyzed_paper_ids = {row['paper_id'] for row in cursor.fetchall()}
            conn.close()
            
            papers_with_analysis = [p for p in all_papers if p["paper_id"] in analyzed_paper_ids]
            
            if len(papers_with_analysis) == len(all_papers):
                logger.info(f"All {len(all_papers)} papers in the database have already been analyzed. Nothing new to process.")
            else:
                logger.info(f"Database has {len(all_papers)} papers, {len(papers_with_analysis)} already analyzed.")
                
                # Check PDF availability
                papers_dir = os.getenv("PAPERS_DIR", "data/pdfs")
                unanalyzed_papers = [p for p in all_papers if p["paper_id"] not in analyzed_paper_ids]
                papers_with_pdfs = [p for p in unanalyzed_papers if Path(papers_dir, f"{p['paper_id']}.pdf").exists()]
                
                if len(papers_with_pdfs) == 0:
                    logger.warning(f"None of the {len(unanalyzed_papers)} unanalyzed papers have PDFs. Try downloading PDFs first.")
                else:
                    logger.warning(f"There are {len(papers_with_pdfs)} unanalyzed papers with PDFs, but none were processed.")


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run arxiv-opportunity-finder with OpenAI only")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape papers from arXiv")
    scrape_parser.add_argument(
        "--categories",
        type=str,
        default="cs.AI,cs.LG,cs.CV",
        help="Comma-separated list of categories to scrape (default: cs.AI,cs.LG,cs.CV)"
    )
    scrape_parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of papers to scrape (default: 30)"
    )
    scrape_parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading PDFs"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze papers with OpenAI")
    analyze_parser.add_argument(
        "--paper-ids",
        type=str,
        help="Comma-separated list of paper IDs to analyze"
    )
    analyze_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of papers to process (default: 10)"
    )
    analyze_parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent workers (default: 8)"
    )
    
    # Pipeline command to run both scrape and analyze
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full pipeline")
    pipeline_parser.add_argument(
        "--categories",
        type=str,
        default="cs.AI,cs.LG,cs.CV",
        help="Comma-separated list of categories to scrape (default: cs.AI,cs.LG,cs.CV)"
    )
    pipeline_parser.add_argument(
        "--scrape-limit",
        type=int,
        default=30,
        help="Maximum number of papers to scrape (default: 30)"
    )
    pipeline_parser.add_argument(
        "--analyze-limit",
        type=int,
        default=15,
        help="Maximum number of papers to analyze (default: 15)"
    )
    pipeline_parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent workers for analysis (default: 8)"
    )
    
    return parser.parse_args()


async def main():
    """
    Main entry point.
    """
    # Create necessary directories
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/pdfs").mkdir(parents=True, exist_ok=True)
    
    # Parse command line arguments
    args = parse_args()
    
    if args.command == "scrape":
        # Parse categories
        categories = [cat.strip() for cat in args.categories.split(",")]
        
        # Scrape papers
        await scrape_papers(
            categories=categories,
            limit=args.limit,
            download_pdfs=not args.no_download
        )
    
    elif args.command == "analyze":
        # Parse paper IDs if provided
        paper_ids = None
        if args.paper_ids:
            paper_ids = [pid.strip() for pid in args.paper_ids.split(",")]
        
        # Analyze papers
        await analyze_papers(
            paper_ids=paper_ids,
            limit=args.limit,
            max_workers=args.max_workers
        )
    
    elif args.command == "pipeline":
        # Parse categories
        categories = [cat.strip() for cat in args.categories.split(",")]
        
        # Run the full pipeline
        logger.info("Running the full pipeline")
        
        # 1. Scrape papers
        downloaded_papers = await scrape_papers(
            categories=categories,
            limit=args.scrape_limit,
            download_pdfs=True
        )
        
        # 2. Analyze papers (prioritize the ones we just downloaded)
        if downloaded_papers:
            logger.info(f"Analyzing the {min(len(downloaded_papers), args.analyze_limit)} papers we just downloaded")
            paper_ids = downloaded_papers[:args.analyze_limit]
            await analyze_papers(
                paper_ids=paper_ids,
                limit=args.analyze_limit,
                max_workers=args.max_workers
            )
        else:
            # Fall back to analyzing whatever's in the database
            await analyze_papers(
                limit=args.analyze_limit,
                max_workers=args.max_workers
            )
    
    else:
        logger.error("Invalid command. Use 'scrape', 'analyze', or 'pipeline'.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 