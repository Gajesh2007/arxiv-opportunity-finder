"""
arXiv API client implementation.
Uses the arxiv package to search for papers and retrieve metadata.
"""
import os
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import arxiv
import httpx

from src.scraper.base import Scraper

logger = logging.getLogger(__name__)


class ArxivClient(Scraper):
    """
    Implementation of the Scraper interface for the arXiv API.
    """
    
    def __init__(self, papers_dir: Optional[str] = None, wait_time: int = 3):
        """
        Initialize the arXiv client.
        
        Args:
            papers_dir: Directory to store downloaded papers
            wait_time: Time to wait between API calls to avoid rate limiting
        """
        self.wait_time = wait_time
        
        if papers_dir is None:
            # Use default from .env or fallback
            papers_dir = os.getenv("PAPERS_DIR", "data/pdfs")
            
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=1,
            num_retries=3
        )
        
        logger.info(f"ArxivClient initialized with papers_dir: {self.papers_dir}")
    
    async def search_papers(
        self, 
        categories: List[str], 
        max_results: int, 
        sort_by: str = "submittedDate", 
        sort_order: str = "descending"
    ) -> List[Dict[str, Any]]:
        """
        Search for papers in the specified categories.
        
        Args:
            categories: List of category identifiers to search in
            max_results: Maximum number of results to return
            sort_by: Field to sort results by
            sort_order: Order of sorting ('ascending' or 'descending')
            
        Returns:
            List of paper metadata dictionaries
        """
        logger.info(f"Searching for papers in categories: {categories}, max_results: {max_results}")
        
        # Convert sort parameters to arxiv package format
        sort_mapping = {
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        }
        
        sort_criterion = sort_mapping.get(sort_by, arxiv.SortCriterion.SubmittedDate)
        
        # Build the category query string: cat:cs.AI OR cat:cs.LG etc.
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        # Create the search query
        search = arxiv.Search(
            query=category_query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending if sort_order == "descending" else arxiv.SortOrder.Ascending
        )
        
        # Run the search synchronously (arxiv package is not async)
        # We'll convert the results to a list of dictionaries
        def _run_search():
            results = list(self.client.results(search))
            return [self._paper_to_dict(paper) for paper in results]
        
        # Run synchronously in a thread pool
        results = await asyncio.to_thread(_run_search)
        
        logger.info(f"Found {len(results)} papers")
        return results
    
    async def get_paper_metadata(self, paper_id: str) -> Dict[str, Any]:
        """
        Retrieve detailed metadata for a specific paper.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            Dictionary containing paper metadata
        """
        logger.info(f"Getting metadata for paper: {paper_id}")
        
        # Ensure paper_id is in the correct format (without version)
        paper_id = paper_id.split("v")[0] if "v" in paper_id else paper_id
        
        # Create a search for a specific paper
        search = arxiv.Search(
            id_list=[paper_id],
            max_results=1
        )
        
        # Run the search synchronously
        def _get_paper():
            results = list(self.client.results(search))
            if not results:
                return None
            return self._paper_to_dict(results[0])
        
        # Run in a thread pool
        paper = await asyncio.to_thread(_get_paper)
        
        if paper is None:
            raise ValueError(f"Paper with ID {paper_id} not found.")
        
        return paper
    
    async def download_pdf(self, paper_id: str, output_path: Optional[str] = None) -> str:
        """
        Download the PDF for a specific paper.
        
        Args:
            paper_id: Unique identifier for the paper
            output_path: Optional path to save the PDF to.
                         If not provided, a default path will be used.
                         
        Returns:
            Path to the downloaded PDF file
        """
        logger.info(f"Downloading PDF for paper: {paper_id}")
        
        # Get the paper metadata first
        paper = await self.get_paper_metadata(paper_id)
        
        # Determine the output path
        if output_path is None:
            # Use the default location: papers_dir/paper_id.pdf
            output_path = self.papers_dir / f"{paper_id}.pdf"
        else:
            output_path = Path(output_path)
            
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if the file already exists
        if output_path.exists():
            logger.info(f"PDF already exists at {output_path}")
            return str(output_path)
        
        # Get the PDF URL
        pdf_url = paper["pdf_url"]
        
        # Download the PDF
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(pdf_url)
            
            if response.status_code == 200:
                # Save the PDF
                with open(output_path, "wb") as f:
                    f.write(response.content)
                    
                logger.info(f"PDF downloaded to {output_path}")
                
                # Wait a bit to avoid overloading the server
                await asyncio.sleep(self.wait_time)
                
                return str(output_path)
            else:
                raise Exception(f"Failed to download PDF: HTTP {response.status_code}")
    
    def _paper_to_dict(self, paper: arxiv.Result) -> Dict[str, Any]:
        """
        Convert an arxiv.Result object to a dictionary.
        
        Args:
            paper: arxiv.Result object
            
        Returns:
            Dictionary representation of the paper
        """
        # Extract authors
        authors = [author.name for author in paper.authors]
        
        # Create the dictionary
        return {
            "paper_id": paper.get_short_id(),
            "title": paper.title,
            "authors": authors,
            "author_str": ", ".join(authors),
            "abstract": paper.summary,
            "categories": paper.categories,
            "primary_category": paper.primary_category,
            "pdf_url": paper.pdf_url,
            "entry_id": paper.entry_id,
            "published": paper.published.isoformat() if paper.published else None,
            "updated": paper.updated.isoformat() if paper.updated else None,
            "comment": paper.comment,
            "journal_ref": paper.journal_ref,
            "doi": paper.doi
        } 