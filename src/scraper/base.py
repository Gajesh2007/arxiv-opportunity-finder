"""
Base scraper module defining the abstract interface for all scrapers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class Scraper(ABC):
    """
    Abstract base class for all paper scrapers.
    Defines the interface that all concrete scraper implementations must follow.
    """

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_paper_metadata(self, paper_id: str) -> Dict[str, Any]:
        """
        Retrieve detailed metadata for a specific paper.

        Args:
            paper_id: Unique identifier for the paper

        Returns:
            Dictionary containing paper metadata
        """
        pass

    @abstractmethod
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
        pass 