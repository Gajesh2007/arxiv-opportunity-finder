"""
Paper queue module for managing papers to be processed.
"""
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class PaperQueue:
    """
    Queue for managing papers to be processed.
    Handles deduplication and persistence.
    """
    
    def __init__(self, queue_file: Optional[str] = None):
        """
        Initialize the paper queue.
        
        Args:
            queue_file: Path to the file where the queue will be persisted.
                       If None, a default path will be used.
        """
        self.queue_file = Path(queue_file) if queue_file else Path("data/paper_queue.json")
        self.queue: List[Dict[str, Any]] = []
        self.processed_ids: Set[str] = set()
        self.lock = asyncio.Lock()
        
        # Load existing queue if available
        self._load_queue()
    
    async def add_papers(self, papers: List[Dict[str, Any]]) -> int:
        """
        Add papers to the queue.
        Deduplicates papers based on their IDs.
        
        Args:
            papers: List of paper metadata dictionaries
            
        Returns:
            Number of new papers added to the queue
        """
        async with self.lock:
            # Get current paper IDs in the queue
            current_ids = {paper["paper_id"] for paper in self.queue}
            current_ids.update(self.processed_ids)
            
            # Filter out papers that are already in the queue or processed
            new_papers = [paper for paper in papers if paper["paper_id"] not in current_ids]
            
            # Add new papers to the queue
            self.queue.extend(new_papers)
            
            # Save the updated queue
            self._save_queue()
            
            logger.info(f"Added {len(new_papers)} new papers to the queue")
            return len(new_papers)
    
    async def get_next_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Get the next batch of papers to process.
        
        Args:
            batch_size: Maximum number of papers to return
            
        Returns:
            List of paper metadata dictionaries
        """
        async with self.lock:
            # Get the next batch of papers
            batch = self.queue[:batch_size]
            
            logger.info(f"Retrieved {len(batch)} papers from queue")
            return batch
    
    async def mark_processed(self, paper_ids: List[str]) -> None:
        """
        Mark papers as processed and remove them from the queue.
        
        Args:
            paper_ids: List of paper IDs that have been processed
        """
        async with self.lock:
            # Add paper IDs to the processed set
            self.processed_ids.update(paper_ids)
            
            # Remove the papers from the queue
            self.queue = [paper for paper in self.queue if paper["paper_id"] not in paper_ids]
            
            # Save the updated queue
            self._save_queue()
            
            logger.info(f"Marked {len(paper_ids)} papers as processed")
    
    async def get_queue_size(self) -> int:
        """
        Get the current size of the queue.
        
        Returns:
            Number of papers in the queue
        """
        async with self.lock:
            return len(self.queue)
    
    async def get_processed_count(self) -> int:
        """
        Get the number of processed papers.
        
        Returns:
            Number of processed papers
        """
        async with self.lock:
            return len(self.processed_ids)
    
    def _load_queue(self) -> None:
        """
        Load the queue from a file if it exists.
        """
        if not self.queue_file.exists():
            logger.info(f"Queue file {self.queue_file} does not exist, starting with empty queue")
            return
        
        try:
            with open(self.queue_file, "r") as f:
                data = json.load(f)
                self.queue = data.get("queue", [])
                self.processed_ids = set(data.get("processed_ids", []))
                
            logger.info(f"Loaded queue with {len(self.queue)} papers and {len(self.processed_ids)} processed papers")
        except Exception as e:
            logger.error(f"Failed to load queue: {e}")
            # Start with empty queue if loading fails
            self.queue = []
            self.processed_ids = set()
    
    def _save_queue(self) -> None:
        """
        Save the queue to a file.
        """
        try:
            # Create the directory if it doesn't exist
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "queue": self.queue,
                "processed_ids": list(self.processed_ids)
            }
            
            with open(self.queue_file, "w") as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Queue saved to {self.queue_file}")
        except Exception as e:
            logger.error(f"Failed to save queue: {e}") 