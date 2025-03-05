"""
Orchestrator for coordinating the analysis of papers.
Manages the flow between Claude and OpenAI analysis.
"""
import os
import sys
import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tenacity.retry import retry_base
import random

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.analyzer.claude_client import ClaudeClient
from src.analyzer.openai_client import OpenAIClient
from src.database.db import Database
from src.config import config

logger = logging.getLogger(__name__)


class AnalyzerOrchestrator:
    """
    Orchestrator for coordinating the analysis of papers.
    """
    
    def __init__(
        self,
        db: Optional[Database] = None,
        claude_client: Optional[ClaudeClient] = None,
        openai_client: Optional[OpenAIClient] = None,
        max_workers: int = 5,
        openai_only: bool = False
    ):
        """
        Initialize the analyzer orchestrator.
        
        Args:
            db: Database instance
            claude_client: Claude client instance
            openai_client: OpenAI client instance
            max_workers: Maximum number of concurrent workers
            openai_only: If True, skip Claude analysis and use only OpenAI
        """
        # Initialize the database
        self.db = db or Database()
        
        # Initialize the Claude client if not in openai_only mode
        self.claude_client = None if openai_only else (claude_client or ClaudeClient())
        
        # Initialize the OpenAI client
        self.openai_client = openai_client or OpenAIClient()
        
        # Whether to use only OpenAI
        self.openai_only = openai_only
        
        # Maximum number of concurrent workers
        self.max_workers = max_workers
        
        # Semaphore for limiting concurrent API calls
        self.semaphore = asyncio.Semaphore(max_workers)
        
        logger.info(f"AnalyzerOrchestrator initialized with max_workers: {max_workers}, openai_only: {openai_only}")
    
    async def analyze_papers(self, paper_ids: Optional[List[str]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Analyze a batch of papers.
        
        Args:
            paper_ids: Optional list of paper IDs to analyze.
                      If None, gets unprocessed papers from the database.
            limit: Maximum number of papers to process if paper_ids is None
            
        Returns:
            List of analysis results
        """
        logger.info(f"Starting analysis of papers: {paper_ids if paper_ids else f'(up to {limit} unprocessed papers)'}")
        
        # Get the papers to analyze
        if paper_ids:
            papers = [self.db.get_paper(paper_id) for paper_id in paper_ids]
            papers = [p for p in papers if p is not None]
        else:
            papers = self.db.get_unprocessed_papers(limit=limit)
        
        if not papers:
            logger.info("No papers to analyze")
            return []
        
        logger.info(f"Analyzing {len(papers)} papers")
        
        # Process the papers in parallel
        tasks = [self._process_paper(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return the results
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing paper: {result}")
            else:
                successful_results.append(result)
        
        logger.info(f"Completed analysis of {len(successful_results)} papers successfully")
        return successful_results
    
    async def _process_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single paper with both Claude and OpenAI, or OpenAI only.
        
        Args:
            paper: Dictionary containing paper metadata
            
        Returns:
            Dictionary containing analysis results
        """
        # Get the paper ID
        paper_id = paper["paper_id"]
        logger.info(f"Processing paper: {paper_id}")
        
        # Get the PDF path
        pdf_path = paper.get("pdf_path")
        if not pdf_path:
            logger.error(f"No PDF path for paper: {paper_id}")
            raise ValueError(f"No PDF path for paper: {paper_id}")
        
        # Check if the PDF exists
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Process with semaphore to limit concurrent API calls
        async with self.semaphore:
            try:
                if self.openai_only:
                    # Skip Claude analysis when in OpenAI-only mode
                    logger.info(f"Using OpenAI-only mode for paper: {paper_id}")
                    openai_analysis = await self._plan_with_openai(paper, pdf_path)
                    analysis = self._create_openai_only_analysis(paper, openai_analysis)
                else:
                    # First, analyze with Claude
                    claude_analysis = await self._analyze_with_claude(paper, pdf_path)
                    
                    # Then, plan with OpenAI
                    openai_analysis = await self._plan_with_openai(paper, pdf_path, claude_analysis)
                    
                    # Combine the results
                    analysis = self._combine_results(paper, claude_analysis, openai_analysis)
                
                # Store the results in the database
                self.db.add_analysis(analysis)
                
                # Mark the paper as processed
                self.db.mark_paper_processed(paper_id)
                
                logger.info(f"Completed analysis for paper: {paper_id}")
                return analysis
            
            except Exception as e:
                logger.error(f"Failed to process paper {paper_id}: {e}", exc_info=True)
                raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def _analyze_with_claude(self, paper: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
        """
        Analyze a paper with Claude.
        Includes retry logic for transient failures.
        
        Args:
            paper: Dictionary containing paper metadata
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing Claude analysis results
        """
        logger.info(f"Analyzing paper with Claude: {paper['paper_id']}")
        try:
            return await self.claude_client.analyze_paper(pdf_path)
        except Exception as e:
            logger.error(f"Claude analysis failed for paper {paper['paper_id']}: {e}", exc_info=True)
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def _plan_with_openai(self, paper: Dict[str, Any], pdf_path: str, claude_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Plan implementation with OpenAI.
        Includes retry logic for transient failures.
        
        Args:
            paper: Dictionary containing paper metadata
            pdf_path: Path to the PDF file
            claude_analysis: Results from Claude analysis (optional in OpenAI-only mode)
            
        Returns:
            Dictionary containing OpenAI implementation plan
        """
        logger.info(f"Planning implementation with OpenAI: {paper['paper_id']}")
        try:
            return await self.openai_client.plan_implementation(paper, pdf_path, claude_analysis)
        except Exception as e:
            logger.error(f"OpenAI planning failed for paper {paper['paper_id']}: {e}", exc_info=True)
            raise
    
    def _combine_results(self, paper: Dict[str, Any], claude_analysis: Dict[str, Any], openai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine results from Claude and OpenAI analyses.
        
        Args:
            paper: Dictionary containing paper metadata
            claude_analysis: Results from Claude analysis
            openai_analysis: Results from OpenAI implementation planning
            
        Returns:
            Combined analysis results
        """
        # Generate a unique ID for the analysis
        analysis_id = str(uuid.uuid4())
        
        # Current datetime
        now = datetime.now().isoformat()
        
        # Extract scores from Claude analysis
        innovation_score = claude_analysis.get("innovation_score", 0) / 10
        poc_potential_score = claude_analysis.get("poc_potential_score", 0) / 10
        wow_factor_score = claude_analysis.get("wow_factor_score", 0) / 10
        implementation_complexity_score = claude_analysis.get("implementation_complexity_score", 0) / 10
        
        # Use Claude's combined score or calculate it
        combined_score = claude_analysis.get("combined_score", 0)
        
        # Create the combined analysis
        analysis = {
            "analysis_id": analysis_id,
            "paper_id": paper["paper_id"],
            "claude_analysis": claude_analysis,
            "openai_analysis": openai_analysis,
            "innovation_score": innovation_score,
            "poc_potential_score": poc_potential_score,
            "wow_factor_score": wow_factor_score,
            "implementation_complexity": implementation_complexity_score,
            "combined_score": combined_score,
            "layman_explanation": openai_analysis.get("layman_explanation", ""),
            "processed_date": now
        }
        
        logger.debug(f"Combined analysis for paper {paper['paper_id']}: combined_score={combined_score}")
        return analysis
        
    def _create_openai_only_analysis(self, paper: Dict[str, Any], openai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create analysis results from OpenAI only, when skipping Claude.
        
        Args:
            paper: Dictionary containing paper metadata
            openai_analysis: Results from OpenAI analysis
            
        Returns:
            Analysis results adapted to match the structure expected from the combined analysis
        """
        # Generate a unique ID for the analysis
        analysis_id = str(uuid.uuid4())
        
        # Current datetime
        now = datetime.now().isoformat()
        
        # Extract scores from OpenAI analysis - converting from 0-10 scale to 0-1 scale
        innovation_score = openai_analysis.get("innovation_score", 0) / 10
        technical_feasibility_score = openai_analysis.get("technical_feasibility_score", 0) / 10
        market_potential_score = openai_analysis.get("market_potential_score", 0) / 10
        impact_score = openai_analysis.get("impact_score", 0) / 10
        
        # Calculate combined score using equal weights
        combined_score = (
            innovation_score * 0.25 +
            technical_feasibility_score * 0.25 +
            market_potential_score * 0.25 +
            impact_score * 0.25
        )
        
        # Create the analysis object
        analysis = {
            "analysis_id": analysis_id,
            "paper_id": paper["paper_id"],
            "openai_analysis": openai_analysis,
            "innovation_score": innovation_score,
            "poc_potential_score": technical_feasibility_score,  # Map to equivalent field
            "wow_factor_score": impact_score,  # Map to equivalent field
            "implementation_complexity": 1 - technical_feasibility_score,  # Invert the score
            "combined_score": combined_score,
            "layman_explanation": openai_analysis.get("layman_explanation", ""),
            "processed_date": now
        }
        
        logger.debug(f"OpenAI-only analysis for paper {paper['paper_id']}: combined_score={combined_score}")
        return analysis 