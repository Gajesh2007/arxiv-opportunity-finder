#!/usr/bin/env python
"""
Test script to validate the Claude and OpenAI clients with PDF handling.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.analyzer.claude_client import ClaudeClient
from src.analyzer.openai_client import OpenAIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def test_claude():
    """Test the Claude client with PDF analysis."""
    logger.info("Testing Claude client...")
    
    # Initialize Claude client
    claude_client = ClaudeClient()
    
    # Sample PDF path
    pdf_path = Path("data/pdfs/sample_paper.pdf")
    
    # Test PDF analysis
    try:
        analysis = await claude_client.analyze_paper(str(pdf_path))
        
        logger.info("Claude analysis results:")
        logger.info(f"Innovation Score: {analysis.get('innovation_score')}")
        logger.info(f"PoC Potential Score: {analysis.get('poc_potential_score')}")
        logger.info(f"Wow Factor Score: {analysis.get('wow_factor_score')}")
        logger.info(f"Implementation Complexity Score: {analysis.get('implementation_complexity_score')}")
        logger.info(f"Combined Score: {analysis.get('combined_score')}")
        
        return analysis
    except Exception as e:
        logger.error(f"Claude test failed: {e}", exc_info=True)
        return None


async def test_openai(claude_analysis=None):
    """Test the OpenAI client with PDF implementation planning."""
    logger.info("Testing OpenAI client...")
    
    # Initialize OpenAI client
    openai_client = OpenAIClient()
    
    # Sample PDF path
    pdf_path = Path("data/pdfs/sample_paper.pdf")
    
    # Sample paper metadata
    paper_metadata = {
        "paper_id": "2303.08774",
        "title": "GPT-4 Technical Report",
        "authors": "OpenAI",
        "categories": "cs.CL, cs.AI",
        "abstract": "We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs."
    }
    
    # Test implementation planning
    try:
        plan = await openai_client.plan_implementation(paper_metadata, str(pdf_path), claude_analysis)
        
        logger.info("OpenAI implementation plan results:")
        logger.info(f"Steps: {len(plan.get('steps', []))}")
        logger.info(f"Time Estimate: {plan.get('time_estimate')}")
        logger.info(f"Resources: {len(plan.get('resources', []))}")
        logger.info(f"Challenges: {len(plan.get('challenges', []))}")
        
        return plan
    except Exception as e:
        logger.error(f"OpenAI test failed: {e}", exc_info=True)
        return None


async def main():
    """Main entry point."""
    logger.info("Starting client tests...")
    
    # Test Claude first
    claude_analysis = await test_claude()
    
    # Test OpenAI next, passing Claude analysis if available
    await test_openai(claude_analysis)
    
    logger.info("Client tests completed.")


if __name__ == "__main__":
    asyncio.run(main()) 