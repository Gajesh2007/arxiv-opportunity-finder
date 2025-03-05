#!/usr/bin/env python3
"""
Run script for the ArXiv Opportunity Finder streaming pipeline.
This script provides a command line interface to run the streaming pipeline.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.streaming_pipeline import main

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("data/logs/run_streaming.log"))
    ]
)

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Create necessary directories
    for directory in ["data/pdfs", "data/logs"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Run the pipeline
    asyncio.run(main()) 