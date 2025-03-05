"""
Main entry point for the ArXiv Opportunity Finder web application.
"""
import os
import sys
from pathlib import Path
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.api.app import app
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "webapp.log"), mode="a")
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run the web application."""
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    
    logger.info(f"Starting ArXiv Opportunity Finder web application on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    # Make sure the logs directory exists
    logs_dir = os.path.join(project_root, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    # Run the app
    app.run(host="0.0.0.0", port=port, debug=debug)

if __name__ == "__main__":
    main() 