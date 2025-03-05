"""
Script to initialize the database.
Creates the database schema if it doesn't exist.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.database.db import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("data/logs/database.log"))
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def init_database():
    """
    Initialize the database.
    """
    logger.info("Initializing database...")
    
    # Get database path from environment or use default
    db_path = os.getenv("DATABASE_PATH", "data/database.sqlite")
    
    # Create log directory
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize the database
        db = Database(db_path)
        db.init_db()
        
        logger.info(f"Database initialized successfully at {db_path}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    init_database() 