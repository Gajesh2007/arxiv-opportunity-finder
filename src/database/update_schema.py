"""
Simple migration script to update the database schema with the layman_explanation field.
"""
import os
import sys
import sqlite3
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_schema():
    """Update the database schema to add the layman_explanation column to the analyses table."""
    db_path = os.getenv("DATABASE_PATH", "data/database.sqlite")
    logger.info(f"Updating schema for database at {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # Check if the column already exists
        cursor = conn.execute("PRAGMA table_info(analyses)")
        columns = [column['name'] for column in cursor.fetchall()]
        
        if 'layman_explanation' not in columns:
            logger.info("Adding layman_explanation column to analyses table")
            conn.execute("ALTER TABLE analyses ADD COLUMN layman_explanation TEXT")
            conn.commit()
            logger.info("Schema update completed successfully")
        else:
            logger.info("layman_explanation column already exists in analyses table")
    except Exception as e:
        logger.error(f"Error updating schema: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    update_schema() 