"""
Database utility functions for storing and retrieving papers and analyses.
Uses SQLite as the backing store.
"""
import os
import json
import uuid
import logging
import sqlite3
import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class Database:
    """
    Database class for interacting with the SQLite database.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file.
                    If None, uses the DATABASE_PATH environment variable or a default path.
        """
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "data/database.sqlite")
            
        self.db_path = Path(db_path)
        
        # Ensure the directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using database at {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a connection to the SQLite database.
        
        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # This enables dictionary-like access to rows
        return conn
    
    def init_db(self) -> None:
        """
        Initialize the database by creating tables if they don't exist.
        """
        try:
            # Read the schema SQL
            schema_path = Path(__file__).parent / "schema.sql"
            with open(schema_path, "r") as f:
                schema_sql = f.read()
            
            # Execute the schema SQL
            conn = self._get_connection()
            conn.executescript(schema_sql)
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def add_paper(self, paper: Dict[str, Any]) -> None:
        """
        Add a paper to the database.
        
        Args:
            paper: Dictionary containing paper metadata
        """
        try:
            # Convert categories list to comma-separated string
            if "categories" in paper and isinstance(paper["categories"], list):
                paper["categories"] = ",".join(paper["categories"])
            
            # Convert authors list to comma-separated string
            if "authors" in paper and isinstance(paper["authors"], list):
                paper["authors"] = ",".join(paper["authors"])
            elif "author_str" in paper:
                paper["authors"] = paper["author_str"]
            
            # Add scrape date if not present
            if "scrape_date" not in paper:
                paper["scrape_date"] = datetime.datetime.now().isoformat()
            
            conn = self._get_connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO papers 
                (paper_id, title, authors, categories, abstract, pdf_path, scrape_date, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper["paper_id"],
                    paper["title"],
                    paper["authors"],
                    paper["categories"],
                    paper["abstract"],
                    paper.get("pdf_path", None),
                    paper["scrape_date"],
                    paper.get("processed", 0)
                )
            )
            conn.commit()
            conn.close()
            
            logger.debug(f"Added paper {paper['paper_id']} to database")
        except Exception as e:
            logger.error(f"Failed to add paper to database: {e}")
            raise
    
    def add_papers(self, papers: List[Dict[str, Any]]) -> None:
        """
        Add multiple papers to the database.
        
        Args:
            papers: List of dictionaries containing paper metadata
        """
        try:
            conn = self._get_connection()
            
            for paper in papers:
                # Convert categories list to comma-separated string
                if "categories" in paper and isinstance(paper["categories"], list):
                    paper["categories"] = ",".join(paper["categories"])
                
                # Convert authors list to comma-separated string
                if "authors" in paper and isinstance(paper["authors"], list):
                    paper["authors"] = ",".join(paper["authors"])
                elif "author_str" in paper:
                    paper["authors"] = paper["author_str"]
                
                # Add scrape date if not present
                if "scrape_date" not in paper:
                    paper["scrape_date"] = datetime.datetime.now().isoformat()
                
                conn.execute(
                    """
                    INSERT OR REPLACE INTO papers 
                    (paper_id, title, authors, categories, abstract, pdf_path, scrape_date, processed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper["paper_id"],
                        paper["title"],
                        paper["authors"],
                        paper["categories"],
                        paper["abstract"],
                        paper.get("pdf_path", None),
                        paper["scrape_date"],
                        paper.get("processed", 0)
                    )
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added {len(papers)} papers to database")
        except Exception as e:
            logger.error(f"Failed to add papers to database: {e}")
            raise
    
    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a paper from the database by ID.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            Dictionary containing paper metadata, or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            # Convert to dictionary
            paper = dict(row)
            
            # Convert comma-separated strings back to lists
            if "categories" in paper and isinstance(paper["categories"], str):
                paper["categories"] = paper["categories"].split(",")
            
            return paper
        except Exception as e:
            logger.error(f"Failed to get paper from database: {e}")
            raise
    
    def get_unprocessed_papers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get papers that have not been processed yet.
        
        Args:
            limit: Maximum number of papers to return
            
        Returns:
            List of dictionaries containing paper metadata
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM papers WHERE processed = 0 ORDER BY scrape_date DESC LIMIT ?",
                (limit,)
            )
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
            logger.error(f"Failed to get unprocessed papers from database: {e}")
            raise
    
    def mark_paper_processed(self, paper_id: str) -> None:
        """
        Mark a paper as processed.
        
        Args:
            paper_id: Unique identifier for the paper
        """
        try:
            conn = self._get_connection()
            conn.execute(
                "UPDATE papers SET processed = 1 WHERE paper_id = ?",
                (paper_id,)
            )
            conn.commit()
            conn.close()
            
            logger.debug(f"Marked paper {paper_id} as processed")
        except Exception as e:
            logger.error(f"Failed to mark paper as processed: {e}")
            raise
    
    def get_all_papers(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get all papers from the database, with optional limit.
        
        Args:
            limit: Maximum number of papers to return (default 1000)
            
        Returns:
            List of dictionaries containing paper metadata
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM papers ORDER BY scrape_date DESC LIMIT ?",
                (limit,)
            )
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
            raise
    
    def add_analysis(self, analysis: Dict[str, Any]) -> None:
        """
        Add an analysis to the database.
        
        Args:
            analysis: Dictionary containing analysis data
        """
        try:
            # Generate a unique ID if not provided
            if "analysis_id" not in analysis:
                analysis["analysis_id"] = str(uuid.uuid4())
            
            # Add processed date if not present
            if "processed_date" not in analysis:
                analysis["processed_date"] = datetime.datetime.now().isoformat()
            
            # Convert JSON objects to strings if necessary
            if "claude_analysis" in analysis and not isinstance(analysis["claude_analysis"], str):
                analysis["claude_analysis"] = json.dumps(analysis["claude_analysis"])
            
            if "openai_analysis" in analysis and not isinstance(analysis["openai_analysis"], str):
                analysis["openai_analysis"] = json.dumps(analysis["openai_analysis"])
            
            conn = self._get_connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO analyses 
                (analysis_id, paper_id, claude_analysis, openai_analysis, 
                 innovation_score, poc_potential_score, wow_factor_score, 
                 implementation_complexity, combined_score, layman_explanation, processed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis["analysis_id"],
                    analysis["paper_id"],
                    analysis.get("claude_analysis", None),
                    analysis.get("openai_analysis", None),
                    analysis.get("innovation_score", 0),
                    analysis.get("poc_potential_score", 0),
                    analysis.get("wow_factor_score", 0),
                    analysis.get("implementation_complexity", 0),
                    analysis.get("combined_score", 0),
                    analysis.get("layman_explanation", ""),
                    analysis["processed_date"]
                )
            )
            
            # Mark the paper as processed
            conn.execute(
                "UPDATE papers SET processed = 1 WHERE paper_id = ?",
                (analysis["paper_id"],)
            )
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Added analysis for paper {analysis['paper_id']} to database")
        except Exception as e:
            logger.error(f"Failed to add analysis to database: {e}")
            raise
    
    def get_analysis(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the analysis for a specific paper.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            Dictionary containing analysis data, or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute("SELECT * FROM analyses WHERE paper_id = ?", (paper_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            # Convert to dictionary
            analysis = dict(row)
            
            # Parse JSON strings
            if "claude_analysis" in analysis and analysis["claude_analysis"]:
                try:
                    analysis["claude_analysis"] = json.loads(analysis["claude_analysis"])
                except json.JSONDecodeError:
                    pass  # Keep as string if not valid JSON
            
            if "openai_analysis" in analysis and analysis["openai_analysis"]:
                try:
                    analysis["openai_analysis"] = json.loads(analysis["openai_analysis"])
                except json.JSONDecodeError:
                    pass  # Keep as string if not valid JSON
            
            return analysis
        except Exception as e:
            logger.error(f"Failed to get analysis from database: {e}")
            raise
    
    def get_top_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top opportunities based on combined score.
        
        Args:
            limit: Maximum number of opportunities to return
            
        Returns:
            List of dictionaries containing paper and analysis data
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                SELECT p.*, a.innovation_score, a.poc_potential_score, 
                       a.wow_factor_score, a.implementation_complexity, a.combined_score
                FROM papers p
                JOIN analyses a ON p.paper_id = a.paper_id
                ORDER BY a.combined_score DESC
                LIMIT ?
                """,
                (limit,)
            )
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            opportunities = [dict(row) for row in rows]
            
            # Convert comma-separated strings back to lists
            for opp in opportunities:
                if "categories" in opp and isinstance(opp["categories"], str):
                    opp["categories"] = opp["categories"].split(",")
            
            return opportunities
        except Exception as e:
            logger.error(f"Failed to get top opportunities from database: {e}")
            raise
    
    def get_opportunity_count(self) -> Tuple[int, int, int]:
        """
        Get counts of total papers, processed papers, and top opportunities.
        
        Returns:
            Tuple of (total_papers, processed_papers, top_opportunities)
        """
        try:
            conn = self._get_connection()
            
            # Get total papers
            cursor = conn.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]
            
            # Get processed papers
            cursor = conn.execute("SELECT COUNT(*) FROM papers WHERE processed = 1")
            processed_papers = cursor.fetchone()[0]
            
            # Get papers with high scores (combined score > 0.7)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM analyses WHERE combined_score > 0.7"
            )
            top_opportunities = cursor.fetchone()[0]
            
            conn.close()
            
            return (total_papers, processed_papers, top_opportunities)
        except Exception as e:
            logger.error(f"Failed to get opportunity counts from database: {e}")
            raise
    
    def search_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers by title, abstract, or authors.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing paper metadata
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                SELECT p.*, a.combined_score
                FROM papers p
                LEFT JOIN analyses a ON p.paper_id = a.paper_id
                WHERE p.title LIKE ? OR p.abstract LIKE ? OR p.authors LIKE ?
                ORDER BY a.combined_score DESC NULLS LAST, p.scrape_date DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", f"%{query}%", limit)
            )
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
            logger.error(f"Failed to search papers in database: {e}")
            raise
    
    def get_analyses_with_papers(self, min_score: float = 0.0, limit: int = 100, 
                               sort_by: str = "combined_score", sort_direction: str = "desc",
                               category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get analyses joined with paper data, filtered and sorted.
        
        Args:
            min_score: Minimum combined score to include
            limit: Maximum number of results to return
            sort_by: Field to sort by
            sort_direction: Sort direction ('asc' or 'desc')
            category: Optional category to filter by
        
        Returns:
            List of dictionaries containing combined paper and analysis data
        """
        try:
            conn = self._get_connection()
            
            # Build the query
            query = """
            SELECT p.*, a.*
            FROM papers p
            JOIN analyses a ON p.paper_id = a.paper_id
            WHERE a.combined_score >= ?
            """
            params = [min_score]
            
            # Add category filter if provided
            if category:
                query += " AND p.categories LIKE ? "
                params.append(f"%{category}%")
            
            # Add sort order
            # Sanitize sort_by to prevent SQL injection
            valid_sort_fields = [
                "combined_score", "innovation_score", "technical_feasibility_score", 
                "market_potential_score", "impact_score", "published"
            ]
            if sort_by not in valid_sort_fields:
                sort_by = "combined_score"
                
            # Sanitize sort_direction
            if sort_direction.lower() not in ["asc", "desc"]:
                sort_direction = "desc"
                
            query += f" ORDER BY a.{sort_by} {sort_direction}"
            
            # Add limit
            query += " LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            opportunities = []
            for row in rows:
                opportunity = dict(row)
                
                # Convert comma-separated strings back to lists
                if "categories" in opportunity and isinstance(opportunity["categories"], str):
                    opportunity["categories"] = opportunity["categories"].split(",")
                    
                # Convert target_markets to list if it's a string
                if "target_markets" in opportunity and isinstance(opportunity["target_markets"], str):
                    try:
                        opportunity["target_markets"] = json.loads(opportunity["target_markets"])
                    except:
                        opportunity["target_markets"] = opportunity["target_markets"].split(",")
                
                # Extract data from openai_analysis JSON
                if "openai_analysis" in opportunity and opportunity["openai_analysis"]:
                    try:
                        if isinstance(opportunity["openai_analysis"], str):
                            openai_data = json.loads(opportunity["openai_analysis"])
                        else:
                            openai_data = opportunity["openai_analysis"]
                            
                        # Add market data and other fields to the root level
                        opportunity['time_to_market'] = openai_data.get('time_to_market', 'Unknown')
                        opportunity['target_markets'] = openai_data.get('target_markets', [])
                        opportunity['steps'] = openai_data.get('steps', [])
                        opportunity['resources'] = openai_data.get('resources', [])
                        opportunity['challenges'] = openai_data.get('challenges', [])
                        
                        # Extract layman's explanation if available
                        if 'layman_explanation' in openai_data and not opportunity.get('layman_explanation'):
                            opportunity['layman_explanation'] = openai_data.get('layman_explanation', '')
                        
                        # Include scores that might be missing
                        if 'technical_feasibility_score' in openai_data and not opportunity.get('technical_feasibility_score'):
                            opportunity['technical_feasibility_score'] = openai_data.get('technical_feasibility_score', 0)
                        if 'market_potential_score' in openai_data and not opportunity.get('market_potential_score'):
                            opportunity['market_potential_score'] = openai_data.get('market_potential_score', 0)
                        if 'impact_score' in openai_data and not opportunity.get('impact_score'):
                            opportunity['impact_score'] = openai_data.get('impact_score', 0)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, continue with existing data
                        pass
                        
                opportunities.append(opportunity)
            
            return opportunities
        except Exception as e:
            logger.error(f"Failed to get analyses with papers: {e}")
            raise
            
    def get_categories(self) -> List[str]:
        """
        Get all unique categories from the papers table.
        
        Returns:
            List of unique categories
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute("SELECT categories FROM papers")
            rows = cursor.fetchall()
            conn.close()
            
            # Extract categories
            all_categories = []
            for row in rows:
                if row["categories"]:
                    categories = row["categories"].split(",")
                    all_categories.extend(categories)
            
            # Return unique categories
            return sorted(list(set(all_categories)))
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            raise 