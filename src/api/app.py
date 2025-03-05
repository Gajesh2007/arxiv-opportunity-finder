"""
Flask API for accessing opportunities from the database.
"""
import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.database.db import Database

app = Flask(
    __name__, 
    template_folder=os.path.join(project_root, "src", "ui", "templates"),
    static_folder=os.path.join(project_root, "src", "ui", "static")
)
CORS(app)  # Enable CORS for all routes

# Initialize database
db = Database()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/")
def index():
    """Render the main dashboard page."""
    return render_template("index.html")

@app.route("/opportunities")
def get_opportunities():
    """Get all opportunities from the database."""
    try:
        # Get query parameters
        min_score = float(request.args.get("min_score", 0.0))
        limit = int(request.args.get("limit", 100))
        sort_by = request.args.get("sort_by", "combined_score")
        sort_direction = request.args.get("sort_direction", "desc")
        category = request.args.get("category", None)
        
        # Get analyses with papers
        opportunities = db.get_analyses_with_papers(
            min_score=min_score,
            limit=limit,
            sort_by=sort_by,
            sort_direction=sort_direction,
            category=category
        )
        
        return jsonify({"opportunities": opportunities})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/opportunities/<paper_id>")
def get_opportunity(paper_id):
    """Get a specific opportunity."""
    try:
        # Get the paper and its analysis
        paper = db.get_paper(paper_id)
        analysis = db.get_analysis(paper_id)
        
        if not paper or not analysis:
            return jsonify({"error": "Opportunity not found"}), 404
        
        # Combine paper and analysis
        opportunity = {**paper, **analysis}
        
        # Extract market data and other fields from the openai_analysis
        if "openai_analysis" in opportunity:
            if isinstance(opportunity["openai_analysis"], str):
                try:
                    openai_data = json.loads(opportunity["openai_analysis"])
                except json.JSONDecodeError:
                    openai_data = {}
            else:
                openai_data = opportunity["openai_analysis"]
                
            # Add market data to the root level
            opportunity['time_to_market'] = openai_data.get('time_to_market', 'Unknown')
            opportunity['target_markets'] = openai_data.get('target_markets', [])
            opportunity['steps'] = openai_data.get('steps', [])
            opportunity['resources'] = openai_data.get('resources', [])
            opportunity['challenges'] = openai_data.get('challenges', [])
            opportunity['layman_explanation'] = openai_data.get('layman_explanation', '')
            
            # Set scores from OpenAI analysis if not available in the analysis table
            if 'technical_feasibility_score' in openai_data and not opportunity.get('technical_feasibility_score'):
                opportunity['technical_feasibility_score'] = openai_data.get('technical_feasibility_score', 0)
            if 'market_potential_score' in openai_data and not opportunity.get('market_potential_score'):
                opportunity['market_potential_score'] = openai_data.get('market_potential_score', 0)
            if 'impact_score' in openai_data and not opportunity.get('impact_score'):
                opportunity['impact_score'] = openai_data.get('impact_score', 0)
                
        # Map database field names to what the UI expects
        if 'poc_potential_score' in opportunity:
            opportunity['market_potential_score'] = opportunity.get('poc_potential_score')
        if 'wow_factor_score' in opportunity:
            opportunity['impact_score'] = opportunity.get('wow_factor_score')
        
        return jsonify({"opportunity": opportunity})
    except Exception as e:
        logger.error(f"Error in get_opportunity: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/categories")
def get_categories():
    """Get all unique categories from the database."""
    try:
        categories = db.get_categories()
        return jsonify({"categories": categories})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stats")
def get_stats():
    """Get statistics about the opportunities."""
    try:
        stats = db.get_stats()
        return jsonify({
            "total_papers": stats[0],
            "processed_papers": stats[1],
            "top_opportunities": stats[2]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/papers/<paper_id>/pdf")
def get_pdf(paper_id):
    """Serve a paper's PDF file."""
    try:
        pdf_dir = os.getenv("PAPERS_DIR", "data/pdfs")
        return send_from_directory(pdf_dir, f"{paper_id}.pdf")
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug) 