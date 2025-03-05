-- schema.sql

-- Papers table to store metadata about arXiv papers
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,
    categories TEXT,
    abstract TEXT,
    pdf_path TEXT,
    scrape_date TEXT,
    processed INTEGER DEFAULT 0
);

-- Analyses table to store results of paper analysis
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY,
    paper_id TEXT,
    claude_analysis TEXT,
    openai_analysis TEXT,
    innovation_score REAL,
    poc_potential_score REAL,
    wow_factor_score REAL,
    implementation_complexity REAL,
    combined_score REAL,
    layman_explanation TEXT,
    processed_date TEXT,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS papers_processed_idx ON papers(processed);
CREATE INDEX IF NOT EXISTS analyses_paper_id_idx ON analyses(paper_id);
CREATE INDEX IF NOT EXISTS analyses_combined_score_idx ON analyses(combined_score); 