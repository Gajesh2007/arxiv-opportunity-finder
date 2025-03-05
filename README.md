# arXiv Opportunity Finder

A system that automatically discovers high-potential research papers from arXiv and evaluates them for proof-of-concept (POC) implementation. The system processes papers in parallel using Claude 3.7 Sonnet for direct PDF analysis and OpenAI o1 for implementation planning, with results stored in SQLite for fast iteration and easy access.

## Features

- **Automated Paper Discovery**: Scrapes the latest papers from arXiv in specified categories
- **Intelligent Analysis**: Uses Claude 3.7 Sonnet to analyze PDFs directly and evaluate innovation potential
- **Implementation Planning**: Uses OpenAI o1 to create detailed proof-of-concept implementation plans
- **Scoring System**: Evaluates papers on innovation, POC potential, wow factor, and implementation complexity
- **Parallel Processing**: Processes multiple papers concurrently for efficiency
- **Persistent Storage**: Stores all data in SQLite for easy querying and analysis

## Architecture

The system consists of several components:

1. **Scraper**: Retrieves papers from arXiv and downloads PDFs
2. **Analyzer**: Processes papers with Claude and OpenAI to evaluate and plan implementations
3. **Database**: Stores paper metadata, analysis results, and implementation plans
4. **Pipeline**: Orchestrates the entire process from scraping to analysis

## Installation

### Prerequisites

- Python 3.10+
- API keys for Anthropic Claude and OpenAI
- Poetry (recommended) or pip

### Setup with Poetry (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/Gajesh2007/arxiv-opportunity-finder.git
   cd arxiv-opportunity-finder
   ```

2. Install Poetry if you don't have it already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

4. Create a `.env` file with your API keys (see `.env.template` for format):
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

5. Initialize the database:
   ```bash
   poetry run python -m src.database.init_db
   ```

### Setup with Pip (Alternative)

1. Clone the repository:
   ```bash
   git clone https://github.com/Gajesh2007/arxiv-opportunity-finder.git
   cd arxiv-opportunity-finder
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys (see `.env.template` for format):
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

4. Initialize the database:
   ```bash
   python -m src.database.init_db
   ```

## Usage

### Running the Full Pipeline

To run the complete pipeline (scrape, analyze, store) with Poetry:

```bash
poetry run python -m src.pipeline --categories cs.AI,cs.LG --scrape-limit 50 --analyze-limit 10
```

Or with pip:

```bash
python -m src.pipeline --categories cs.AI,cs.LG --scrape-limit 50 --analyze-limit 10
```

### Running Individual Components

To scrape papers only (with Poetry):

```bash
poetry run python -m src.scraper.main --categories cs.AI,cs.LG --limit 50
```

To analyze papers only (with Poetry):

```bash
poetry run python -m src.analyzer.main --limit 10
```

### Command Line Options

- `--categories`: Comma-separated list of arXiv categories to scrape (default: cs.AI,cs.LG,cs.CV,cs.CL,cs.NE)
- `--scrape-limit`: Maximum number of papers to scrape (default: 100)
- `--analyze-limit`: Maximum number of papers to analyze (default: 10)
- `--max-workers`: Maximum number of concurrent workers (default: 5)
- `--no-download`: Do not download PDFs during scraping
- `--daily`: Run in daily mode (uses smaller limits suitable for daily updates)

## Configuration

Configuration is managed through environment variables in the `.env` file:

- `ANTHROPIC_API_KEY`: API key for Anthropic Claude
- `OPENAI_API_KEY`: API key for OpenAI
- `DATABASE_PATH`: Path to the SQLite database (default: data/database.sqlite)
- `PAPERS_DIR`: Directory to store downloaded PDFs (default: data/pdfs)
- `PROCESSED_DIR`: Directory to store processed results (default: data/processed)
- `ARXIV_MAX_RESULTS`: Maximum number of results to retrieve from arXiv (default: 100)
- `ARXIV_WAIT_TIME`: Time to wait between arXiv API calls (default: 3 seconds)
- `BATCH_SIZE`: Number of papers to process in a batch (default: 10)
- `MAX_WORKERS`: Maximum number of concurrent workers (default: 5)

## Project Structure

```
arxiv-opportunity-finder/
├── data/                  # Data storage
│   ├── pdfs/              # Downloaded PDFs
│   ├── processed/         # Processed results
│   └── logs/              # Log files
├── src/                   # Source code
│   ├── scraper/           # Paper scraping module
│   ├── analyzer/          # Paper analysis module
│   ├── database/          # Database utilities
│   ├── api/               # API for web interface
│   ├── ui/                # Web interface
│   └── utils/             # Utility functions
├── scripts/               # Automation scripts
├── .env.template          # Template for environment variables
├── pyproject.toml         # Poetry configuration and dependencies
├── requirements.txt       # Python dependencies (for pip)
└── README.md              # This file
```

## License

MIT

## Acknowledgements

- [arXiv](https://arxiv.org/) for providing access to research papers
- [Anthropic Claude](https://www.anthropic.com/) for the Claude API
- [OpenAI](https://openai.com/) for the OpenAI API 