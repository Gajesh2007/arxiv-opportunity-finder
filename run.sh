#!/bin/bash

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH for this script
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if data directories exist
if [ ! -d "data" ]; then
    mkdir -p data/pdfs
    mkdir -p data/json
    echo "Created data directories"
fi

# Check if logs directory exists
if [ ! -d "logs" ]; then
    mkdir -p logs
    echo "Created logs directory"
fi

# Function to run Python scripts with the correct environment
function run_script() {
    echo "Running: $@"
    python "$@"
}

# Function to run the web server
function run_server() {
    echo "Starting web server..."
    python src/api/main.py "$@"
}

# Function to run the enhanced pipeline
function run_enhanced_pipeline() {
    echo "Running enhanced pipeline..."
    python scripts/run_enhanced_pipeline.py "$@"
}

# Parse command-line arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 [run|server|pipeline] [args...]"
    exit 1
fi

command="$1"
shift

case "$command" in
    run)
        run_script "$@"
        ;;
    server)
        run_server "$@"
        ;;
    pipeline)
        run_enhanced_pipeline "$@"
        ;;
    "install")
        echo "Installing dependencies..."
        poetry install --no-root
        ;;
    "shell")
        echo "Starting Poetry shell..."
        poetry shell
        ;;
    "update")
        echo "Updating dependencies..."
        poetry update
        ;;
    "requirements")
        echo "Generating requirements.txt..."
        poetry run pip freeze > requirements.txt
        ;;
    "streaming")
        echo "Running streaming pipeline with OpenAI-only mode (Claude is bypassed)..."
        python run_streaming_pipeline.py "$@"
        ;;
    *)
        echo "Unknown command: $command"
        echo "Usage: $0 [run|server|pipeline|streaming] [args...]"
        exit 1
        ;;
esac 