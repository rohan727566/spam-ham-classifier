#!/bin/bash
# Launch FastAPI server

set -e

echo "🚀 Starting Spam Classifier API Server..."
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run server
python -m src.spam_classifier.server

# Alternative with uvicorn directly:
# uvicorn src.spam_classifier.server:app --host 0.0.0.0 --port 8000 --reload
