#!/bin/bash

echo "======================================"
echo "  VeriCheck - Starting Server..."
echo "======================================"

cd "$(dirname "$0")/backend" || exit 1

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start Flask server
echo ""
echo "🚀 Starting Flask backend on http://127.0.0.1:5000"
echo "📂 Frontend: Open frontend/index.html in your browser"
echo ""
python run.py