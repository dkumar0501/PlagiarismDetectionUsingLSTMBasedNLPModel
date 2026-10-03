#!/bin/bash

echo "======================================"
echo "  VeriCheck - Plagiarism Detector"
echo "  IIT Patna Capstone Project"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.12+"
    exit 1
fi

echo "✓ Python found"

# Navigate to backend directory
cd "$(dirname "$0")/backend" || exit 1

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the server, run:"
echo "  python run.py"
echo ""
echo "The API will be available at: http://127.0.0.1:5000/api"
echo ""