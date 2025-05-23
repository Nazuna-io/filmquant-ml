#!/bin/bash
# Setup script for film data enrichment

echo "=== FilmQuant ML Data Enrichment Setup ==="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this from the filmquant-ml project root directory"
    exit 1
fi

# Install required packages
echo "Installing Python dependencies..."
pip install python-dotenv beautifulsoup4 lxml

# Create data directories
echo "Creating data directories..."
mkdir -p data/mappings

# Check environment variables
echo ""
echo "=== Environment Setup ==="
if [ -f ".env" ]; then
    echo "Found .env file. Please ensure the following API keys are set:"
    echo "  - TMDB_API_KEY (required)"
    echo "  - KINOCHECK_API_KEY (optional, for trailer views)"
    echo "  - YOUTUBE_API_KEY (optional, fallback for trailer views)"
    echo ""
    echo "Current .env status:"
    if grep -q "TMDB_API_KEY=your_tmdb_api_key_here" .env; then
        echo "  ⚠ TMDB_API_KEY needs to be updated"
    else
        echo "  ✓ TMDB_API_KEY appears to be set"
    fi

    if grep -q "KINOCHECK_API_KEY=your_kinocheck_api_key_here" .env; then
        echo "  ⚠ KINOCHECK_API_KEY needs to be updated (optional)"
    else
        echo "  ✓ KINOCHECK_API_KEY appears to be set"
    fi

    if grep -q "YOUTUBE_API_KEY=your_youtube_api_key_here" .env; then
        echo "  ⚠ YOUTUBE_API_KEY needs to be updated (optional)"
    else
        echo "  ✓ YOUTUBE_API_KEY appears to be set"
    fi
else
    echo "Error: .env file not found"
    exit 1
fi

echo ""
echo "=== Data Files ==="
if [ -f "data/filmquant-ml-historical-data-2023.csv" ]; then
    FILM_COUNT=$(tail -n +2 data/filmquant-ml-historical-data-2023.csv | wc -l)
    echo "✓ Found historical dataset with $FILM_COUNT films"
else
    echo "✗ Historical dataset not found at data/filmquant-ml-historical-data-2023.csv"
fi

echo ""
echo "=== Quick Test ==="
echo "To test your API connections, run:"
echo "  python scripts/test_enrichment.py"
echo ""
echo "To start enriching your data:"
echo "  # Dry run first (no API calls)"
echo "  python scripts/enrich_historical_data.py --dry-run"
echo ""
echo "  # Process a single film for testing"
echo "  python scripts/enrich_historical_data.py --film-id 1"
echo ""
echo "  # Process all films"
echo "  python scripts/enrich_historical_data.py"

echo ""
echo "Setup complete!"
