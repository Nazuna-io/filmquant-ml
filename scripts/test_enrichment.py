#!/usr/bin/env python3
"""
Test script for the data enrichment pipeline
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.enrich_historical_data import DataEnricher


def test_api_connections():
    """Test if API keys are working"""
    print("Testing API connections...")

    enricher = DataEnricher()

    # Test with a known film
    test_title = "Oppenheimer"
    test_year = 2023

    print(f"\nTesting TMDB with: {test_title} ({test_year})")
    tmdb_data = enricher.get_tmdb_data(test_title, test_year)

    if tmdb_data:
        print(f"✓ TMDB connection successful")
        print(f"  Title: {tmdb_data['details']['title']}")
        print(f"  Budget: ${tmdb_data['details']['budget']:,}")
        print(f"  Runtime: {tmdb_data['details']['runtime']} minutes")
        print(f"  Genres: {[g['name'] for g in tmdb_data['details']['genres']]}")

        # Test trailer views
        print(f"\nTesting trailer views...")
        views = enricher.get_trailer_views(tmdb_data["tmdb_id"], test_title)
        if views:
            print(f"✓ Trailer views found: {views:,}")
        else:
            print(f"⚠ No trailer views found")
    else:
        print(f"✗ TMDB connection failed")

    print(f"\nMappings summary:")
    for mapping_type, mapping_data in enricher.mappings.items():
        print(f"  {mapping_type}: {len(mapping_data)} entries")


if __name__ == "__main__":
    test_api_connections()
