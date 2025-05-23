#!/usr/bin/env python3
"""
Basic tests for the data enrichment functionality
"""

import json
import os
import sys
import tempfile

import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Mock environment before importing
os.environ["TMDB_API_KEY"] = "test_token"
os.environ["KINOCHECK_API_KEY"] = "test_kinocheck"

from scripts.enrich_historical_data import DataEnricher


class TestBasicEnrichment:
    """Basic tests for the enrichment system"""

    def test_enricher_creation(self):
        """Test that DataEnricher can be created"""
        enricher = DataEnricher()
        assert enricher is not None
        assert hasattr(enricher, "mappings")
        assert "genres" in enricher.mappings
        assert "people" in enricher.mappings
        assert "studios" in enricher.mappings
        assert "tmdb_ids" in enricher.mappings

    def test_mapping_updates(self):
        """Test mapping update functionality"""
        enricher = DataEnricher()

        # Mock TMDB data
        tmdb_details = {
            "genres": [{"id": 18, "name": "Drama"}, {"id": 35, "name": "Comedy"}]
        }
        tmdb_credits = {
            "cast": [{"id": 2037, "name": "Cillian Murphy"}],
            "crew": [{"id": 525, "name": "Christopher Nolan", "job": "Director"}],
        }

        initial_genre_count = len(enricher.mappings["genres"])
        initial_people_count = len(enricher.mappings["people"])

        enricher._update_mappings_from_tmdb(tmdb_details, tmdb_credits)

        # Check that new entries were added
        assert len(enricher.mappings["genres"]) >= initial_genre_count + 2
        assert len(enricher.mappings["people"]) >= initial_people_count + 2

        # Check specific entries
        assert "Drama" in enricher.mappings["genres"]
        assert "Comedy" in enricher.mappings["genres"]
        assert "Cillian Murphy" in enricher.mappings["people"]
        assert "Christopher Nolan" in enricher.mappings["people"]

    def test_view_extraction(self):
        """Test trailer view extraction logic"""
        enricher = DataEnricher()

        # Test dict with direct views
        result = enricher._extract_views_from_kinocheck({"views": 1000000})
        assert result == 1000000

        # Test list with multiple items
        result = enricher._extract_views_from_kinocheck(
            [{"views": 500000}, {"views": 1500000}, {"views": 1000000}]
        )
        assert result == 1500000  # Should return the max

        # Test empty/invalid data
        result = enricher._extract_views_from_kinocheck({})
        assert result is None

        result = enricher._extract_views_from_kinocheck([])
        assert result is None

    def test_data_enrichment_logic(self):
        """Test the data enrichment logic"""
        enricher = DataEnricher()

        # Set up test mappings
        enricher.mappings["genres"] = {"Drama": 1, "Comedy": 2}
        enricher.mappings["people"] = {
            "Test Director": {"id": 100, "tmdb_id": 1, "role": "director"},
            "Test Actor": {"id": 101, "tmdb_id": 2, "role": "actor"},
        }
        enricher.mappings["studios"] = {"Test Studio": {"id": 50, "tmdb_id": 1}}

        # Test row with empty fields
        row = {
            "id": "1",
            "title": "Test Movie",
            "genre1_id": "",
            "director_id": "",
            "actor1_id": "",
            "studio_id": "",
            "runtime_minutes": "",
            "budget_usd": "",
        }

        # Mock TMDB data
        tmdb_data = {
            "tmdb_id": 12345,
            "details": {
                "runtime": 120,
                "budget": 50000000,
                "release_date": "2023-01-01",
                "genres": [{"name": "Drama"}],
                "production_companies": [{"name": "Test Studio"}],
            },
            "credits": {
                "cast": [{"name": "Test Actor"}],
                "crew": [{"name": "Test Director", "job": "Director"}],
            },
        }

        enriched_row = enricher._enrich_with_tmdb_data(row, tmdb_data)

        # Check that fields were filled
        assert enriched_row["genre1_id"] == 1
        assert enriched_row["director_id"] == 100
        assert enriched_row["actor1_id"] == 101
        assert enriched_row["studio_id"] == 50
        assert enriched_row["runtime_minutes"] == 120
        assert enriched_row["budget_usd"] == 50  # $50M from $50,000,000
        assert enriched_row["release_date"] == "2023-01-01"

    def test_preserves_existing_data(self):
        """Test that enrichment preserves existing data"""
        enricher = DataEnricher()

        # Row with some existing data
        row = {
            "id": "1",
            "title": "Test Movie",
            "runtime_minutes": "100",  # Existing value
            "genre1_id": "Drama",  # Existing value
            "budget_usd": "",  # Empty - should be filled
        }

        tmdb_data = {
            "tmdb_id": 12345,
            "details": {
                "runtime": 120,  # Different from existing
                "budget": 50000000,  # Should fill empty field
                "genres": [{"name": "Comedy"}],  # Different genre
            },
            "credits": {"cast": [], "crew": []},
        }

        enriched_row = enricher._enrich_with_tmdb_data(row, tmdb_data)

        # Check that existing values were preserved
        assert enriched_row["runtime_minutes"] == "100"  # Original kept
        assert enriched_row["genre1_id"] == "Drama"  # Original kept
        assert enriched_row["budget_usd"] == 50  # Empty field filled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
