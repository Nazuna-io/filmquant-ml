#!/usr/bin/env python3
"""
Unit tests for the data enrichment pipeline
"""

import pytest
import os
import sys
import json
import tempfile
import csv
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.enrich_historical_data import DataEnricher


class TestDataEnricher:
    """Test suite for DataEnricher class"""
    
    @pytest.fixture
    def enricher(self):
        """Create a DataEnricher instance for testing"""
        with patch.dict(os.environ, {
            'TMDB_API_KEY': 'test_bearer_token',
            'KINOCHECK_API_KEY': 'test_kinocheck_key',
            'YOUTUBE_API_KEY': 'test_youtube_key'
        }):
            return DataEnricher()
    
    @pytest.fixture
    def sample_tmdb_response(self):
        """Sample TMDB API response for testing"""
        return {
            'id': 872585,
            'title': 'Oppenheimer',
            'release_date': '2023-07-19',
            'runtime': 181,
            'budget': 100000000,
            'genres': [
                {'id': 18, 'name': 'Drama'},
                {'id': 36, 'name': 'History'}
            ],
            'production_companies': [
                {'id': 33, 'name': 'Universal Pictures', 'logo_path': '/path.png'}
            ]
        }
    
    @pytest.fixture
    def sample_credits_response(self):
        """Sample TMDB credits response for testing"""
        return {
            'cast': [
                {'id': 2037, 'name': 'Cillian Murphy', 'character': 'J. Robert Oppenheimer'},
                {'id': 5081, 'name': 'Emily Blunt', 'character': 'Kitty Oppenheimer'},
                {'id': 1892, 'name': 'Matt Damon', 'character': 'Leslie Groves'}
            ],
            'crew': [
                {'id': 525, 'name': 'Christopher Nolan', 'job': 'Director'},
                {'id': 525, 'name': 'Christopher Nolan', 'job': 'Producer'}
            ]
        }

    def test_initialization(self, enricher):
        """Test DataEnricher initialization"""
        assert enricher.tmdb_headers == {'Authorization': 'Bearer test_bearer_token'}
        assert 'X-Api-Key' in enricher.kinocheck_headers
        assert enricher.kinocheck_headers['X-Api-Key'] == 'test_kinocheck_key'
        assert 'genres' in enricher.mappings
        assert 'people' in enricher.mappings
        assert 'studios' in enricher.mappings
        assert 'tmdb_ids' in enricher.mappings

    @patch('scripts.enrich_historical_data.requests.get')
    def test_search_tmdb_film_success(self, mock_get, enricher):
        """Test successful TMDB film search"""
        # Mock the search response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'results': [
                {'id': 872585, 'title': 'Oppenheimer', 'release_date': '2023-07-19'}
            ]
        }
        mock_get.return_value = mock_response
        
        result = enricher._search_tmdb_film('Oppenheimer', 2023)
        
        assert result == 872585
        mock_get.assert_called_once()
        
    @patch('scripts.enrich_historical_data.requests.get')
    def test_search_tmdb_film_no_results(self, mock_get, enricher):
        """Test TMDB film search with no results"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {'results': []}
        mock_get.return_value = mock_response
        
        result = enricher._search_tmdb_film('Nonexistent Movie', 2023)
        
        assert result is None

    @patch('scripts.enrich_historical_data.requests.get')
    def test_get_tmdb_details_success(self, mock_get, enricher, sample_tmdb_response, sample_credits_response):
        """Test successful TMDB details retrieval"""
        # Mock two API calls - details and credits
        mock_responses = [
            Mock(status_code=200, json=lambda: sample_tmdb_response),
            Mock(status_code=200, json=lambda: sample_credits_response)
        ]
        for resp in mock_responses:
            resp.raise_for_status.return_value = None
        mock_get.side_effect = mock_responses
        
        result = enricher._get_tmdb_details(872585)
        
        assert result['tmdb_id'] == 872585
        assert result['details']['title'] == 'Oppenheimer'
        assert result['credits']['cast'][0]['name'] == 'Cillian Murphy'
        assert mock_get.call_count == 2

    def test_update_mappings_from_tmdb(self, enricher, sample_tmdb_response, sample_credits_response):
        """Test mapping updates from TMDB data"""
        initial_genre_count = len(enricher.mappings['genres'])
        initial_people_count = len(enricher.mappings['people'])
        initial_studio_count = len(enricher.mappings['studios'])
        
        enricher._update_mappings_from_tmdb(sample_tmdb_response, sample_credits_response)
        
        # Check that genres were added
        assert 'Drama' in enricher.mappings['genres']
        assert 'History' in enricher.mappings['genres']
        assert len(enricher.mappings['genres']) >= initial_genre_count + 2
        
        # Check that people were added
        assert 'Cillian Murphy' in enricher.mappings['people']
        assert 'Christopher Nolan' in enricher.mappings['people']
        assert enricher.mappings['people']['Cillian Murphy']['role'] == 'actor'
        assert enricher.mappings['people']['Christopher Nolan']['role'] == 'director'
        
        # Check that studios were added
        assert 'Universal Pictures' in enricher.mappings['studios']

    @patch('scripts.enrich_historical_data.requests.get')
    def test_get_kinocheck_views_success(self, mock_get, enricher):
        """Test successful Kinocheck trailer views retrieval"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'views': 5000000}
        mock_get.return_value = mock_response
        
        result = enricher._get_kinocheck_views(872585, 'Oppenheimer')
        
        assert result == 5000000

    def test_extract_views_from_kinocheck_dict(self, enricher):
        """Test view extraction from Kinocheck response (dict format)"""
        # Test direct views in dict
        result = enricher._extract_views_from_kinocheck({'views': 1000000})
        assert result == 1000000
        
        # Test views in trailer object
        result = enricher._extract_views_from_kinocheck({
            'trailer': {'views': 2000000}
        })
        assert result == 2000000
        
        # Test views in videos array
        result = enricher._extract_views_from_kinocheck({
            'videos': [
                {'views': 500000},
                {'views': 1500000},
                {'views': 1000000}
            ]
        })
        assert result == 1500000  # Should return the max

    def test_extract_views_from_kinocheck_list(self, enricher):
        """Test view extraction from Kinocheck response (list format)"""
        result = enricher._extract_views_from_kinocheck([
            {'views': 100000},
            {'views': 300000},
            {'views': 200000}
        ])
        assert result == 300000  # Should return the max

    def test_enrich_with_tmdb_data(self, enricher, sample_tmdb_response, sample_credits_response):
        """Test data enrichment with TMDB data"""
        # Set up mappings
        enricher.mappings['genres'] = {'Drama': 1, 'History': 2}
        enricher.mappings['people'] = {
            'Christopher Nolan': {'id': 100, 'tmdb_id': 525, 'role': 'director'},
            'Cillian Murphy': {'id': 101, 'tmdb_id': 2037, 'role': 'actor'},
            'Emily Blunt': {'id': 102, 'tmdb_id': 5081, 'role': 'actor'}
        }
        enricher.mappings['studios'] = {
            'Universal Pictures': {'id': 50, 'tmdb_id': 33}
        }
        
        # Test row with missing data
        row = {
            'id': '1',
            'title': 'Oppenheimer',
            'genre1_id': '',
            'genre2_id': '',
            'director_id': '',
            'actor1_id': '',
            'actor2_id': '',
            'studio_id': '',
            'runtime_minutes': '',
            'budget_usd': '',
            'release_date': ''
        }
        
        tmdb_data = {
            'tmdb_id': 872585,
            'details': sample_tmdb_response,
            'credits': sample_credits_response
        }
        
        enriched_row = enricher._enrich_with_tmdb_data(row, tmdb_data)
        
        # Check that fields were filled
        assert enriched_row['genre1_id'] == 1  # Drama
        assert enriched_row['genre2_id'] == 2  # History
        assert enriched_row['director_id'] == 100  # Christopher Nolan
        assert enriched_row['actor1_id'] == 101  # Cillian Murphy
        assert enriched_row['actor2_id'] == 102  # Emily Blunt
        assert enriched_row['studio_id'] == 50  # Universal Pictures
        assert enriched_row['runtime_minutes'] == 181
        assert enriched_row['budget_usd'] == 100  # $100M (converted from $100,000,000)
        assert enriched_row['release_date'] == '2023-07-19'

    def test_enrich_with_tmdb_data_preserves_existing(self, enricher, sample_tmdb_response, sample_credits_response):
        """Test that enrichment preserves existing data"""
        row = {
            'id': '1',
            'title': 'Oppenheimer',
            'runtime_minutes': '180',  # Existing value (slightly different from TMDB)
            'budget_usd': '95',        # Existing value
            'genre1_id': 'Drama'       # Existing value
        }
        
        tmdb_data = {
            'tmdb_id': 872585,
            'details': sample_tmdb_response,
            'credits': sample_credits_response
        }
        
        enriched_row = enricher._enrich_with_tmdb_data(row, tmdb_data)
        
        # Check that existing values were preserved
        assert enriched_row['runtime_minutes'] == '180'  # Original value kept
        assert enriched_row['budget_usd'] == '95'        # Original value kept
        assert enriched_row['genre1_id'] == 'Drama'      # Original value kept


class TestDataEnrichmentIntegration:
    """Integration tests for the complete enrichment pipeline"""
    
    def test_csv_processing_workflow(self):
        """Test the complete CSV processing workflow"""
        # Create temporary CSV file
        test_data = [
            ['id', 'title', 'genre1_id', 'director_id', 'runtime_minutes', 'budget_usd'],
            ['1', 'Test Movie', '', '', '', '']
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file:
            writer = csv.writer(input_file)
            writer.writerows(test_data)
            input_csv = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
            output_csv = output_file.name
        
        try:
            # Mock environment variables
            with patch.dict(os.environ, {
                'TMDB_API_KEY': 'test_bearer_token'
            }):
                enricher = DataEnricher()
                
                # Mock the TMDB API calls
                with patch.object(enricher, 'get_tmdb_data') as mock_tmdb:
                    mock_tmdb.return_value = None  # Simulate no data found
                    
                    # Run enrichment
                    success = enricher.run_enrichment(
                        input_csv, 
                        output_csv, 
                        dry_run=False, 
                        target_film_id='1'
                    )
                    
                    assert success
                    assert enricher.stats['processed'] == 1
                    
                    # Check output file exists
                    assert os.path.exists(output_csv)
                    
        finally:
            # Cleanup
            if os.path.exists(input_csv):
                os.unlink(input_csv)
            if os.path.exists(output_csv):
                os.unlink(output_csv)

    def test_mappings_persistence(self):
        """Test that mappings are saved and loaded correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the mappings directory
            with patch('scripts.enrich_historical_data.MAPPINGS_DIR', temp_dir):
                enricher = DataEnricher()
                
                # Add some test mappings
                enricher.mappings['genres']['Action'] = 1
                enricher.mappings['people']['Test Actor'] = {'id': 1, 'role': 'actor'}
                
                # Save mappings
                enricher.save_mappings()
                
                # Create new enricher to test loading
                enricher2 = DataEnricher()
                
                # Check that mappings were loaded
                assert 'Action' in enricher2.mappings['genres']
                assert 'Test Actor' in enricher2.mappings['people']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
