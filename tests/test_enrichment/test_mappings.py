#!/usr/bin/env python3
"""
Tests for ID mapping functionality in the enrichment pipeline
"""

import pytest
import os
import sys
import tempfile
import json

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.enrich_historical_data import DataEnricher


class TestMappingFunctionality:
    """Test the ID mapping system"""
    
    def test_genre_mapping_creation(self):
        """Test genre mapping creation and consistency"""
        enricher = DataEnricher()
        
        # Test genres from TMDB data
        tmdb_details = {
            'genres': [
                {'id': 18, 'name': 'Drama'},
                {'id': 35, 'name': 'Comedy'},
                {'id': 28, 'name': 'Action'}
            ]
        }
        tmdb_credits = {'cast': [], 'crew': []}
        
        enricher._update_mappings_from_tmdb(tmdb_details, tmdb_credits)
        
        # Check that genres were mapped with sequential IDs
        assert 'Drama' in enricher.mappings['genres']
        assert 'Comedy' in enricher.mappings['genres']
        assert 'Action' in enricher.mappings['genres']
        
        # Check that IDs are sequential
        drama_id = enricher.mappings['genres']['Drama']
        comedy_id = enricher.mappings['genres']['Comedy']
        action_id = enricher.mappings['genres']['Action']
        
        assert isinstance(drama_id, int)
        assert isinstance(comedy_id, int) 
        assert isinstance(action_id, int)
        
        # Test adding the same genre again doesn't change the ID
        enricher._update_mappings_from_tmdb(tmdb_details, tmdb_credits)
        assert enricher.mappings['genres']['Drama'] == drama_id

    def test_people_mapping_with_roles(self):
        """Test people mapping with different roles"""
        enricher = DataEnricher()
        
        tmdb_details = {'genres': []}
        tmdb_credits = {
            'cast': [
                {'id': 2037, 'name': 'Cillian Murphy'},
                {'id': 5081, 'name': 'Emily Blunt'}
            ],
            'crew': [
                {'id': 525, 'name': 'Christopher Nolan', 'job': 'Director'},
                {'id': 1234, 'name': 'Hans Zimmer', 'job': 'Composer'},  # Not tracked
                {'id': 5678, 'name': 'Emma Thomas', 'job': 'Producer'}
            ]
        }
        
        enricher._update_mappings_from_tmdb(tmdb_details, tmdb_credits)
        
        # Check actors were added
        assert 'Cillian Murphy' in enricher.mappings['people']
        assert enricher.mappings['people']['Cillian Murphy']['role'] == 'actor'
        assert enricher.mappings['people']['Cillian Murphy']['tmdb_id'] == 2037
        
        # Check director was added
        assert 'Christopher Nolan' in enricher.mappings['people']
        assert enricher.mappings['people']['Christopher Nolan']['role'] == 'director'
        
        # Check producer was added
        assert 'Emma Thomas' in enricher.mappings['people']
        assert enricher.mappings['people']['Emma Thomas']['role'] == 'producer'
        
        # Check composer was NOT added (only Director/Producer tracked from crew)
        assert 'Hans Zimmer' not in enricher.mappings['people']

    def test_mapping_consistency_across_updates(self):
        """Test that mappings remain consistent across multiple updates"""
        enricher = DataEnricher()
        
        # First update
        tmdb_details1 = {
            'genres': [{'name': 'Drama'}, {'name': 'Comedy'}]
        }
        tmdb_credits1 = {
            'cast': [{'id': 1, 'name': 'Actor One'}],
            'crew': [{'id': 2, 'name': 'Director One', 'job': 'Director'}]
        }
        
        enricher._update_mappings_from_tmdb(tmdb_details1, tmdb_credits1)
        
        # Store initial IDs
        initial_drama_id = enricher.mappings['genres']['Drama']
        initial_actor_id = enricher.mappings['people']['Actor One']['id']
        
        # Second update with overlapping data
        tmdb_details2 = {
            'genres': [{'name': 'Drama'}, {'name': 'Action'}]  # Drama repeats
        }
        tmdb_credits2 = {
            'cast': [{'id': 1, 'name': 'Actor One'}, {'id': 3, 'name': 'Actor Two'}],  # Actor One repeats
            'crew': [{'id': 4, 'name': 'Director Two', 'job': 'Director'}]
        }
        
        enricher._update_mappings_from_tmdb(tmdb_details2, tmdb_credits2)
        
        # Check that repeated items kept their original IDs
        assert enricher.mappings['genres']['Drama'] == initial_drama_id
        assert enricher.mappings['people']['Actor One']['id'] == initial_actor_id
        
        # Check that new items got new IDs
        assert 'Action' in enricher.mappings['genres']
        assert 'Actor Two' in enricher.mappings['people']
        assert 'Director Two' in enricher.mappings['people']
        
        # Check IDs are still unique
        all_genre_ids = list(enricher.mappings['genres'].values())
        assert len(all_genre_ids) == len(set(all_genre_ids))  # No duplicates
        
        all_people_ids = [p['id'] for p in enricher.mappings['people'].values()]
        assert len(all_people_ids) == len(set(all_people_ids))  # No duplicates


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
