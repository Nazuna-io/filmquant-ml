#!/usr/bin/env python3
"""
Film Data Enrichment Script for FilmQuant ML

This script enriches the historical dataset by:
1. Fetching data from TMDB API (genres, cast, crew, budget, runtime, etc.)
2. Getting trailer views from Kinocheck API with YouTube fallback
3. Scraping opening day screen counts from Box Office Mojo
4. Creating consistent ID mappings for all entities

Usage:
    python scripts/enrich_historical_data.py [--dry-run] [--film-id ID]
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
from urllib.parse import quote_plus
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR, "filmquant-ml-historical-data-2023.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "filmquant-ml-historical-data-2023-enriched.csv")
MAPPINGS_DIR = os.path.join(DATA_DIR, "mappings")

# API Keys
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
KINOCHECK_API_KEY = os.getenv('KINOCHECK_API_KEY')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Rate limiting
TMDB_RATE_LIMIT = 0.25  # 4 requests per second
KINOCHECK_RATE_LIMIT = 1.0  # 1 request per second
SCRAPING_RATE_LIMIT = 2.0  # 0.5 requests per second for scraping

class DataEnricher:
    def __init__(self):
        self.tmdb_headers = {'Authorization': f'Bearer {TMDB_API_KEY}'} if TMDB_API_KEY else {}
        self.kinocheck_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-Api-Key': KINOCHECK_API_KEY,
            'X-Api-Host': 'api.kinocheck.com'
        } if KINOCHECK_API_KEY else {}
        
        # Initialize mapping storage
        self.mappings = {
            'genres': {},
            'people': {},  # directors, actors
            'studios': {},
            'tmdb_ids': {}
        }
        
        # Load existing mappings
        self.load_mappings()
        
        # Stats tracking
        self.stats = {
            'processed': 0,
            'tmdb_success': 0,
            'trailer_success': 0,
            'screens_success': 0,
            'errors': []
        }

    def load_mappings(self):
        """Load existing ID mappings from JSON files"""
        os.makedirs(MAPPINGS_DIR, exist_ok=True)
        
        for mapping_type in self.mappings.keys():
            mapping_file = os.path.join(MAPPINGS_DIR, f"{mapping_type}.json")
            if os.path.exists(mapping_file):
                try:
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        self.mappings[mapping_type] = json.load(f)
                    print(f"Loaded {len(self.mappings[mapping_type])} {mapping_type} mappings")
                except Exception as e:
                    print(f"Error loading {mapping_type} mappings: {e}")

    def save_mappings(self):
        """Save ID mappings to JSON files"""
        for mapping_type, mapping_data in self.mappings.items():
            mapping_file = os.path.join(MAPPINGS_DIR, f"{mapping_type}.json")
            try:
                with open(mapping_file, 'w', encoding='utf-8') as f:
                    json.dump(mapping_data, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(mapping_data)} {mapping_type} mappings")
            except Exception as e:
                print(f"Error saving {mapping_type} mappings: {e}")

    def get_tmdb_data(self, title: str, year: int = None) -> Optional[Dict]:
        """Get comprehensive film data from TMDB API"""
        # Check if we already have the TMDB ID
        cache_key = f"{title}_{year}" if year else title
        if cache_key in self.mappings['tmdb_ids']:
            tmdb_id = self.mappings['tmdb_ids'][cache_key]
            print(f"  Using cached TMDB ID: {tmdb_id}")
        else:
            # Search for the film
            tmdb_id = self._search_tmdb_film(title, year)
            if not tmdb_id:
                return None
            self.mappings['tmdb_ids'][cache_key] = tmdb_id

        # Get detailed film data
        return self._get_tmdb_details(tmdb_id)

    def _search_tmdb_film(self, title: str, year: int = None) -> Optional[int]:
        """Search TMDB for a film and return its ID"""
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            'query': title,
            'language': 'en-US'
        }
        if year:
            params['year'] = year

        try:
            response = requests.get(search_url, headers=self.tmdb_headers, params=params)
            response.raise_for_status()
            time.sleep(TMDB_RATE_LIMIT)
            
            data = response.json()
            if data.get('results'):
                # Get the best match
                results = data['results']
                print(f"  Found {len(results)} TMDB matches for '{title}'")
                
                # Look for exact title match first
                for result in results:
                    if result['title'].lower() == title.lower():
                        release_year = result.get('release_date', '')[:4]
                        if not year or release_year == str(year):
                            print(f"  ✓ Exact match: {result['title']} ({release_year})")
                            return result['id']
                
                # Fall back to first result
                first_result = results[0]
                release_year = first_result.get('release_date', '')[:4]
                print(f"  ⚠ Using best match: {first_result['title']} ({release_year})")
                return first_result['id']
            
            print(f"  ✗ No TMDB results for '{title}'")
            return None
            
        except Exception as e:
            print(f"  ✗ TMDB search error: {e}")
            return None

    def _get_tmdb_details(self, tmdb_id: int) -> Optional[Dict]:
        """Get detailed film data from TMDB"""
        details_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        credits_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/credits"
        
        try:
            # Get main details
            details_response = requests.get(details_url, headers=self.tmdb_headers, 
                                          params={'language': 'en-US'})
            details_response.raise_for_status()
            time.sleep(TMDB_RATE_LIMIT)
            
            # Get credits
            credits_response = requests.get(credits_url, headers=self.tmdb_headers)
            credits_response.raise_for_status()
            time.sleep(TMDB_RATE_LIMIT)
            
            details = details_response.json()
            credits = credits_response.json()
            
            # Combine data
            film_data = {
                'tmdb_id': tmdb_id,
                'details': details,
                'credits': credits
            }
            
            # Update mappings
            self._update_mappings_from_tmdb(details, credits)
            
            return film_data
            
        except Exception as e:
            print(f"  ✗ TMDB details error: {e}")
            return None

    def _update_mappings_from_tmdb(self, details: Dict, credits: Dict):
        """Update ID mappings from TMDB data"""
        # Update genre mappings
        for genre in details.get('genres', []):
            genre_name = genre['name']
            if genre_name not in self.mappings['genres']:
                self.mappings['genres'][genre_name] = len(self.mappings['genres']) + 1

        # Update people mappings (cast and crew)
        for person in credits.get('cast', [])[:10]:  # Top 10 cast
            name = person['name']
            if name not in self.mappings['people']:
                self.mappings['people'][name] = {
                    'id': len(self.mappings['people']) + 1,
                    'tmdb_id': person['id'],
                    'role': 'actor'
                }

        for person in credits.get('crew', []):
            if person['job'] in ['Director', 'Producer', 'Executive Producer']:
                name = person['name']
                if name not in self.mappings['people']:
                    self.mappings['people'][name] = {
                        'id': len(self.mappings['people']) + 1,
                        'tmdb_id': person['id'],
                        'role': person['job'].lower()
                    }

        # Update studio mappings
        for company in details.get('production_companies', []):
            studio_name = company['name']
            if studio_name not in self.mappings['studios']:
                self.mappings['studios'][studio_name] = {
                    'id': len(self.mappings['studios']) + 1,
                    'tmdb_id': company['id']
                }

    def get_trailer_views(self, tmdb_id: int, title: str) -> Optional[int]:
        """Get trailer views using Kinocheck API with YouTube fallback"""
        # First try Kinocheck
        views = self._get_kinocheck_views(tmdb_id, title)
        if views:
            return views
        
        # Fallback to YouTube if available
        if YOUTUBE_API_KEY:
            return self._get_youtube_views(title)
        
        return None

    def _get_kinocheck_views(self, tmdb_id: int, title: str) -> Optional[int]:
        """Get trailer views from Kinocheck API"""
        if not KINOCHECK_API_KEY:
            return None
            
        endpoints = [
            f"https://api.kinocheck.com/trailers?tmdb_id={tmdb_id}",
            f"https://api.kinocheck.com/movies?tmdb_id={tmdb_id}",
            f"https://api.kinocheck.com/trailers?title={quote_plus(title)}"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, headers=self.kinocheck_headers)
                time.sleep(KINOCHECK_RATE_LIMIT)
                
                if response.status_code == 200:
                    data = response.json()
                    views = self._extract_views_from_kinocheck(data)
                    if views:
                        print(f"  ✓ Found {views:,} trailer views via Kinocheck")
                        return views
                        
            except Exception as e:
                print(f"  ⚠ Kinocheck error for {endpoint}: {e}")
                continue
        
        return None

    def _extract_views_from_kinocheck(self, data: Any) -> Optional[int]:
        """Extract view count from Kinocheck API response"""
        try:
            if isinstance(data, dict):
                if 'views' in data:
                    return int(data['views'])
                if 'trailer' in data and isinstance(data['trailer'], dict):
                    return int(data['trailer'].get('views', 0))
                if 'videos' in data:
                    max_views = 0
                    for video in data['videos']:
                        if isinstance(video, dict) and 'views' in video:
                            max_views = max(max_views, int(video['views']))
                    return max_views if max_views > 0 else None
            
            elif isinstance(data, list):
                max_views = 0
                for item in data:
                    if isinstance(item, dict) and 'views' in item:
                        max_views = max(max_views, int(item['views']))
                return max_views if max_views > 0 else None
                
        except (ValueError, TypeError):
            pass
        
        return None

    def _get_youtube_views(self, title: str) -> Optional[int]:
        """Get trailer views from YouTube API"""
        try:
            # Search for official trailer
            search_url = "https://www.googleapis.com/youtube/v3/search"
            search_params = {
                'part': 'snippet',
                'q': f"{title} official trailer",
                'type': 'video',
                'maxResults': 5,
                'key': YOUTUBE_API_KEY
            }
            
            search_response = requests.get(search_url, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            if not search_data.get('items'):
                return None
            
            # Get video statistics for the first result
            video_id = search_data['items'][0]['id']['videoId']
            stats_url = "https://www.googleapis.com/youtube/v3/videos"
            stats_params = {
                'part': 'statistics',
                'id': video_id,
                'key': YOUTUBE_API_KEY
            }
            
            stats_response = requests.get(stats_url, params=stats_params)
            stats_response.raise_for_status()
            stats_data = stats_response.json()
            
            if stats_data.get('items'):
                view_count = stats_data['items'][0]['statistics'].get('viewCount')
                if view_count:
                    views = int(view_count)
                    print(f"  ✓ Found {views:,} trailer views via YouTube")
                    return views
            
        except Exception as e:
            print(f"  ⚠ YouTube API error: {e}")
        
        return None

    def get_opening_screens(self, title: str, year: int = None) -> Optional[int]:
        """Scrape opening day screen count from Box Office Mojo"""
        # This is a simplified version - full implementation would require
        # more sophisticated web scraping
        try:
            # Construct search URL for Box Office Mojo
            search_term = title.replace(' ', '+')
            if year:
                search_term += f"+{year}"
            
            # Note: This is a placeholder - actual implementation would need
            # to handle Box Office Mojo's structure and potential anti-scraping measures
            print(f"  ⚠ Opening screens scraping not implemented yet")
            return None
            
        except Exception as e:
            print(f"  ✗ Screen count scraping error: {e}")
            return None

    def process_film(self, row: Dict, dry_run: bool = False) -> Dict:
        """Process a single film row and enrich it with additional data"""
        film_id = row['id']
        title = row['title']
        year = None
        
        # Extract year from release_date if available
        if row.get('release_date'):
            try:
                year = int(row['release_date'][:4])
            except (ValueError, TypeError):
                pass
        
        print(f"\nProcessing Film {film_id}: {title} ({year or 'Unknown year'})")
        
        enriched_row = row.copy()
        
        if dry_run:
            print("  [DRY RUN] - Would fetch TMDB data, trailer views, and screen counts")
            return enriched_row
        
        # 1. Get TMDB data
        tmdb_data = self.get_tmdb_data(title, year)
        if tmdb_data:
            enriched_row = self._enrich_with_tmdb_data(enriched_row, tmdb_data)
            self.stats['tmdb_success'] += 1
        else:
            self.stats['errors'].append(f"Film {film_id}: No TMDB data found")
        
        # 2. Get trailer views
        if tmdb_data and not row.get('kinocheck_trailer_views'):
            views = self.get_trailer_views(tmdb_data['tmdb_id'], title)
            if views:
                enriched_row['kinocheck_trailer_views'] = views
                self.stats['trailer_success'] += 1
        
        # 3. Get opening screens (if not already populated)
        if not row.get('screens_opening_day'):
            screens = self.get_opening_screens(title, year)
            if screens:
                enriched_row['screens_opening_day'] = screens
                self.stats['screens_success'] += 1
        
        self.stats['processed'] += 1
        return enriched_row

    def _enrich_with_tmdb_data(self, row: Dict, tmdb_data: Dict) -> Dict:
        """Enrich a row with TMDB data"""
        details = tmdb_data['details']
        credits = tmdb_data['credits']
        
        # Fill in missing basic data
        if not row.get('runtime_minutes') and details.get('runtime'):
            row['runtime_minutes'] = details['runtime']
        
        if not row.get('release_date') and details.get('release_date'):
            row['release_date'] = details['release_date']
        
        if not row.get('budget_usd') and details.get('budget'):
            row['budget_usd'] = details['budget'] // 1000000  # Convert to millions
        
        # Map rating from MPAA rating or certification
        if not row.get('rating'):
            # This would need more sophisticated mapping from releases data
            row['rating'] = 'TBD'
        
        # Fill in genre IDs
        genres = details.get('genres', [])[:5]  # Max 5 genres
        for i, genre in enumerate(genres, 1):
            genre_key = f'genre{i}_id'
            if not row.get(genre_key):
                row[genre_key] = self.mappings['genres'].get(genre['name'], genre['name'])
        
        # Fill in people IDs
        cast = credits.get('cast', [])
        crew = credits.get('crew', [])
        
        # Director
        directors = [person for person in crew if person['job'] == 'Director']
        if directors and not row.get('director_id'):
            director_name = directors[0]['name']
            row['director_id'] = self.mappings['people'].get(director_name, {}).get('id', director_name)
        
        # Main cast
        for i, actor in enumerate(cast[:6], 1):
            actor_key = f'actor{i}_id'
            if not row.get(actor_key):
                actor_name = actor['name']
                row[actor_key] = self.mappings['people'].get(actor_name, {}).get('id', actor_name)
        
        # Studio
        if not row.get('studio_id') and details.get('production_companies'):
            studio_name = details['production_companies'][0]['name']
            row['studio_id'] = self.mappings['studios'].get(studio_name, {}).get('id', studio_name)
        
        return row

    def run_enrichment(self, input_file: str, output_file: str, 
                      dry_run: bool = False, target_film_id: str = None):
        """Main enrichment process"""
        print(f"Starting film data enrichment...")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} not found")
            return False
        
        # Read input CSV
        rows = []
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"Found {len(rows)} films to process")
        
        # Filter to specific film if requested
        if target_film_id:
            rows = [row for row in rows if row['id'] == target_film_id]
            print(f"Filtering to film ID {target_film_id}: {len(rows)} films")
        
        # Process each film
        enriched_rows = []
        for row in rows:
            enriched_row = self.process_film(row, dry_run)
            enriched_rows.append(enriched_row)
        
        # Save results
        if not dry_run:
            fieldnames = list(enriched_rows[0].keys()) if enriched_rows else []
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(enriched_rows)
            
            # Save mappings
            self.save_mappings()
            
            print(f"\nResults written to {output_file}")
        
        # Print statistics
        self.print_stats()
        return True

    def print_stats(self):
        """Print processing statistics"""
        print(f"\n=== Processing Statistics ===")
        print(f"Films processed: {self.stats['processed']}")
        print(f"TMDB data retrieved: {self.stats['tmdb_success']}")
        print(f"Trailer views found: {self.stats['trailer_success']}")
        print(f"Screen counts found: {self.stats['screens_success']}")
        
        if self.stats['errors']:
            print(f"\nErrors ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(description='Enrich film dataset with TMDB and other sources')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run without making API calls or writing files')
    parser.add_argument('--film-id', type=str, 
                       help='Process only the film with this ID')
    parser.add_argument('--input', type=str, default=INPUT_CSV,
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV,
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Validate API keys
    if not args.dry_run:
        if not TMDB_API_KEY:
            print("Error: TMDB_API_KEY not found in environment variables")
            print("Please add your TMDB API key to the .env file")
            return 1
        
        if not KINOCHECK_API_KEY:
            print("Warning: KINOCHECK_API_KEY not found - trailer views will be limited")
        
        if not YOUTUBE_API_KEY:
            print("Warning: YOUTUBE_API_KEY not found - no YouTube fallback for trailer views")
    
    # Run enrichment
    enricher = DataEnricher()
    success = enricher.run_enrichment(
        args.input, 
        args.output, 
        args.dry_run, 
        args.film_id
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
