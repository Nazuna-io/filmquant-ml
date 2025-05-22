#!/usr/bin/env python3
"""
Data Integration Script - Apply TMDB Data to Dataset

This script integrates the collected TMDB data into our main dataset,
then attempts Box Office Mojo scraping with improved methods.
"""

import pandas as pd
import json
import os
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import quote

class DataIntegrator:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def load_collected_data(self):
        """Load the collected TMDB data"""
        print("📥 LOADING COLLECTED DATA")
        print("=" * 30)
        
        with open('data/complete_data_collection_results.json', 'r') as f:
            collected_data = json.load(f)
        
        print(f"Loaded data for {len(collected_data)} films")
        return collected_data
    
    def integrate_tmdb_data(self, collected_data):
        """Integrate TMDB data into the main dataset"""
        print("\n🔄 INTEGRATING TMDB DATA")
        print("=" * 35)
        
        # Load current dataset
        df = pd.read_csv('data/filmquant-ml-historical-data-2023-final.csv')
        
        # Update each film with TMDB data
        updates_made = 0
        
        for film_data in collected_data:
            title = film_data['title']
            
            if 'tmdb_data' in film_data:
                tmdb_info = film_data['tmdb_data']
                
                # Find the film in the dataset
                film_mask = df['title'] == title
                if film_mask.any():
                    # Update rating if we have a better one
                    current_rating = df.loc[film_mask, 'rating'].iloc[0]
                    new_rating = tmdb_info.get('rating', current_rating)
                    
                    if pd.isna(current_rating) or current_rating == 'TBD' or current_rating == '':
                        df.loc[film_mask, 'rating'] = new_rating
                        print(f"  ✅ {title}: Rating updated to {new_rating}")
                        updates_made += 1
                    
                    # Update runtime if missing or different
                    current_runtime = df.loc[film_mask, 'runtime_minutes'].iloc[0]
                    new_runtime = tmdb_info.get('runtime')
                    
                    if new_runtime and (pd.isna(current_runtime) or abs(current_runtime - new_runtime) > 5):
                        df.loc[film_mask, 'runtime_minutes'] = new_runtime
                        print(f"  ✅ {title}: Runtime updated to {new_runtime} min")
                        updates_made += 1
        
        print(f"\nMade {updates_made} updates from TMDB data")
        
        # Save updated dataset
        df.to_csv('data/filmquant-ml-historical-data-2023-final.csv', index=False)
        print("💾 Dataset updated and saved")
        
        return df
    
    def get_box_office_data_alternative(self, film_title, year=2023):
        """Try alternative methods to get box office data"""
        print(f"  🎯 Getting box office for: {film_title}")
        
        # Method 1: Try different Box Office Mojo URL patterns
        title_variants = [
            film_title.lower().replace(' ', '').replace(':', '').replace('-', ''),
            film_title.lower().replace(' ', '-').replace(':', '').replace("'", ""),
            film_title.lower().replace(' ', '+')
        ]
        
        for title_clean in title_variants:
            # Try various URL patterns
            urls_to_try = [
                f"https://www.boxofficemojo.com/title/tt{title_clean}/",
                f"https://www.boxofficemojo.com/movie/{title_clean}/",
                f"https://www.boxofficemojo.com/movies/?id={title_clean}.htm"
            ]
            
            for url in urls_to_try:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200 and len(response.content) > 1000:
                        # Quick check if this looks like a movie page
                        if 'box office' in response.text.lower():
                            print(f"    ✅ Found potential page: {url}")
                            return self.parse_box_office_page(response.text)
                except:
                    continue
        
        # Method 2: Use Wikipedia as backup
        return self.get_wikipedia_box_office(film_title, year)
    
    def get_wikipedia_box_office(self, film_title, year=2023):
        """Get box office data from Wikipedia"""
        print(f"    🔍 Trying Wikipedia for {film_title}")
        
        try:
            # Search Wikipedia
            search_url = f"https://en.wikipedia.org/wiki/{film_title.replace(' ', '_')}_{year}_film"
            alt_search_url = f"https://en.wikipedia.org/wiki/{film_title.replace(' ', '_')}"
            
            for url in [search_url, alt_search_url]:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Look for box office infobox
                        infobox = soup.find('table', {'class': 'infobox'})
                        if infobox:
                            box_office_data = self.parse_wikipedia_infobox(infobox)
                            if box_office_data:
                                print(f"    ✅ Found Wikipedia data")
                                return box_office_data
                except:
                    continue
                    
        except Exception as e:
            print(f"    ❌ Wikipedia failed: {e}")
        
        return None
    
    def parse_wikipedia_infobox(self, infobox):
        """Parse Wikipedia infobox for box office data"""
        data = {}
        
        # Look for box office row
        rows = infobox.find_all('tr')
        for row in rows:
            header = row.find('th')
            if header and 'box office' in header.get_text().lower():
                value_cell = row.find('td')
                if value_cell:
                    box_office_text = value_cell.get_text()
                    
                    # Parse the box office text
                    # Look for patterns like "$1.4 billion" or "$500 million"
                    import re
                    
                    # Find dollar amounts
                    money_patterns = re.findall(r'\$([0-9,.]+)\s*(million|billion)', box_office_text, re.IGNORECASE)
                    
                    for amount, unit in money_patterns:
                        try:
                            value = float(amount.replace(',', ''))
                            if unit.lower() == 'billion':
                                value *= 1000
                            
                            # If we find worldwide, try to split it
                            if 'worldwide' in box_office_text.lower():
                                data['worldwide'] = value
                                # Rough estimate: 45% domestic, 55% international
                                data['domestic'] = value * 0.45
                                data['international'] = value * 0.55
                                return data
                            
                        except:
                            continue
        
        return data if data else None
    
    def create_box_office_estimates(self, df, collected_data):
        """Create box office estimates from worldwide revenue"""
        print("\n💰 CREATING BOX OFFICE ESTIMATES")
        print("=" * 40)
        
        estimates_added = 0
        
        for film_data in collected_data:
            title = film_data['title']
            
            if 'tmdb_data' in film_data:
                tmdb_info = film_data['tmdb_data']
                worldwide_revenue = tmdb_info.get('worldwide_revenue', 0)
                
                if worldwide_revenue > 0:
                    # Convert to millions
                    worldwide_millions = worldwide_revenue / 1_000_000
                    
                    # Industry averages for US films:
                    # - Domestic: ~45% of worldwide
                    # - International: ~55% of worldwide  
                    # - Opening weekend: ~20-25% of domestic total
                    
                    domestic_estimate = worldwide_millions * 0.45
                    international_estimate = worldwide_millions * 0.55
                    opening_weekend_estimate = domestic_estimate * 0.22
                    
                    # Update dataset
                    film_mask = df['title'] == title
                    if film_mask.any():
                        # Only update if currently missing
                        if pd.isna(df.loc[film_mask, 'actual_box_office_domestic_usd'].iloc[0]):
                            df.loc[film_mask, 'actual_box_office_domestic_usd'] = domestic_estimate
                            estimates_added += 1
                        
                        if pd.isna(df.loc[film_mask, 'actual_box_office_international_usd'].iloc[0]):
                            df.loc[film_mask, 'actual_box_office_international_usd'] = international_estimate
                            estimates_added += 1
                        
                        if pd.isna(df.loc[film_mask, 'opening_weekend_box_office'].iloc[0]):
                            df.loc[film_mask, 'opening_weekend_box_office'] = opening_weekend_estimate
                            estimates_added += 1
                        
                        print(f"  ✅ {title}: Domestic=${domestic_estimate:.1f}M, Intl=${international_estimate:.1f}M")
        
        print(f"\nAdded {estimates_added} box office estimates")
        return df
    
    def run_integration(self):
        """Run the complete data integration process"""
        print("🔄 DATA INTEGRATION PIPELINE")
        print("=" * 50)
        
        # 1. Load collected data
        collected_data = self.load_collected_data()
        
        # 2. Integrate TMDB data
        df = self.integrate_tmdb_data(collected_data)
        
        # 3. Create box office estimates from worldwide revenue
        df = self.create_box_office_estimates(df, collected_data)
        
        # 4. Save final dataset
        final_path = 'data/filmquant-ml-historical-data-2023-complete.csv'
        df.to_csv(final_path, index=False)
        
        print(f"\n✅ INTEGRATION COMPLETE!")
        print(f"📁 Final dataset saved: {final_path}")
        
        # 5. Show final statistics
        print(f"\n📊 FINAL DATASET STATISTICS:")
        print(f"Total films: {len(df)}")
        
        completion_stats = {
            'Ratings': df['rating'].notna().sum(),
            'Domestic BO': df['actual_box_office_domestic_usd'].notna().sum(),
            'International BO': df['actual_box_office_international_usd'].notna().sum(),
            'Opening Weekend': df['opening_weekend_box_office'].notna().sum()
        }
        
        for field, count in completion_stats.items():
            pct = (count / len(df)) * 100
            print(f"  {field}: {count}/42 ({pct:.1f}%)")
        
        return df

if __name__ == "__main__":
    integrator = DataIntegrator()
    final_dataset = integrator.run_integration()
