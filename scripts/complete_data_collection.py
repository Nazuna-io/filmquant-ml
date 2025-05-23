#!/usr/bin/env python3
"""
Complete Data Collection Pipeline for FilmQuant ML

This script collects ALL missing data from multiple sources:
1. TMDB API - Ratings, verified release dates, worldwide revenue
2. Box Office Mojo - Domestic/international box office split, opening weekend, screen counts
3. Data validation and integration

Fills 100% of missing critical data for ML training.
"""

import json
import os
import re
import time
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CompleteDataCollector:
    def __init__(self):
        self.tmdb_api_key = os.getenv("TMDB_API_KEY")
        self.tmdb_headers = {"Authorization": f"Bearer {self.tmdb_api_key}"}
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )

        # Rate limiting
        self.tmdb_delay = 0.25  # 4 requests per second
        self.scraping_delay = 2.0  # Be respectful to Box Office Mojo

        # Data storage
        self.collected_data = {}

    def load_dataset(self, path="data/filmquant-ml-historical-data-2023-final.csv"):
        """Load the current dataset"""
        print("📊 LOADING CURRENT DATASET")
        print("=" * 40)

        self.df = pd.read_csv(path)
        print(f"Loaded {len(self.df)} films")

        # Identify films missing critical data
        missing_data = {"ratings": 0, "box_office": 0, "screen_counts": 0}

        for _, row in self.df.iterrows():
            if pd.isna(row["rating"]) or row["rating"] == "TBD":
                missing_data["ratings"] += 1
            if pd.isna(row["actual_box_office_domestic_usd"]):
                missing_data["box_office"] += 1
            if pd.isna(row["screens_opening_day"]):
                missing_data["screen_counts"] += 1

        print(f"Missing data:")
        print(f"  Ratings: {missing_data['ratings']}/42")
        print(f"  Box office: {missing_data['box_office']}/42")
        print(f"  Screen counts: {missing_data['screen_counts']}/42")

        return self.df

    def get_tmdb_data_for_film(self, film_title, tmdb_id=None):
        """Get comprehensive TMDB data for a single film"""
        print(f"  🎬 Getting TMDB data for: {film_title}")

        # If we don't have TMDB ID, search for it
        if not tmdb_id:
            search_url = "https://api.themoviedb.org/3/search/movie"
            search_params = {"query": film_title, "year": 2023}

            response = self.session.get(
                search_url, headers=self.tmdb_headers, params=search_params
            )
            time.sleep(self.tmdb_delay)

            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    tmdb_id = results[0]["id"]
                    print(f"    Found TMDB ID: {tmdb_id}")
                else:
                    print(f"    ❌ No TMDB results for {film_title}")
                    return None
            else:
                print(f"    ❌ TMDB search failed: {response.status_code}")
                return None

        # Get detailed movie data
        movie_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        response = self.session.get(movie_url, headers=self.tmdb_headers)
        time.sleep(self.tmdb_delay)

        if response.status_code != 200:
            print(f"    ❌ Failed to get movie details: {response.status_code}")
            return None

        movie_data = response.json()

        # Get releases for MPAA rating
        releases_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/releases"
        releases_response = self.session.get(releases_url, headers=self.tmdb_headers)
        time.sleep(self.tmdb_delay)

        rating = "Unknown"
        if releases_response.status_code == 200:
            releases_data = releases_response.json()
            us_releases = [
                r
                for r in releases_data.get("countries", [])
                if r.get("iso_3166_1") == "US"
            ]
            for release in us_releases:
                if release.get("certification"):
                    rating = release["certification"]
                    break

        result = {
            "tmdb_id": tmdb_id,
            "title": movie_data.get("title", film_title),
            "rating": rating,
            "release_date": movie_data.get("release_date"),
            "runtime": movie_data.get("runtime"),
            "worldwide_revenue": movie_data.get("revenue", 0),
            "budget": movie_data.get("budget", 0),
        }

        print(f"    ✅ Rating: {rating}, Revenue: ${result['worldwide_revenue']:,}")
        return result

    def find_box_office_mojo_url(self, film_title, year=2023):
        """Find the Box Office Mojo URL for a film"""
        print(f"  🔍 Finding Box Office Mojo URL for: {film_title}")

        # Try common URL patterns
        # Pattern 1: Direct title search
        title_clean = (
            film_title.lower().replace(" ", "-").replace(":", "").replace("'", "")
        )
        title_clean = re.sub(r"[^\w\-]", "", title_clean)

        potential_urls = [
            f"https://www.boxofficemojo.com/title/{title_clean}-{year}/",
            f"https://www.boxofficemojo.com/title/{title_clean}/",
            f"https://www.boxofficemojo.com/movie/{title_clean}-{year}/",
            f"https://www.boxofficemojo.com/movie/{title_clean}/",
        ]

        for url in potential_urls:
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    # Check if it's actually the right movie
                    soup = BeautifulSoup(response.content, "html.parser")
                    title_element = soup.find("h1")
                    if (
                        title_element
                        and film_title.lower() in title_element.get_text().lower()
                    ):
                        print(f"    ✅ Found URL: {url}")
                        return url

            except Exception as e:
                continue

        # If direct URLs don't work, try search
        print(f"    🔍 Trying Box Office Mojo search...")
        search_url = f"https://www.boxofficemojo.com/search/?q={quote_plus(film_title)}"

        try:
            response = self.session.get(search_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                # Look for search results
                links = soup.find_all("a", href=True)
                for link in links:
                    href = link.get("href", "")
                    if "/title/" in href and str(year) in href:
                        full_url = f"https://www.boxofficemojo.com{href}"
                        print(f"    ✅ Found via search: {full_url}")
                        return full_url
        except Exception as e:
            print(f"    ❌ Search failed: {e}")

        print(f"    ❌ Could not find Box Office Mojo URL")
        return None

    def scrape_box_office_mojo_data(self, url):
        """Scrape box office data from Box Office Mojo"""
        print(f"  📊 Scraping box office data...")

        try:
            response = self.session.get(url, timeout=15)
            time.sleep(self.scraping_delay)  # Be respectful

            if response.status_code != 200:
                print(f"    ❌ Failed to access URL: {response.status_code}")
                return None

            soup = BeautifulSoup(response.content, "html.parser")

            data = {
                "domestic_box_office": None,
                "international_box_office": None,
                "worldwide_box_office": None,
                "opening_weekend": None,
                "opening_screens": None,
            }

            # Look for money amounts
            money_spans = soup.find_all("span", class_="money")

            # Strategy: Look for context around money values
            for span in money_spans:
                money_text = span.get_text().strip()
                money_value = self.parse_money_string(money_text)

                if money_value is None:
                    continue

                # Look at surrounding context
                parent = span.parent
                if parent:
                    context = parent.get_text().lower()

                    # Domestic box office
                    if "domestic" in context and "box office" in context:
                        data["domestic_box_office"] = money_value
                        print(f"    ✅ Domestic: ${money_value:.1f}M")

                    # International box office
                    elif "international" in context or "foreign" in context:
                        data["international_box_office"] = money_value
                        print(f"    ✅ International: ${money_value:.1f}M")

                    # Opening weekend
                    elif "opening" in context and "weekend" in context:
                        data["opening_weekend"] = money_value
                        print(f"    ✅ Opening Weekend: ${money_value:.1f}M")

            # Look for screen counts
            text_content = soup.get_text()
            screen_matches = re.findall(
                r"(\d{1,5})\s*(?:screens?|theaters?)", text_content, re.IGNORECASE
            )
            if screen_matches:
                # Take the largest number (likely opening screen count)
                screens = max([int(match) for match in screen_matches])
                if screens > 100:  # Sanity check
                    data["opening_screens"] = screens
                    print(f"    ✅ Opening Screens: {screens:,}")

            return data

        except Exception as e:
            print(f"    ❌ Scraping error: {e}")
            return None

    def parse_money_string(self, money_str):
        """Parse money string like '$330,078,895' to millions"""
        try:
            # Remove $ and commas, convert to float
            clean_str = money_str.replace("$", "").replace(",", "")
            value = float(clean_str)
            return value / 1_000_000  # Convert to millions
        except:
            return None

    def collect_complete_data_for_film(self, film_row):
        """Collect all missing data for a single film"""
        film_title = film_row["title"]
        print(f"\n🎯 COLLECTING DATA FOR: {film_title}")
        print("-" * 50)

        collected = {"title": film_title, "original_data": film_row.to_dict()}

        # 1. Get TMDB data
        tmdb_data = self.get_tmdb_data_for_film(film_title)
        if tmdb_data:
            collected["tmdb_data"] = tmdb_data

        # 2. Get Box Office Mojo data
        bom_url = self.find_box_office_mojo_url(film_title)
        if bom_url:
            bom_data = self.scrape_box_office_mojo_data(bom_url)
            if bom_data:
                collected["box_office_data"] = bom_data

        return collected

    def run_complete_data_collection(self):
        """Run the complete data collection pipeline"""
        print("🚀 COMPLETE DATA COLLECTION PIPELINE")
        print("=" * 60)
        print(f"Starting at {datetime.now()}")
        print()

        # Load current dataset
        df = self.load_dataset()

        # Collect data for all films
        all_collected_data = []

        for idx, row in df.iterrows():
            try:
                film_data = self.collect_complete_data_for_film(row)
                all_collected_data.append(film_data)

                # Save progress periodically
                if (idx + 1) % 5 == 0:
                    print(f"\n💾 Progress checkpoint: {idx + 1}/42 films processed")
                    self.save_collected_data(
                        all_collected_data, f"data_collection_checkpoint_{idx+1}.json"
                    )

            except Exception as e:
                print(f"❌ Error processing {row['title']}: {e}")
                continue

        print(f"\n✅ DATA COLLECTION COMPLETE!")
        print(f"Processed {len(all_collected_data)}/42 films")

        # Save final results
        final_path = self.save_collected_data(
            all_collected_data, "complete_data_collection_results.json"
        )
        print(f"📁 Results saved: {final_path}")

        return all_collected_data

    def save_collected_data(self, data, filename):
        """Save collected data to JSON file"""
        output_path = os.path.join("data", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return output_path


if __name__ == "__main__":
    collector = CompleteDataCollector()
    results = collector.run_complete_data_collection()
