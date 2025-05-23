#!/usr/bin/env python3
import os
import sys

sys.path.append(".")
from scripts.enrich_historical_data import DataEnricher

print("Testing TMDB with Oppenheimer...")
enricher = DataEnricher()
tmdb_data = enricher.get_tmdb_data("Oppenheimer", 2023)

if tmdb_data:
    details = tmdb_data["details"]
    print(f"✓ TMDB search successful!")
    print(f'  Title: {details["title"]}')
    print(f'  Budget: ${details["budget"]:,}')
    print(f'  Runtime: {details["runtime"]} minutes')
    print(f'  Release Date: {details["release_date"]}')
    print(f'  Genres: {[g["name"] for g in details["genres"]]}')

    # Show some cast
    cast = tmdb_data["credits"]["cast"][:5]
    print(f'  Top Cast: {[actor["name"] for actor in cast]}')

    # Show director
    crew = tmdb_data["credits"]["crew"]
    directors = [person["name"] for person in crew if person["job"] == "Director"]
    print(f"  Director(s): {directors}")
else:
    print("✗ Failed to get TMDB data")
