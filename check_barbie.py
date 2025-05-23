#!/usr/bin/env python3
import sys

sys.path.append(".")
from scripts.enrich_historical_data import DataEnricher

enricher = DataEnricher()
tmdb_data = enricher.get_tmdb_data("Barbie", 2023)
if tmdb_data:
    details = tmdb_data["details"]
    print(f"Barbie TMDB data:")
    print(f'  Runtime: {details["runtime"]} minutes')
    print(f'  Budget: ${details["budget"]:,}')
    print(f'  Genres: {[g["name"] for g in details["genres"]]}')
