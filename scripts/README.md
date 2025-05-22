# Film Data Enrichment Pipeline

This directory contains tools to automatically enrich the historical film dataset with data from multiple sources.

## Overview

The enrichment pipeline fills missing data in your historical dataset by:

1. **TMDB API Integration**: Fetches comprehensive film data including genres, cast, crew, budget, runtime, release dates
2. **Trailer Views**: Gets trailer view counts from Kinocheck API with YouTube fallback
3. **Box Office Data**: Scrapes opening day screen counts and other box office metrics
4. **ID Mapping System**: Creates consistent internal IDs for genres, people, and studios

## Quick Start

1. **Install dependencies**:
   ```bash
   cd /home/todd/filmquant-ml
   ./scripts/setup_enrichment.sh
   ```

2. **Set up your API keys** in `.env`:
   ```bash
   # Required
   TMDB_API_KEY=your_actual_tmdb_key_here
   
   # Optional (improves trailer view coverage)
   KINOCHECK_API_KEY=your_kinocheck_key_here
   YOUTUBE_API_KEY=your_youtube_key_here
   ```

3. **Test the connection**:
   ```bash
   python scripts/test_enrichment.py
   ```

4. **Run enrichment**:
   ```bash
   # Test with dry run first
   python scripts/enrich_historical_data.py --dry-run
   
   # Process one film for testing
   python scripts/enrich_historical_data.py --film-id 1
   
   # Process all films
   python scripts/enrich_historical_data.py
   ```

## Getting API Keys

### TMDB API Key (Required)
1. Go to https://www.themoviedb.org/
2. Create a free account
3. Go to Settings > API
4. Request an API key (free for non-commercial use)
5. Add to your `.env` file

### Kinocheck API Key (Optional)
1. Visit https://www.kinocheck.com/api
2. Sign up for a free account
3. Generate API key from dashboard
4. Add to your `.env` file

### YouTube API Key (Optional)
1. Go to Google Cloud Console
2. Create/select a project
3. Enable YouTube Data API v3
4. Create credentials (API key)
5. Add to your `.env` file

## Data Sources & Coverage

| Field | Primary Source | Fallback | Coverage Expected |
|-------|---------------|----------|------------------|
| Genre IDs | TMDB | Manual mapping | ~95% |
| Cast/Crew IDs | TMDB | Manual mapping | ~90% |
| Budget | TMDB | Manual research | ~80% |
| Runtime | TMDB | Manual research | ~98% |
| Release Date | TMDB | Manual research | ~98% |
| Trailer Views | Kinocheck | YouTube API | ~70% |
| Opening Screens | Box Office Mojo | Manual research | ~60% |
| Studio IDs | TMDB | Manual mapping | ~85% |
| Rating | TMDB Releases | Manual research | ~80% |

## Output Files

- `data/filmquant-ml-historical-data-2023-enriched.csv` - Enriched dataset
- `data/mappings/genres.json` - Genre name to ID mapping
- `data/mappings/people.json` - Cast/crew name to ID mapping  
- `data/mappings/studios.json` - Studio name to ID mapping
- `data/mappings/tmdb_ids.json` - Film title to TMDB ID mapping

## Command Line Options

```bash
python scripts/enrich_historical_data.py [OPTIONS]

Options:
  --dry-run          Show what would be processed without making API calls
  --film-id ID       Process only the film with this ID
  --input PATH       Custom input CSV file path
  --output PATH      Custom output CSV file path
  --help            Show help message
```

## Troubleshooting

### "No TMDB results found"
- Check if film title exactly matches TMDB
- Try adding year to the search
- Some international films may have different titles

### "No trailer views found"
- Kinocheck doesn't have all films
- YouTube fallback helps but isn't comprehensive
- Consider manual lookup for key films

### Rate Limiting
- TMDB: 40 requests per 10 seconds (handled automatically)
- Kinocheck: 1 request per second (handled automatically)
- If you hit limits, the script will pause and retry

### Missing Fields
- Some data simply isn't available in public APIs
- The script will flag missing data in the output
- Consider manual research for critical missing values

## Integration with FilmQuant ML

After enrichment, you may need to update the FilmQuant ML codebase to match the new CSV schema:

1. Update `app/static_data_loader.py` to use the new field names
2. Modify `app/ml/prediction.py` to handle the new ID mappings
3. Update any hardcoded field references in the Gradio interface

## Data Quality Notes

- **Budget values**: TMDB sometimes has estimated/unconfirmed budgets
- **Trailer views**: Numbers can vary significantly by platform and time
- **Opening screens**: Historical data may be incomplete for older films
- **Genre mapping**: Maintains consistency across your dataset but may differ from TMDB's genre IDs

## Future Enhancements

- **Box Office Mojo scraping**: Currently placeholder, needs implementation
- **International data**: TMDB has good international coverage
- **Historical validation**: Cross-reference with multiple sources
- **Automated updates**: Keep data fresh with new releases
