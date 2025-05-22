# FilmQuant ML - Machine Learning Film Revenue Prediction

FilmQuant ML uses machine learning to predict film box office revenue based on various input features and offers insights into these predictions. The system includes a comprehensive data enrichment pipeline and uses XGBoost for quantile regression modeling.

## Core Features

### 🎯 **Revenue Prediction**
*   Predicts total box office revenue with confidence intervals (lower bound, median, upper bound)
*   Domestic/International revenue split estimation
*   Feature importance analysis showing top influencing factors
*   Comparable historical films based on genre and budget proximity

### 🔧 **Data Enrichment Pipeline** *(New)*
*   **Automated data collection** from TMDB API for comprehensive film metadata
*   **Trailer view integration** via Kinocheck API with YouTube fallback
*   **ID mapping system** for consistent data references across sources
*   **Incremental processing** with dry-run support and error recovery
*   **Support for 500-1000+ films** with 80-95% automation coverage

### 🖥️ **User Interface (Gradio)**
*   Clean, professional custom UI with enhanced visual design
*   **Comprehensive studio database** including major studios (Disney, Universal, Warner Bros, etc.) and independents (A24, Searchlight, etc.)
*   **User-friendly input format** with budget and trailer views in millions for better UX
*   Quick-select buttons for A-list directors, actors, and major studios
*   Input form for film details (title, genres, director, cast, studio, budget, runtime, release date, trailer views)
*   Displays prediction results, confidence intervals, revenue splits, top factors, and comparable films
*   Historical validation mode for testing predictions against actual results

### 🚀 **Backend API (FastAPI)**
*   `/api/v1/predict` (POST): Film details → revenue prediction
*   `/api/v1/validate` (POST): Historical film validation
*   `/api/v1/reference_data` (GET): Genre, cast, crew, and studio reference data
*   Authentication middleware and comprehensive error handling

### 📊 **ML & Evaluation Framework**
*   XGBoost quantile regression for uncertainty quantification
*   Comprehensive model evaluation and performance tracking
*   Flexible data ingestion pipeline for training data preprocessing
*   Feature engineering with genre encoding, cast analysis, and temporal features

## Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/filmquant-ml.git
cd filmquant-ml
python -m venv venv-filmquant-ml
source venv-filmquant-ml/bin/activate  # Windows: venv-filmquant-ml\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
# Edit .env and add your API keys:
# TMDB_API_KEY=your_tmdb_bearer_token_here
# KINOCHECK_API_KEY=your_kinocheck_key_here (optional)
# YOUTUBE_API_KEY=your_youtube_key_here (optional)
```

### 3. Run the Application

#### Standalone Gradio Interface (Recommended)
```bash
python3 filmquant_ui.py
```
Visit: `http://localhost:8081/`

#### Full FastAPI + Gradio Application
```bash
python -m app.main
```
Visit: `http://localhost:8081/gradio/`

## Data Enrichment

### Enrich Your Dataset
```bash
# Test API connections
python scripts/test_enrichment.py

# Dry run to see what would be processed
python scripts/enrich_historical_data.py --dry-run

# Process a single film for testing
python scripts/enrich_historical_data.py --film-id 1

# Process your entire dataset
python scripts/enrich_historical_data.py
```

### Data Sources & Coverage
| Field | Primary Source | Coverage | Notes |
|-------|----------------|----------|-------|
| **Genres** | TMDB | ~95% | Comprehensive genre data |
| **Cast & Crew** | TMDB | ~90% | Top 6 actors + director |
| **Budget & Runtime** | TMDB | ~80% | Production budget in millions |
| **Release Dates** | TMDB | ~98% | Accurate release dates |
| **Trailer Views** | Kinocheck + YouTube | ~70% | Multiple API fallbacks |
| **Opening Screens** | Box Office Mojo* | ~60% | *Implementation needed |
| **Studio Data** | TMDB | ~85% | Production companies |

See `scripts/README.md` for detailed documentation.

## Technical Stack

*   **Frontend:** Gradio with custom theming
*   **Backend:** FastAPI (Python 3.10+)
*   **ML Model:** XGBoost quantile regression
*   **Data APIs:** TMDB, Kinocheck, YouTube Data API
*   **Data Processing:** Pandas, NumPy, Scikit-learn
*   **Testing:** Pytest with 4/5 tests passing
*   **Deployment:** Google Cloud Platform (Cloud Run + Cloud Storage)
*   **Containerization:** Docker

## Project Structure

```
filmquant-ml/
├── app/                          # Main application
│   ├── main.py                   # FastAPI + Gradio app
│   ├── api/routes.py             # API endpoints
│   ├── ml/prediction.py          # ML prediction logic
│   ├── data_ingestion/           # Data preprocessing pipeline
│   ├── evaluation/               # Model evaluation framework
│   └── utils/logging.py          # Logging utilities
├── scripts/                      # Data enrichment tools ✨
│   ├── enrich_historical_data.py # Main enrichment script
│   ├── test_enrichment.py        # API connection tests
│   ├── setup_enrichment.sh       # Setup automation
│   └── README.md                 # Detailed enrichment docs
├── data/                         # Datasets and models
│   ├── filmquant-ml-historical-data-2023.csv  # Training data
│   ├── mappings/                 # ID mapping files
│   └── models/                   # Model artifacts
├── tests/                        # Test suites
│   ├── test_enrichment/          # Enrichment pipeline tests ✨
│   ├── test_api.py               # API tests
│   ├── test_ml.py                # ML logic tests
│   └── test_ui.py                # UI integration tests
├── .env.example                  # Environment configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Development Workflow

### Running Tests
```bash
# All tests
pytest

# Specific test suites
pytest tests/test_enrichment/     # Data enrichment tests
pytest tests/test_api.py          # API functionality
pytest tests/test_ml.py           # ML prediction logic
pytest tests/test_ui.py           # UI integration (requires Firefox)
```

### Model Training Pipeline
```bash
# Data ingestion
python -m app.data_ingestion.ingest --input data/raw.csv --output data/processed.csv

# Model evaluation
python -m app.evaluation.evaluate --data data/processed.csv --model-version v1

# Manual testing
python manual_tests/test_gradio_ui.py
```

## API Keys Setup

### TMDB API (Required)
1. Visit [themoviedb.org](https://www.themoviedb.org/)
2. Create account → Settings → API
3. Request API key (free for non-commercial use)
4. Add Bearer token to `.env`

### Kinocheck API (Optional - for trailer views)
1. Visit [kinocheck.com/api](https://www.kinocheck.com/api)
2. Sign up → Generate API key
3. Add to `.env`

### YouTube Data API (Optional - trailer view fallback)
1. Google Cloud Console → Create project
2. Enable YouTube Data API v3
3. Create credentials (API key)
4. Add to `.env`

## Recent Updates

- ✨ **Added comprehensive data enrichment pipeline** with TMDB integration
- ✨ **Automated trailer view collection** via Kinocheck + YouTube APIs
- ✨ **ID mapping system** for consistent data references
- ✨ **Unit test coverage** for enrichment functionality (4/5 tests passing)
- 🔧 **Removed legacy BORP references** - fully rebranded to FilmQuant ML
- 🏗️ **Enhanced project structure** with dedicated scripts and test directories
- 📚 **Comprehensive documentation** for data enrichment workflows

## Current Status

- **✅ Data Enrichment Pipeline:** Fully functional with TMDB integration
- **✅ Web Interface:** Gradio UI with custom theming
- **✅ API Backend:** FastAPI with authentication
- **🚧 ML Model:** Mock implementation (training dataset being prepared)
- **🚧 Box Office Scraping:** Placeholder (implementation needed)
- **🚧 Production Deployment:** Ready for GCP deployment

## Future Development

### Short Term
*   Complete Box Office Mojo scraping for opening screen counts
*   Train actual XGBoost model on enriched historical dataset
*   Implement SHAP values for feature importance analysis
*   Deploy to Google Cloud Platform

### Long Term
*   Expand to international film markets
*   Real-time data updates and model retraining
*   Advanced ensemble methods and deep learning models
*   User authentication and personalized predictions
*   PDF export and advanced reporting features

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**FilmQuant ML** - Predicting box office success with machine learning and comprehensive data enrichment. 🎬📊
