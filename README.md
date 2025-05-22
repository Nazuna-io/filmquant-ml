# FilmQuant ML - Machine Learning Film Revenue Prediction

FilmQuant ML uses machine learning to predict film box office revenue based on various input features and offers insights into these predictions. The system uses a pre-trained XGBoost model (currently mocked) for inference.

## Core Features

*   **Revenue Prediction:** Predicts total box office revenue with a 90% confidence interval (lower bound, median, upper bound).
*   **Domestic/International Split:** Estimates the percentage split between domestic and international revenue (currently fixed at 33% domestic / 67% international for MVP).
*   **Top Influencing Factors:** Identifies the top 3-5 features most influencing the prediction (based on mock feature importance).
*   **Comparable Historical Films:** Shows 3-5 similar historical films based on genre and budget proximity (+/- 25%).
*   **User Interface (Gradio):**
    *   Clean, professional custom UI with enhanced visual design.
    *   Input form for film details (title, genres, director, cast, studio, budget, runtime, release date, screens, marketing, trailer views).
    *   Displays prediction results, confidence intervals, revenue splits, top factors, and comparable films.
    *   "Historical Validation Mode": Allows users to select a historical film, pre-filling its data to see actuals and then re-predict or modify inputs.
*   **Backend API (FastAPI):**
    *   `/api/v1/predict` (POST): Accepts film details and returns a full prediction.
    *   `/api/v1/validate` (POST): Accepts a historical film ID and optional overrides, returns actuals vs. prediction.
    *   `/api/v1/reference_data` (GET): Provides lists of genres, cast/crew, historical films, and studios for UI population.
*   **Data Ingestion Pipeline:** Flexible pipeline for preprocessing and transforming training data.
*   **Evaluation Framework:** Comprehensive framework for model evaluation and performance tracking.

## Technical Stack

*   **Frontend:** Gradio
*   **Backend:** FastAPI (Python 3.10)
*   **ML Model:** XGBoost (mocked for MVP, intended for quantile regression)
*   **Data Handling:** Pandas, NumPy, Scikit-learn (for preprocessing, currently mocked)
*   **Data Storage (MVP):** Static JSON files for reference data (`genres`, `cast_and_crew`, `historical_films`, `studios`). Model files are placeholders.
*   **Deployment Target:** Google Cloud Platform (Cloud Run for app, Cloud Storage for data/model).
*   **Containerization:** Docker
*   **Testing:** Pytest, Selenium for UI tests

## Project Structure

```
filmquant-ml/
├── app/                    # Main application source code
│   ├── __init__.py
│   ├── main.py             # FastAPI app initialization & Gradio UI
│   ├── api/                # FastAPI API routes
│   │   └── routes.py
│   ├── data_ingestion/     # Data preprocessing and ingestion pipeline
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── pipeline.py
│   │   └── processors.py
│   ├── evaluation/         # Model evaluation framework
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── model_tracker.py
│   │   └── visualization.py
│   ├── ml/                 # ML prediction logic
│   │   ├── __init__.py
│   │   ├── prediction.py
│   │   └── utils.py
│   ├── static_data_loader.py # Loads reference JSON data
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── logging.py
├── config/                 # Configuration files
│   └── data_ingestion_default.json
├── data/                   # Sample JSON data and model placeholders
│   ├── sample_*.json
│   └── models/
│       └── placeholder_*.ubj/pkl
├── logs/                   # Application logs (created at runtime)
├── tests/                  # Unit and integration tests
│   ├── conftest.py         # Pytest fixtures
│   ├── test_api.py
│   ├── test_data_ingestion.py
│   ├── test_error_handling.py
│   ├── test_evaluation.py
│   ├── test_gradio_logic.py
│   ├── test_ml.py
│   └── test_ui.py          # Selenium UI tests
├── manual_tests/           # Scripts for manual testing
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── requirements-dev.txt    # Development dependencies
└── README.md               # This file
```

## Setup and Running

### Prerequisites

*   Python 3.10+
*   Git
*   Firefox (for Selenium UI tests)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/filmquant-ml.git
    cd filmquant-ml
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv-filmquant-ml
    source venv-filmquant-ml/bin/activate  # On Windows: venv-filmquant-ml\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt  # For development and testing
    ```

### Running the Application Locally

To run the Gradio web interface and FastAPI backend:

```bash
python -m app.main
```

The application will be available at:
- FastAPI: `http://localhost:8081/`
- Gradio UI: `http://localhost:8081/gradio/`
- API endpoints: `http://localhost:8081/api/v1/`

### Running the Data Ingestion Pipeline

To run the data ingestion pipeline:

```bash
python -m app.data_ingestion.ingest --input /path/to/input.csv --output /path/to/output.csv --config /path/to/config.json
```

For more options:

```bash
python -m app.data_ingestion.ingest --help
```

### Running Model Evaluation

To evaluate a model:

```bash
python -m app.evaluation.evaluate --data /path/to/data.csv --model-version v1 --true-column actual_revenue --pred-column predicted_revenue
```

For more options:

```bash
python -m app.evaluation.evaluate --help
```

### Running Tests

To run all the automated tests:

```bash
pytest
```

To run specific test files:

```bash
pytest tests/test_gradio_logic.py  # Run only Gradio logic tests
pytest tests/test_api.py           # Run only API tests
pytest tests/test_ui.py            # Run Selenium UI tests (requires Firefox)
```

### Manual Testing

For manual testing of the Gradio UI:

```bash
python manual_tests/test_gradio_ui.py  # Opens a browser to the Gradio UI
```

## Recent Updates

- Enhanced UI with a clean, professional custom design
- Implemented centralized configuration system with support for YAML files, environment variables, and command-line arguments
- Improved error handling and input validation throughout the application
- Implemented comprehensive logging system
- Created a flexible data ingestion pipeline for preprocessing and transforming training data
- Added a robust model evaluation framework for tracking model performance
- Enhanced API schema validation and error handling
- Added extensive unit tests for all new components

## Future Development (Post-MVP)

*   Integrate actual pre-trained XGBoost model and feature processor from GCS.
*   Implement SHAP values for more accurate feature importance.
*   Enhance "similar films" logic (e.g., consider cast, release date proximity).
*   Expand reference data and move to a more robust storage solution if needed.
*   User authentication and persistent user data (if required).
*   PDF export of predictions.
*   Improve test coverage, especially for edge cases.

## License

This project is licensed under the MIT License.


