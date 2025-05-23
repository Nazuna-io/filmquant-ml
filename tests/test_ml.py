import pandas as pd
import pytest

from app.ml.prediction import (
    _feature_processor,  # Access the mock processor for test inspection
)
from app.ml.prediction import _model  # Access mock model
from app.ml.prediction import (  # Access loaded ref data
    _all_genres_list,
    _all_personnel_list,
    _all_studios_list,
    find_similar_films,
    get_prediction,
    initialize_model_and_processor,
    preprocess_input,
)
from app.static_data_loader import (
    get_cast_and_crew,
    get_genres,
    get_historical_films,
    get_studios,
)


@pytest.fixture(scope="module", autouse=True)
def setup_ml_module():
    """Ensure ML model and reference data are initialized before ML tests run."""
    initialize_model_and_processor()


@pytest.fixture
def sample_raw_input_data():
    # Ensure these IDs are present in your sample data files
    genres = get_genres()
    personnel = get_cast_and_crew()
    studios = get_studios()

    return {
        "title": "Test ML Film",
        "genre_ids": [genres[0]["id"], genres[1]["id"]] if len(genres) >= 2 else [],
        "director_id": next(
            (p["id"] for p in personnel if p["role"] == "director"), None
        ),
        "cast_ids": [p["id"] for p in personnel if p["role"] == "actor"][:2],
        "studio_id": studios[0]["id"] if studios else None,
        "budget_usd": 80000000,
        "runtime_minutes": 115,
        "release_date": "2024-07-15",
        "screens_opening_day": 3100,
        "marketing_budget_est_usd": 25000000,
        "trailer_views_prerelease": 15000000,
    }


def test_preprocess_input(sample_raw_input_data):
    """Test the preprocessing function."""
    df = preprocess_input(
        sample_raw_input_data,
        _feature_processor,
        _all_genres_list,
        _all_personnel_list,
        _all_studios_list,
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "budget_usd" in df.columns
    assert (
        "genre_Action" in df.columns or "genre_Comedy" in df.columns
    )  # Check for some genre columns
    assert df.shape[0] == 1  # Should be a single row
    # Check if the number of columns matches the mock model's expected features
    assert len(df.columns) == len(_model.feature_names)
    assert all(col_name in _model.feature_names for col_name in df.columns)


def test_get_prediction(sample_raw_input_data):
    """Test the main prediction orchestration function."""
    output = get_prediction(sample_raw_input_data)
    assert isinstance(output, dict)
    assert "predicted_revenue_usd" in output
    assert "confidence_interval_low_usd" in output
    assert "top_factors" in output
    assert isinstance(output["top_factors"], list)
    assert "comparable_films" in output
    assert isinstance(output["comparable_films"], list)
    assert output["predicted_revenue_usd"] > 0  # Mock prediction should be positive


def test_find_similar_films(sample_raw_input_data):
    """Test finding similar films."""
    # Need a primary_genre_id for this test. Get it from sample_raw_input_data
    predicted_film_info = {
        "primary_genre_id": (
            sample_raw_input_data["genre_ids"][0]
            if sample_raw_input_data["genre_ids"]
            else None
        ),
        "budget_usd": sample_raw_input_data["budget_usd"],
    }
    if not predicted_film_info["primary_genre_id"]:
        pytest.skip(
            "Skipping test_find_similar_films as no primary genre ID in sample data."
        )

    historical_films = get_historical_films()
    similar = find_similar_films(predicted_film_info, historical_films, top_n=3)
    assert isinstance(similar, list)
    # Depending on sample_historical_films.json, this might return 0 to 3 films
    assert len(similar) <= 3
    if similar:
        assert "title" in similar[0]
        assert "predicted_or_actual_revenue_usd" in similar[0]
