# tests/test_error_handling.py
"""
Tests for error handling and input validation.
"""
import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes import PredictRequest, ValidateRequest


def test_predict_request_validation():
    """Test validation of PredictRequest model."""
    # Valid data should pass validation
    valid_data = {
        "title": "Test Film",
        "genre_ids": ["g001", "g002"],
        "director_id": "p001",
        "cast_ids": ["p002", "p003"],
        "studio_id": "s001",
        "budget_usd": 100000000,
        "runtime_minutes": 120,
        "release_date": "2024-07-04",
        "screens_opening_day": 3000,
        "marketing_budget_est_usd": 50000000,
        "trailer_views_prerelease": 20000000,
    }

    # This should not raise an exception
    request = PredictRequest(**valid_data)
    assert request.title == valid_data["title"]
    assert request.budget_usd == valid_data["budget_usd"]

    # Test required fields
    missing_title_data = {k: v for k, v in valid_data.items() if k != "title"}
    with pytest.raises(ValidationError):
        PredictRequest(**missing_title_data)

    missing_genre_data = {k: v for k, v in valid_data.items() if k != "genre_ids"}
    with pytest.raises(ValidationError):
        PredictRequest(**missing_genre_data)

    # Test field constraints
    invalid_title_data = valid_data.copy()
    invalid_title_data["title"] = ""  # Empty title
    with pytest.raises(ValidationError):
        PredictRequest(**invalid_title_data)

    invalid_budget_data = valid_data.copy()
    invalid_budget_data["budget_usd"] = -1000  # Negative budget
    with pytest.raises(ValidationError):
        PredictRequest(**invalid_budget_data)

    invalid_runtime_data = valid_data.copy()
    invalid_runtime_data["runtime_minutes"] = 0  # Zero runtime
    with pytest.raises(ValidationError):
        PredictRequest(**invalid_runtime_data)

    invalid_date_data = valid_data.copy()
    invalid_date_data["release_date"] = "not-a-date"  # Invalid date format
    with pytest.raises(ValidationError):
        PredictRequest(**invalid_date_data)


def test_validate_request_validation():
    """Test validation of ValidateRequest model."""
    # Valid data should pass validation
    valid_data = {
        "historical_film_id": "hf001",
        "override_features": {"budget_usd": 150000000},
    }

    # Mocking the get_historical_films function would be needed for full validation
    # This is a basic test without that mock
    request = ValidateRequest(**valid_data)
    assert request.historical_film_id == valid_data["historical_film_id"]
    assert request.override_features == valid_data["override_features"]

    # Test required fields
    missing_id_data = {"override_features": valid_data["override_features"]}
    with pytest.raises(ValidationError):
        ValidateRequest(**missing_id_data)

    # Test default value
    no_override_data = {"historical_film_id": valid_data["historical_film_id"]}
    request = ValidateRequest(**no_override_data)
    assert request.override_features == {}
