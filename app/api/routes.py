from fastapi import APIRouter, Query, HTTPException, Body, status
from pydantic import BaseModel, Field, field_validator, ValidationError, constr, confloat, conint, ConfigDict
from typing import List, Dict, Any, Optional, Annotated
from datetime import date, datetime
import logging
from enum import Enum

from app.ml.prediction import get_prediction, initialize_model_and_processor
from app.static_data_loader import get_genres, get_cast_and_crew, get_historical_films, get_studios
import copy # For deep copying override_features

# Configure logger
logger = logging.getLogger("filmquant_ml.api")

# --- Pydantic Models ---

class PredictRequest(BaseModel):
    """
    Request model for the box office revenue prediction endpoint.
    Contains comprehensive validation for all input fields.
    """
    title: constr(min_length=1, max_length=200) = Field(
        ..., description="Movie title, 1-200 characters"
    )
    genre_ids: List[str] = Field(
        ..., min_items=1, description="List of genre IDs. At least one genre is required."
    )
    director_id: Optional[str] = Field(
        None, description="ID of the director. Must exist in the reference data."
    )
    cast_ids: List[str] = Field(
        ..., min_items=1, max_items=10, description="List of cast member IDs. At least one is required."
    )
    studio_id: Optional[str] = Field(
        None, description="ID of the studio producing the film"
    )
    budget_usd: confloat(gt=0) = Field(
        ..., description="Production budget in USD. Must be positive."
    )
    runtime_minutes: conint(gt=0, lt=600) = Field(
        ..., description="Runtime in minutes. Must be positive and less than 600."
    )
    release_date: str = Field(
        ..., description="Release date in YYYY-MM-DD format"
    )
    screens_opening_day: conint(ge=0) = Field(
        ..., description="Number of screens on opening day. Must be non-negative."
    )
    marketing_budget_est_usd: Optional[confloat(ge=0)] = Field(
        None, description="Estimated marketing budget in USD. Must be non-negative if provided."
    )
    trailer_views_prerelease: Optional[conint(ge=0)] = Field(
        None, description="Number of trailer views before release. Must be non-negative if provided."
    )
    mpaa_rating: Optional[str] = Field(
        None, description="MPAA rating (e.g., 'G', 'PG', 'PG-13', 'R')"
    )
    is_sequel_or_franchise: Optional[bool] = Field(
        None, description="Whether the film is part of a franchise or a sequel"
    )

    @field_validator('release_date')
    @classmethod
    def validate_date_format(cls, v):
        try:
            parsed_date = date.fromisoformat(v)
            # Check if date is not too far in the past or future
            today = date.today()
            min_date = date(today.year - 10, 1, 1)  # 10 years ago
            max_date = date(today.year + 10, 12, 31)  # 10 years in future
            
            if parsed_date < min_date or parsed_date > max_date:
                logger.warning(f"Date {v} is outside the reasonable range")
            
            return v
        except ValueError:
            raise ValueError("release_date must be in YYYY-MM-DD format")

    @field_validator('genre_ids')
    @classmethod
    def validate_genre_ids(cls, v):
        valid_genres = {g['id'] for g in get_genres()}
        for genre_id in v:
            if genre_id not in valid_genres:
                raise ValueError(f"Genre ID '{genre_id}' not found in reference data")
        return v

    @field_validator('director_id')
    @classmethod
    def validate_director_id(cls, v):
        if v is None:
            return v
        
        valid_directors = {p['id'] for p in get_cast_and_crew() if p['role'] == 'director'}
        if v not in valid_directors:
            raise ValueError(f"Director ID '{v}' not found in reference data")
        return v

    @field_validator('studio_id')
    @classmethod
    def validate_studio_id(cls, v):
        if v is None:
            return v
        
        valid_studios = {s['id'] for s in get_studios()}
        if v not in valid_studios:
            raise ValueError(f"Studio ID '{v}' not found in reference data")
        return v

    @field_validator('cast_ids')
    @classmethod
    def validate_cast_ids(cls, v):
        valid_actors = {p['id'] for p in get_cast_and_crew() if p['role'] == 'actor'}
        for actor_id in v:
            if actor_id not in valid_actors:
                raise ValueError(f"Actor ID '{actor_id}' not found in reference data")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Example Movie",
                "genre_ids": ["g001", "g002"],
                "director_id": "p001",
                "cast_ids": ["p002", "p003"],
                "studio_id": "s001",
                "budget_usd": 100000000,
                "runtime_minutes": 120,
                "release_date": "2024-07-04",
                "screens_opening_day": 3000,
                "marketing_budget_est_usd": 50000000,
                "trailer_views_prerelease": 20000000
            }
        }
    )

class ValidateRequest(BaseModel):
    """
    Request model for validating predictions against historical films.
    """
    historical_film_id: str = Field(
        ..., description="ID of the historical film to compare against"
    )
    override_features: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Features to override in the historical film data"
    )
    
    @field_validator('historical_film_id')
    @classmethod
    def validate_historical_film_id(cls, v):
        historical_films = get_historical_films()
        if not any(film.get('id') == v for film in historical_films):
            raise ValueError(f"Historical film with ID '{v}' not found in reference data")
        return v

# --- Enum for reference data types ---
class ReferenceDataType(str, Enum):
    GENRES = "genres"
    CAST_AND_CREW = "cast_and_crew"
    HISTORICAL_FILMS = "historical_films"
    STUDIOS = "studios"

# --- APIRouter Initialization ---
router = APIRouter()

# Initialize the ML model and processor when this module is loaded
try:
    initialize_model_and_processor()
    logger.info("Successfully initialized ML model and processor")
except Exception as e:
    logger.error(f"Failed to initialize ML model and processor: {str(e)}")
    # We don't raise an exception here to allow the API to start,
    # but individual endpoints will fail until the issue is resolved

# --- API Endpoints ---

@router.post("/predict", status_code=status.HTTP_200_OK, response_model_exclude_none=True)
async def predict_route(payload: PredictRequest):
    """
    Generates a box office revenue prediction based on the provided film data.
    
    Returns:
        dict: Prediction output including revenue estimate and confidence interval
    
    Raises:
        422 Validation Error: If the input data fails validation
        500 Internal Server Error: If prediction fails due to server issues
    """
    try:
        logger.info(f"Received prediction request for film: {payload.title}")
        
        # Convert Pydantic model to dict for get_prediction
        raw_input_data = payload.model_dump()
        
        # Additional pre-processing or validation can be done here if needed
        prediction_output = get_prediction(raw_input_data)
        
        logger.info(f"Successfully generated prediction for {payload.title} with revenue: {prediction_output.get('predicted_revenue_usd')}")
        return prediction_output
    
    except FileNotFoundError as e:
        error_msg = f"Internal server error: Missing critical file - {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)
    
    except ValueError as e:
        # For expected value errors that should return 400 Bad Request
        error_msg = f"Invalid input data: {str(e)}"
        logger.warning(error_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
    
    except Exception as e:
        # For unexpected errors
        error_msg = f"An unexpected error occurred during prediction: {str(e)}"
        logger.error(error_msg, exc_info=True)  # Include full traceback in logs
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)

@router.get("/reference_data", status_code=status.HTTP_200_OK)
async def reference_data_route(
    data_type: ReferenceDataType = Query(
        ..., 
        description="Type of reference data to retrieve"
    )
):
    """
    Returns reference data based on the specified type.
    
    Args:
        data_type: Enum specifying which reference data to return
        
    Returns:
        list: The requested reference data
        
    Raises:
        400 Bad Request: If an invalid data type is specified
        500 Internal Server Error: If the reference data cannot be retrieved
    """
    try:
        logger.info(f"Retrieving reference data of type: {data_type}")
        
        if data_type == ReferenceDataType.GENRES:
            result = get_genres()
        elif data_type == ReferenceDataType.CAST_AND_CREW:
            result = get_cast_and_crew()
        elif data_type == ReferenceDataType.HISTORICAL_FILMS:
            result = get_historical_films()
        elif data_type == ReferenceDataType.STUDIOS:
            result = get_studios()
        else:
            # This should never happen due to the Enum validation, but just in case
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Invalid data type: {data_type}. Valid options: {[e.value for e in ReferenceDataType]}"
            )
        
        logger.info(f"Successfully retrieved {len(result)} {data_type} records")
        return result
    
    except Exception as e:
        error_msg = f"Failed to retrieve reference data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)

@router.post("/validate", status_code=status.HTTP_200_OK)
async def validate_route(payload: ValidateRequest):
    """
    Compares a prediction for a historical film against its actual performance.
    Optionally allows overriding specific features of the historical film.
    
    Args:
        payload: ValidateRequest containing historical_film_id and optional override_features
        
    Returns:
        dict: Validation results comparing predicted vs actual revenue
        
    Raises:
        404 Not Found: If the specified historical film is not found
        422 Validation Error: If the input data fails validation
        500 Internal Server Error: If validation fails due to server issues
    """
    try:
        logger.info(f"Validating prediction for historical film ID: {payload.historical_film_id}")
        
        historical_film_id = payload.historical_film_id
        override_features = payload.override_features

        # The validator in ValidateRequest already checks if the film exists, but we need the film data
        historical_films = get_historical_films()
        target_film = next((film for film in historical_films if film.get('id') == historical_film_id), None)

        # This should never happen due to the validator, but just in case
        if not target_film:
            error_msg = f"Historical film with ID '{historical_film_id}' not found"
            logger.warning(error_msg)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)

        film_data_for_prediction = copy.deepcopy(target_film)
        
        # Remove fields that shouldn't be included in prediction input
        fields_to_remove = ['actual_box_office_total_usd', 'actual_box_office_domestic_usd', 
                            'actual_box_office_international_usd', 'id', 'budget_difference']
        for field in fields_to_remove:
            film_data_for_prediction.pop(field, None)
            
        # Apply override features
        film_data_for_prediction.update(override_features)

        # Ensure title is present
        if 'title' not in film_data_for_prediction:
            film_data_for_prediction['title'] = target_film.get('title', 'Unknown Historical Title')
        
        # Validate the resulting data against the PredictRequest model to ensure it has all required fields
        try:
            # This only checks the fields that are in the model - won't remove extra fields
            valid_data = PredictRequest(**film_data_for_prediction).model_dump()
            # We discard valid_data and continue using film_data_for_prediction because
            # get_prediction may use fields not defined in PredictRequest
        except ValidationError as e:
            error_msg = f"Historical film data (with overrides) is invalid: {str(e)}"
            logger.warning(error_msg)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=error_msg)

        # Generate prediction
        prediction_output = get_prediction(film_data_for_prediction)
        
        # Calculate prediction accuracy metrics
        actual_revenue = target_film.get('actual_box_office_total_usd', 0)
        predicted_revenue = prediction_output.get('predicted_revenue_usd', 0)
        
        # Only calculate if both values are available and non-zero
        if actual_revenue and predicted_revenue:
            absolute_error = abs(predicted_revenue - actual_revenue)
            absolute_percentage_error = (absolute_error / actual_revenue) * 100
            
            # Check if actual revenue is within confidence interval
            conf_low = prediction_output.get('confidence_interval_low_usd', 0)
            conf_high = prediction_output.get('confidence_interval_high_usd', float('inf'))
            within_confidence_interval = conf_low <= actual_revenue <= conf_high
        else:
            absolute_error = None
            absolute_percentage_error = None
            within_confidence_interval = None
        
        validation_response = {
            "historical_film_id": historical_film_id,
            "film_title": target_film.get('title'),
            "actual_revenue_usd": actual_revenue,
            "predicted_revenue_usd": predicted_revenue,
            "confidence_interval_low_usd": prediction_output.get('confidence_interval_low_usd'),
            "confidence_interval_high_usd": prediction_output.get('confidence_interval_high_usd'),
            "absolute_error_usd": absolute_error,
            "absolute_percentage_error": absolute_percentage_error,
            "within_confidence_interval": within_confidence_interval,
            "prediction_details": prediction_output,
            "input_data_used_for_prediction": film_data_for_prediction
        }
        
        logger.info(f"Validation completed for film '{target_film.get('title')}' with APE: {absolute_percentage_error}%")
        return validation_response
        
    except HTTPException:
        # Re-raise HTTP exceptions as they already have the correct format
        raise
        
    except FileNotFoundError as e:
        error_msg = f"Internal server error: Missing critical file - {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)
        
    except Exception as e:
        error_msg = f"An unexpected error occurred during validation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)
