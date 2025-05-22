import gradio as gr
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn # For running the app
import requests # For Gradio to call Flask API if needed, or direct function calls
import json
import os
import pandas as pd # For displaying tables in Gradio
from datetime import datetime
import logging
import traceback

from app.api.routes import router as api_router # Import the FastAPI router
from app.api.auth import verify_api_key  # Import authentication middleware
from app.static_data_loader import get_genres, get_cast_and_crew, get_historical_films, get_studios
from app.ml.prediction import get_prediction # We can call this directly from Gradio callbacks
from app.utils.logging import configure_logging
from app.config import settings, get

# Configure logging
logger = configure_logging(
    app_name="filmquant_ml", 
    log_level=logging.INFO,
    log_to_console=True,
    log_to_file=True
)

# API blueprint will be handled differently with FastAPI
app = FastAPI(
    title="Box Office Revenue Predictor API",
    description="API for predicting box office revenue for films",
    version="1.0.0"
)

# Add CORS middleware with more secure settings based on environment
cors_origins = ["*"] if get("app.debug", False) else [
    "https://filmquant-ml.example.com",  # Main production site
    "https://app.filmquant-ml.example.com"  # Alternative production domain
]

logger.info(f"CORS configuration: allowing origins {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # More restrictive HTTP methods
    allow_headers=["Content-Type", "Authorization"],  # Only necessary headers
)

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."}
    )

# Include the API routes with authentication
app.include_router(
    api_router, 
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)]  # Apply authentication to all API routes
)

# --- Gradio Interface Definition ---

# Gradio callback functions
def predict_button_callback(title, genre_ids, director_id, cast_id1, cast_id2, cast_id3, studio_id, 
                            budget_usd, runtime_minutes, release_date, screens_opening_day, 
                            marketing_budget_est_usd, trailer_views_prerelease):
    """Callback for the 'Predict Box Office Revenue' button."""
    logger.info(f"Prediction requested for film: '{title}'")
    logger.debug(f"release_date is '{release_date}', type is {type(release_date)}")
    
    processed_release_date = None
    if release_date:
        if hasattr(release_date, 'strftime'): # Check if it's a date/datetime object
            processed_release_date = release_date.strftime('%Y-%m-%d')
        else: # Assume it's already a string (e.g. from direct Textbox input)
            processed_release_date = str(release_date) 
            # Validate date format
            try:
                datetime.strptime(processed_release_date, '%Y-%m-%d')
            except ValueError:
                error_message = f"Invalid date format: '{processed_release_date}'. Please use YYYY-MM-DD format."
                logger.warning(error_message)
                return error_message, None, None, None, None, None, None

    # Ensure we don't have None values for numerical inputs, default to 0 instead
    budget_usd = budget_usd if budget_usd is not None else 0
    runtime_minutes = runtime_minutes if runtime_minutes is not None else 0
    screens_opening_day = screens_opening_day if screens_opening_day is not None else 0

    # Validate required fields
    if not title:
        error_message = "Film title is required."
        logger.warning(error_message)
        return error_message, None, None, None, None, None, None
    
    if not genre_ids or (isinstance(genre_ids, list) and len(genre_ids) == 0):
        error_message = "At least one genre must be selected."
        logger.warning(error_message)
        return error_message, None, None, None, None, None, None

    raw_input_data = {
        "title": title if title is not None else "",
        "genre_ids": genre_ids if isinstance(genre_ids, list) else [genre_ids] if genre_ids else [],
        "director_id": director_id,
        "cast_ids": [c_id for c_id in [cast_id1, cast_id2, cast_id3] if c_id],
        "studio_id": studio_id if studio_id else None, # Ensure None if empty string
        "budget_usd": budget_usd,
        "runtime_minutes": runtime_minutes,
        "release_date": processed_release_date,
        "screens_opening_day": screens_opening_day,
        "trailer_views_prerelease": trailer_views_prerelease if trailer_views_prerelease else 0
    }
    
    # Only include marketing budget if provided
    if marketing_budget_est_usd is not None and marketing_budget_est_usd > 0:
        raw_input_data["marketing_budget_est_usd"] = marketing_budget_est_usd

    logger.debug(f"Processed input data: {json.dumps(raw_input_data)}")

    try:
        prediction_output = get_prediction(raw_input_data)
        logger.info(f"Prediction successful for '{title}' with revenue: ${prediction_output.get('predicted_revenue_usd', 0):,.0f}")
    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        logger.error(error_message, exc_info=True)
        # Return empty/error values for all outputs
        return error_message, None, None, None, None, None, None

    # Parse output for Gradio components
    predicted_revenue = f"${prediction_output.get('predicted_revenue_usd', 0):,.0f}"
    confidence_interval = (
        f"Low: ${prediction_output.get('confidence_interval_low_usd', 0):,.0f}\n" 
        f"High: ${prediction_output.get('confidence_interval_high_usd', 0):,.0f}"
    )
    domestic_share = f"{prediction_output.get('domestic_revenue_share_percent', 0)*100:.1f}%"
    international_share = f"{prediction_output.get('international_revenue_share_percent', 0)*100:.1f}%"
    
    top_factors_data = prediction_output.get('top_factors', [])
    top_factors_df = pd.DataFrame(top_factors_data) 
    if not top_factors_df.empty:
        top_factors_df['importance_score'] = top_factors_df['importance_score'].round(3)

    comparable_films_data = prediction_output.get('comparable_films', [])    
    for film in comparable_films_data:
        if 'predicted_or_actual_revenue_usd' in film and isinstance(film['predicted_or_actual_revenue_usd'], (int, float)):
            film['predicted_or_actual_revenue_usd'] = f"${film['predicted_or_actual_revenue_usd']:,.0f}"
    comparable_films_df = pd.DataFrame(comparable_films_data)

    raw_input_display = json.dumps(prediction_output.get('raw_input_data', {}), indent=2)

    return (
        predicted_revenue, confidence_interval, domestic_share, international_share,
        top_factors_df, comparable_films_df, raw_input_display
    )

def historical_film_selected_callback(historical_film_id):
    """Callback when a historical film is selected. Pre-fills input fields."""
    if not historical_film_id:
        logger.debug("No historical film selected, returning default values")
        # Return defaults or clear fields
        return "", [], None, None, None, None, None, 0, 0, None, 0, 0, 0, "No film selected."

    logger.info(f"Historical film selected: ID {historical_film_id}")
    film_detail = next((f for f in get_historical_films() if f['id'] == historical_film_id), None)
    if not film_detail:
        logger.warning(f"Historical film with ID {historical_film_id} not found in reference data")
        return "", [], None, None, None, None, None, 0, 0, None, 0, 0, 0, f"Film ID {historical_film_id} not found."
    
    cast_ids = film_detail.get('cast_ids', [])
    
    # Get marketing budget, which might be null
    marketing_budget = film_detail.get('marketing_budget_usd_estimated')
    
    # For displaying actuals if available
    actuals_info = (
        f"Actual Total: ${film_detail.get('actual_box_office_total_usd', 0):,.0f}\n" 
        f"Actual Domestic: ${film_detail.get('actual_box_office_domestic_usd', 0):,.0f}\n"
        f"Actual International: ${film_detail.get('actual_box_office_international_usd', 0):,.0f}"
    )
    
    logger.info(f"Historical film '{film_detail.get('title')}' data loaded successfully")
    return (
        film_detail.get('title', ''),
        film_detail.get('genre_ids', []),
        film_detail.get('director_id'),
        cast_ids[0] if len(cast_ids) > 0 else None,
        cast_ids[1] if len(cast_ids) > 1 else None,
        cast_ids[2] if len(cast_ids) > 2 else None,
        film_detail.get('studio_id'),
        film_detail.get('budget_usd', 0),
        film_detail.get('runtime_minutes', 0),
        datetime.strptime(film_detail['release_date'], '%Y-%m-%d') if film_detail.get('release_date') else None,
        film_detail.get('screens_opening_day', 0),
        marketing_budget,
        film_detail.get('trailer_views_prerelease', 0),
        actuals_info
    )

# Import and create the Apple-inspired UI
from app.theme import create_filmquant_ml_interface

# Pack the callback functions to pass to the UI creator
callbacks = {
    "predict_button_callback": predict_button_callback,
    "historical_film_selected_callback": historical_film_selected_callback
}

# Create the Gradio interface with the Apple-inspired UI
filmquant_ml_interface = create_filmquant_ml_interface(callbacks)

# Mount Gradio app to the FastAPI app instance
app = gr.mount_gradio_app(app, filmquant_ml_interface, path="/gradio")

if __name__ == '__main__':
    # This block is for local development if you run `python -m app.main`
    # It will start a Uvicorn development server.
    # The app object here is the FastAPI instance with Gradio mounted.
    logger.info("Starting FilmQuant ML application in development mode")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8081, reload=True)
