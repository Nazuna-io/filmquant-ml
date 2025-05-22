#!/bin/bash
# Script to implement the Apple-inspired theme in your BORP project

# Set working directory to the project root
cd /home/todd/borp

# Create the theme.py file in the app directory
echo "Creating the theme file in app/theme.py..."
mkdir -p app
cat > app/theme.py << 'EOF'
"""
Apple-inspired theme for the BORP Gradio UI
This file contains a custom Gradio theme that mimics Apple's design language,
along with an implementation example that can be used to replace the existing UI.

Usage:
1. Save this file as app/theme.py
2. Import and use the apple_theme in app/main.py:
   
   from app.theme import apple_theme, create_borp_interface
   
   # Option 1: Use only the theme with your existing UI structure
   with gr.Blocks(theme=apple_theme, title="Box Office Revenue Predictor (BORP)") as borp_interface:
       # Your existing UI code here
       
   # Option 2: Use the complete enhanced UI implementation
   borp_interface = create_borp_interface()
"""

import gradio as gr

def create_apple_theme():
    """
    Creates a custom Gradio theme that mimics Apple's design language.
    """
    apple_theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f0f9ff",
            c100="#e0f2fe",
            c200="#bae6fd",
            c300="#7dd3fc",
            c400="#38bdf8",
            c500="#0ea5e9",
            c600="#0284c7",
            c700="#0369a1",
            c800="#075985",
            c900="#0c4a6e",
            c950="#082f49",
        ),
        secondary_hue=gr.themes.Color(
            c50="#eff6ff",
            c100="#dbeafe",
            c200="#bfdbfe",
            c300="#93c5fd",
            c400="#60a5fa",
            c500="#3b82f6",
            c600="#2563eb",
            c700="#1d4ed8",
            c800="#1e40af",
            c900="#1e3a8a",
            c950="#172554",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f8fafc",
            c100="#f1f5f9",
            c200="#e2e8f0",
            c300="#cbd5e1",
            c400="#94a3b8",
            c500="#64748b",
            c600="#475569",
            c700="#334155",
            c800="#1e293b",
            c900="#0f172a",
            c950="#020617",
        ),
        spacing_size=gr.themes.sizes.spacing_md,
        radius_size=gr.themes.sizes.radius_lg,
        text_size=gr.themes.sizes.text_md,
    )
    
    # Apply specific Apple-inspired styling
    return apple_theme.set(
        # General elements
        body_background_fill="rgb(245, 245, 247)",
        body_background_fill_dark="rgb(28, 28, 30)",
        body_text_color="rgb(29, 29, 31)",
        body_text_color_dark="rgb(245, 245, 247)",
        body_text_size="15px",
        body_text_weight="400",
        
        # Container elements
        block_background_fill="rgb(255, 255, 255)",
        block_background_fill_dark="rgb(44, 44, 46)",
        block_border_color="rgb(210, 210, 215)",
        block_border_color_dark="rgb(68, 68, 70)",
        block_border_width="1px",
        block_label_background_fill="rgb(255, 255, 255)",
        block_label_background_fill_dark="rgb(44, 44, 46)",
        block_label_border_width="0px",
        block_label_text_color="rgb(29, 29, 31)",
        block_label_text_color_dark="rgb(245, 245, 247)",
        block_label_text_size="14px",
        block_label_text_weight="500",
        block_title_text_color="rgb(29, 29, 31)",
        block_title_text_color_dark="rgb(255, 255, 255)",
        block_title_text_size="20px",
        block_title_text_weight="600",
        block_radius="12px",
        block_shadow="rgba(0, 0, 0, 0.05) 0px 1px 2px",
        
        # Form elements
        checkbox_background_color="rgb(255, 255, 255)",
        checkbox_background_color_dark="rgb(28, 28, 30)",
        checkbox_background_color_selected="rgb(0, 122, 255)",
        checkbox_background_color_selected_dark="rgb(10, 132, 255)",
        checkbox_border_color="rgb(174, 174, 178)",
        checkbox_border_color_dark="rgb(99, 99, 102)",
        checkbox_border_color_focus="rgb(0, 122, 255)",
        checkbox_border_color_focus_dark="rgb(10, 132, 255)",
        checkbox_border_width="1px",
        checkbox_label_text_size="14px",
        checkbox_label_text_weight="400",
        checkbox_label_text_color="rgb(29, 29, 31)",
        checkbox_label_text_color_dark="rgb(245, 245, 247)",
        checkbox_size="16px",
        
        # Buttons
        button_primary_background_fill="rgb(0, 122, 255)",
        button_primary_background_fill_dark="rgb(10, 132, 255)",
        button_primary_background_fill_hover="rgb(10, 132, 255)",
        button_primary_background_fill_hover_dark="rgb(64, 156, 255)",
        button_primary_border_width="0px",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        button_primary_text_weight="500",
        
        button_secondary_background_fill="rgb(242, 242, 247)",
        button_secondary_background_fill_dark="rgb(58, 58, 60)",
        button_secondary_background_fill_hover="rgb(229, 229, 234)",
        button_secondary_background_fill_hover_dark="rgb(72, 72, 74)",
        button_secondary_border_color="rgb(229, 229, 234)",
        button_secondary_border_color_dark="rgb(72, 72, 74)",
        button_secondary_border_width="1px",
        button_secondary_text_color="rgb(29, 29, 31)",
        button_secondary_text_color_dark="rgb(245, 245, 247)",
        button_secondary_text_weight="500",
        
        # Input fields
        input_background_fill="rgb(255, 255, 255)",
        input_background_fill_dark="rgb(44, 44, 46)",
        input_border_color="rgb(210, 210, 215)",
        input_border_color_dark="rgb(68, 68, 70)",
        input_border_color_focus="rgb(0, 122, 255)",
        input_border_color_focus_dark="rgb(10, 132, 255)",
        input_border_width="1px",
        input_placeholder_color="rgb(142, 142, 147)",
        input_placeholder_color_dark="rgb(142, 142, 147)",
        input_radius="8px",
        input_shadow="rgba(0, 0, 0, 0.05) 0px 1px 2px",
        input_shadow_focus="rgba(0, 122, 255, 0.2) 0px 0px 0px 3px",
        input_text_color="rgb(29, 29, 31)",
        input_text_color_dark="rgb(245, 245, 247)",
        input_text_weight="400",
        
        # Dropdowns
        dropdown_background_fill="rgb(255, 255, 255)",
        dropdown_background_fill_dark="rgb(44, 44, 46)",
        dropdown_border_color="rgb(210, 210, 215)",
        dropdown_border_color_dark="rgb(68, 68, 70)",
        dropdown_border_width="1px", 
        dropdown_radius="8px",
        dropdown_text_color="rgb(29, 29, 31)",
        dropdown_text_color_dark="rgb(245, 245, 247)",
        
        # Sliders
        slider_color="rgb(0, 122, 255)",
        slider_color_dark="rgb(10, 132, 255)",
        
        # Tables
        table_border_color="rgb(210, 210, 215)",
        table_border_color_dark="rgb(68, 68, 70)",
        table_even_background_fill="rgb(246, 246, 248)",
        table_even_background_fill_dark="rgb(44, 44, 46)",
        table_odd_background_fill="rgb(255, 255, 255)",
        table_odd_background_fill_dark="rgb(28, 28, 30)",
        table_radius="8px",
        
        # Tabs
        tab_selected_background_fill="rgb(0, 122, 255)",
        tab_selected_background_fill_dark="rgb(10, 132, 255)",
        tab_selected_text_color="white",
        tab_selected_text_color_dark="white",
    )

def create_borp_interface():
    """
    Creates an enhanced Gradio interface for BORP with Apple-inspired design elements.
    This function replicates the existing UI functionality but with improved visuals.
    
    Returns:
        gr.Blocks: The constructed Gradio interface
    """
    from app.static_data_loader import get_genres, get_cast_and_crew, get_historical_films, get_studios
    from app.main import predict_button_callback, historical_film_selected_callback
    
    # Load reference data for dropdowns
    def get_choices_from_data(data_list, name_key='name', id_key='id'):
        return [(item[name_key], item[id_key]) for item in data_list]

    genre_choices = get_choices_from_data(get_genres())
    personnel_choices = get_choices_from_data(get_cast_and_crew())
    director_choices = [(name, id) for name, id in personnel_choices if next((p for p in get_cast_and_crew() if p['id'] == id and p['role'] == 'director'), None)]
    actor_choices = [(name, id) for name, id in personnel_choices if next((p for p in get_cast_and_crew() if p['id'] == id and p['role'] == 'actor'), None)]
    studio_choices = get_choices_from_data(get_studios())
    historical_film_choices = get_choices_from_data(get_historical_films(), name_key='title')
    
    # Create the enhanced interface
    with gr.Blocks(theme=apple_theme, title="Box Office Revenue Predictor (BORP)") as borp_interface:
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="font-size: 32px; font-weight: 700; margin-bottom: 10px;">Box Office Revenue Predictor</h1>
                <p style="font-size: 16px; color: #64748b;">Predict film revenue using machine learning</p>
            </div>
            """
        )

        with gr.Tabs() as tabs:
            with gr.TabItem("Predict New Film", id=0):
                with gr.Row().style(equal_height=True):
                    with gr.Column(scale=1):
                        with gr.Box():
                            gr.Markdown(
                                """
                                <div style="margin-bottom: 15px;">
                                    <h3 style="font-size: 20px; font-weight: 600; margin-bottom: 10px;">Film Details</h3>
                                    <p style="font-size: 14px; color: #64748b;">Enter the details below to predict box office revenue</p>
                                </div>
                                """
                            )
                            
                            title_input = gr.Textbox(
                                label="Film Title", 
                                placeholder="e.g., My Awesome Film",
                                elem_id="film-title"
                            )
                            
                            genre_input = gr.Dropdown(
                                label="Genre(s)",
                                choices=genre_choices,
                                multiselect=True,
                                elem_id="film-genres"
                            )
                            
                            director_input = gr.Dropdown(
                                label="Director",
                                choices=director_choices,
                                allow_custom_value=False,
                                elem_id="film-director"
                            )
                            
                            with gr.Row(equal_height=True):
                                cast_input1 = gr.Dropdown(
                                    label="Lead Actor 1",
                                    choices=actor_choices,
                                    allow_custom_value=False,
                                    elem_id="film-actor1"
                                )
                                cast_input2 = gr.Dropdown(
                                    label="Lead Actor 2",
                                    choices=actor_choices,
                                    allow_custom_value=False,
                                    elem_id="film-actor2" 
                                )
                                cast_input3 = gr.Dropdown(
                                    label="Lead Actor 3",
                                    choices=actor_choices,
                                    allow_custom_value=False,
                                    elem_id="film-actor3"
                                )
                            
                            studio_input = gr.Dropdown(
                                label="Studio",
                                choices=studio_choices,
                                allow_custom_value=False,
                                elem_id="film-studio"
                            )
                            
                            with gr.Row(equal_height=True):
                                budget_input = gr.Number(
                                    label="Budget (USD)",
                                    value=50000000,
                                    step=1000000,
                                    elem_id="film-budget"
                                )
                                runtime_input = gr.Number(
                                    label="Runtime (minutes)",
                                    value=120,
                                    step=1,
                                    elem_id="film-runtime"
                                )
                            
                            release_date_input = gr.Textbox(
                                label="Release Date",
                                value="2025-07-04",
                                placeholder="YYYY-MM-DD",
                                elem_id="film-release-date"
                            )
                            
                            with gr.Row(equal_height=True):
                                screens_input = gr.Number(
                                    label="Screens Opening Day",
                                    value=3000,
                                    step=100,
                                    elem_id="film-screens"
                                )
                                marketing_budget_input = gr.Number(
                                    label="Marketing Budget (USD, Optional)",
                                    value=None,
                                    step=1000000,
                                    elem_id="film-marketing"
                                )
                            
                            trailer_views_input = gr.Number(
                                label="Trailer Views Prerelease",
                                value=10000000,
                                step=100000,
                                elem_id="film-trailer-views"
                            )
                            
                            predict_button = gr.Button(
                                "Predict Box Office Revenue",
                                variant="primary",
                                elem_id="predict-button",
                                size="lg"
                            ).style(full_width=True)

                    with gr.Column(scale=2):
                        with gr.Box():
                            gr.Markdown(
                                """
                                <div style="margin-bottom: 15px;">
                                    <h3 style="font-size: 20px; font-weight: 600; margin-bottom: 10px;">Prediction Results</h3>
                                </div>
                                """
                            )
                            
                            with gr.Row():
                                predicted_revenue_output = gr.Textbox(
                                    label="Predicted Total Revenue",
                                    elem_id="predicted-revenue",
                                    interactive=False,
                                    container=False,
                                    scale=2
                                ).style(
                                    container=True,
                                    rounded=(True, True, True, True),
                                    padding=True,
                                    bgcolor="*primary_50",
                                    border_color="*primary_300",
                                    color="*primary_800"
                                )
                                
                            with gr.Row():
                                confidence_interval_output = gr.Textbox(
                                    label="Confidence Interval (90%)",
                                    lines=2,
                                    elem_id="confidence-interval",
                                    interactive=False,
                                    container=False,
                                    scale=1
                                ).style(
                                    container=True,
                                    rounded=(True, True, True, True),
                                    padding=True
                                )
                                
                                with gr.Column(scale=1):
                                    domestic_share_output = gr.Textbox(
                                        label="Est. Domestic Share",
                                        elem_id="domestic-share",
                                        interactive=False,
                                        container=False
                                    ).style(
                                        container=True, 
                                        rounded=(True, True, True, True),
                                        padding=True
                                    )
                                    
                                    international_share_output = gr.Textbox(
                                        label="Est. International Share",
                                        elem_id="international-share",
                                        interactive=False,
                                        container=False
                                    ).style(
                                        container=True,
                                        rounded=(True, True, True, True),
                                        padding=True
                                    )
                            
                            gr.Markdown(
                                """
                                <div style="margin-top: 20px; margin-bottom: 10px;">
                                    <h4 style="font-size: 16px; font-weight: 600;">Top Influencing Factors</h4>
                                </div>
                                """
                            )
                            
                            top_factors_output = gr.DataFrame(
                                headers=["feature", "importance_score", "value_contributed"],
                                col_count=(3, "fixed"),
                                interactive=False,
                                wrap=True,
                                elem_id="top-factors"
                            )
                            
                            gr.Markdown(
                                """
                                <div style="margin-top: 20px; margin-bottom: 10px;">
                                    <h4 style="font-size: 16px; font-weight: 600;">Comparable Historical Films</h4>
                                </div>
                                """
                            )
                            
                            comparable_films_output = gr.DataFrame(
                                headers=["title", "predicted_or_actual_revenue_usd", "type"],
                                col_count=(3,"fixed"),
                                interactive=False,
                                wrap=True,
                                elem_id="comparable-films"
                            )
                            
                            with gr.Accordion("Raw Input (for verification)", open=False):
                                raw_input_display_output = gr.JSON(
                                    label="Raw Input to Model (Post-Processed)",
                                    elem_id="raw-input"
                                )
            
            with gr.TabItem("Historical Validation Mode", id=1):
                with gr.Box():
                    gr.Markdown(
                        """
                        <div style="margin-bottom: 15px;">
                            <h3 style="font-size: 20px; font-weight: 600; margin-bottom: 10px;">Historical Validation</h3>
                            <p style="font-size: 14px; color: #64748b;">Select a historical film to see its actuals and pre-fill data for a new prediction</p>
                        </div>
                        """
                    )
                    
                    historical_film_dropdown = gr.Dropdown(
                        label="Select Historical Film",
                        choices=historical_film_choices,
                        allow_custom_value=False,
                        elem_id="historical-film"
                    )
                    
                    historical_actuals_output = gr.Textbox(
                        label="Historical Actual Revenue",
                        lines=3,
                        interactive=False,
                        elem_id="historical-actuals"
                    ).style(
                        container=True,
                        rounded=(True, True, True, True),
                        padding=True,
                        bgcolor="*neutral_50",
                        border_color="*neutral_300"
                    )
                    
                    gr.Markdown(
                        """
                        <div style="margin-top: 15px;">
                            <p>The selected film's data will populate the 'Predict New Film' tab fields. You can then modify and predict.</p>
                        </div>
                        """
                    )

        # Wire up callbacks - using the same callbacks as the original interface
        predict_button.click(
            fn=predict_button_callback,
            inputs=[
                title_input, genre_input, director_input, cast_input1, cast_input2, cast_input3, 
                studio_input, budget_input, runtime_input, release_date_input, screens_input,
                marketing_budget_input, trailer_views_input
            ],
            outputs=[
                predicted_revenue_output, confidence_interval_output, 
                domestic_share_output, international_share_output,
                top_factors_output, comparable_films_output, raw_input_display_output
            ]
        )

        historical_film_dropdown.change(
            fn=historical_film_selected_callback,
            inputs=[historical_film_dropdown],
            outputs=[
                title_input, genre_input, director_input, cast_input1, cast_input2, cast_input3, studio_input,
                budget_input, runtime_input, release_date_input, screens_input,
                marketing_budget_input, trailer_views_input, historical_actuals_output
            ]
        )
        
    return borp_interface

# Create and export the apple_theme instance for direct import
apple_theme = create_apple_theme()

# If this module is run directly, show a demo of the theme
if __name__ == "__main__":
    # Simple demo interface to show the theme
    with gr.Blocks(theme=apple_theme, title="Apple Theme Demo") as demo:
        gr.Markdown("# Apple Theme Demo")
        gr.Markdown("This is a demonstration of the Apple-inspired Gradio theme")
        
        with gr.Row():
            with gr.Column():
                gr.Textbox(label="Text Input Example", placeholder="Enter some text...")
                gr.Dropdown(label="Dropdown Example", choices=["Option 1", "Option 2", "Option 3"])
                gr.Checkbox(label="Checkbox Example")
                gr.Slider(label="Slider Example", minimum=0, maximum=100, value=50)
                gr.Button("Primary Button", variant="primary")
                gr.Button("Secondary Button")
            
            with gr.Column():
                gr.Markdown("## Results")
                gr.Textbox(label="Output Example", value="This is some output text")
                with gr.Accordion("Details", open=True):
                    gr.JSON({"key": "value", "nested": {"data": [1, 2, 3]}})
        
    demo.launch()
EOF

echo "Theme file created successfully!"

# Create a backup of the main.py file
echo "Creating backup of the original main.py file..."
cp app/main.py app/main.py.backup

# Now we'll modify the main.py file to use the fully enhanced UI
echo "Modifying main.py to use the enhanced Apple UI..."
cat > app/main.py.new << 'EOF'
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
from app.theme import create_borp_interface  # Import the enhanced UI function

# Configure logging
logger = configure_logging(
    app_name="borp", 
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
    "https://borp.example.com",  # Main production site
    "https://app.borp.example.com"  # Alternative production domain
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
# Create the enhanced UI with Apple-inspired design
borp_interface = create_borp_interface()

# Mount Gradio app to the FastAPI app instance
app = gr.mount_gradio_app(app, borp_interface, path="/gradio")

if __name__ == '__main__':
    # This block is for local development if you run `python -m app.main`
    # It will start a Uvicorn development server.
    # The app object here is the FastAPI instance with Gradio mounted.
    logger.info("Starting BORP application in development mode")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8081, reload=True)
EOF

# Replace the main.py file with the new version
mv app/main.py.new app/main.py

echo "Implementation complete!"
echo ""
echo "To test the new Apple-inspired UI, run your application as usual:"
echo "python -m app.main"
echo ""
echo "If you want to revert changes:"
echo "mv app/main.py.backup app/main.py"
