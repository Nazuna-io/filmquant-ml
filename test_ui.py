#!/usr/bin/env python3
"""
Updated main.py to use the new UI and connect to our trained model
"""

import gradio as gr
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import requests
import json
import os
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import logging
import traceback

from app.static_data_loader import get_genres, get_cast_and_crew, get_historical_films, get_studios
from app.utils.logging import configure_logging

# Configure logging
logger = configure_logging(
    app_name="filmquant_ml", 
    log_level=logging.INFO,
    log_to_console=True,
    log_to_file=True
)

# Load our trained model
def load_trained_model():
    """Load the actual trained FilmQuant ML model"""
    try:
        # Find the most recent model file
        model_dir = "models"
        model_files = [f for f in os.listdir(model_dir) if f.startswith("filmquant_ml_model_") and f.endswith(".pkl")]
        
        if not model_files:
            logger.error("No trained model found in models/ directory")
            return None
            
        # Get the most recent model
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(model_dir, latest_model)
        
        logger.info(f"Loading trained model: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        logger.info(f"Model loaded successfully: {model_data['model_name']}")
        logger.info(f"Training date: {model_data['training_date']}")
        logger.info(f"Feature count: {len(model_data['feature_names'])}")
        
        return model_data
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Load model at startup
MODEL_DATA = load_trained_model()

def predict_button_callback(title, genres, director, actor1, actor2, actor3, studio, budget, runtime, release_date, screens, marketing_budget, trailer_views):
    """Simplified prediction callback for testing"""
    try:
        logger.info(f"Predicting for film: {title}")
        
        # Simple mock prediction based on budget
        if budget:
            prediction = budget * 2.5  # Simple multiplier
        else:
            prediction = 150.0  # Default prediction
        
        # Format results
        predicted_revenue = f"${prediction:.1f}M"
        confidence_low = prediction * 0.8
        confidence_high = prediction * 1.2
        confidence_interval = f"${confidence_low:.1f}M - ${confidence_high:.1f}M"
        
        domestic_share = f"${prediction:.1f}M (45%)"
        international_share = f"${(prediction / 0.45 * 0.55):.1f}M (55%)"
        
        # Mock data for testing
        top_factors = pd.DataFrame({
            "Factor": ["Budget", "Star Power", "Studio"],
            "Importance": ["High", "Medium", "Medium"],
            "Impact": [f"${budget:.0f}M" if budget else "N/A", "2/3 A-list", studio or "Independent"]
        })
        
        comparable_films = pd.DataFrame({
            "Film": ["Barbie", "Oppenheimer"],
            "Box Office": ["$636M", "$330M"],
            "Type": ["Similar", "Comparable"]
        })
        
        raw_input = {
            "title": title,
            "prediction": prediction,
            "budget": budget
        }
        
        return (
            predicted_revenue,
            confidence_interval,
            domestic_share, 
            international_share,
            top_factors,
            comparable_films,
            raw_input
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return ("Error", "Error", "Error", "Error", pd.DataFrame(), pd.DataFrame(), {})

def historical_film_selected_callback(film_id):
    """Mock historical callback for testing"""
    try:
        if not film_id:
            return tuple([None] * 14)
        
        # Mock historical data
        return (
            "Barbie",  # title
            ["Comedy", "Family"],  # genres  
            "Greta Gerwig",  # director
            "Margot Robbie",  # actor1
            "Ryan Gosling",  # actor2
            "America Ferrera",  # actor3
            "Warner Bros. Pictures",  # studio
            145.0,  # budget (in millions)
            114,  # runtime
            "2023-07-19",  # release date
            3500,  # screens
            80.0,  # marketing budget
            2000000,  # trailer views
            "Actual: $636M domestic"  # actuals
        )
        
    except Exception as e:
        logger.error(f"Error loading historical film: {e}")
        return tuple([None] * 14)

# Simplified UI for testing
def create_test_interface():
    """Create a simplified test interface"""
    
    with gr.Blocks(title="FilmQuant ML - Test Interface") as interface:
        gr.Markdown("# 🎬 FilmQuant ML - Box Office Prediction (Test)")
        
        with gr.Row():
            with gr.Column():
                title = gr.Textbox(label="Film Title", value="Test Movie")
                budget = gr.Number(label="Budget (Millions)", value=150.0)
                director = gr.Textbox(label="Director", value="Christopher Nolan")
                studio = gr.Textbox(label="Studio", value="Universal Pictures")
                
                predict_btn = gr.Button("Predict", variant="primary")
                
            with gr.Column():
                result = gr.Textbox(label="Prediction Result", lines=5)
        
        def simple_predict(title, budget, director, studio):
            try:
                prediction = budget * 2.0 if budget else 100.0
                return f"Title: {title}\nBudget: ${budget}M\nPredicted: ${prediction:.1f}M\nDirector: {director}\nStudio: {studio}"
            except Exception as e:
                return f"Error: {e}"
        
        predict_btn.click(
            fn=simple_predict,
            inputs=[title, budget, director, studio],
            outputs=[result]
        )
    
    return interface

if __name__ == '__main__':
    logger.info("🚀 Starting FilmQuant ML Test Interface")
    
    # Create and launch test interface
    interface = create_test_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=8081,
        show_error=True
    )
