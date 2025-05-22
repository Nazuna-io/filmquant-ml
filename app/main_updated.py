usd', 0)
        international = film_detail.get('actual_box_office_international_usd', 0)
        total = domestic + international
        
        actuals_info = (
            f"💰 Actual Performance:\n"
            f"Domestic: ${domestic:.1f}M\n" 
            f"International: ${international:.1f}M\n"
            f"Total: ${total:.1f}M"
        )
        
        logger.info(f"Historical film '{film_detail.get('title')}' loaded")
        
        return (
            film_detail.get('title', ''),
            film_detail.get('genre_ids', []),
            film_detail.get('director_id'),
            actors[0] if len(actors) > 0 else None,
            actors[1] if len(actors) > 1 else None,
            actors[2] if len(actors) > 2 else None,
            film_detail.get('studio_id'),
            budget,
            film_detail.get('runtime_minutes', 0),
            film_detail.get('release_date', ''),
            film_detail.get('screens_opening_day', 0),
            marketing_budget,
            film_detail.get('trailer_views_prerelease', 0),
            actuals_info
        )
        
    except Exception as e:
        logger.error(f"Error loading historical film: {e}")
        return tuple([None] * 14)

# Create FastAPI app
app = FastAPI(
    title="FilmQuant ML API",
    description="AI-powered box office prediction system",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])

# Health check endpoint
@app.get("/health")
async def health_check():
    model_status = "loaded" if MODEL_DATA else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Import our updated UI
from app.updated_ui import create_updated_filmquant_ml_interface

# Pack callbacks
callbacks = {
    "predict_button_callback": predict_button_callback,
    "historical_film_selected_callback": historical_film_selected_callback
}

# Create the Gradio interface
filmquant_ml_interface = create_updated_filmquant_ml_interface(callbacks)

# Mount Gradio app
app = gr.mount_gradio_app(app, filmquant_ml_interface, path="/")

if __name__ == '__main__':
    logger.info("🚀 Starting FilmQuant ML v2.0 with trained model")
    logger.info(f"Model status: {'✅ Loaded' if MODEL_DATA else '❌ Not loaded'}")
    
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8081, 
        reload=True,
        log_level="info"
    )
