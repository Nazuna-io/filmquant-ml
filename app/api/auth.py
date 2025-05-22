"""
Authentication and authorization middleware for FilmQuant ML API.
"""
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi import Security, Depends
import logging
from typing import Optional

from app.config import settings, get

logger = logging.getLogger("filmquant_ml.auth")

# Define API key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(
    api_key_header: Optional[str] = Security(API_KEY_HEADER),
) -> Optional[str]:
    """
    Extracts and validates API key from request header.
    Returns the API key if valid, or None if no key is provided.
    
    This allows endpoints to optionally require authentication.
    """
    if api_key_header:
        return api_key_header
    return None

async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(get_api_key)
) -> None:
    """
    Middleware to verify API key for protected routes.
    Raises HTTPException if authentication fails.
    """
    # Skip authentication in debug mode if configured to do so
    if get("app.debug", False) and get("api.skip_auth_in_debug", False):
        return
    
    # Get the configured API key from settings
    valid_api_key = get("api.api_key", None)
    
    # Get the path to check if it's a protected route
    path = request.url.path
    
    # Skip authentication for unprotected routes
    unprotected_routes = [
        "/docs", 
        "/redoc", 
        "/openapi.json",
        "/gradio"  # Gradio UI doesn't need API key
    ]
    
    # Also consider routes that don't start with the API prefix as unprotected
    api_prefix = get("api.prefix", "/api/v1")
    if not path.startswith(api_prefix) or any(path.startswith(route) for route in unprotected_routes):
        return
    
    # If we reach here, this is a protected route requiring authentication
    if not valid_api_key:
        logger.error("API authentication is enabled but no API key is configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server authentication configuration error",
        )
    
    if not api_key:
        logger.warning(f"Unauthorized access attempt to {path} - No API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
        
    if api_key != valid_api_key:
        logger.warning(f"Unauthorized access attempt to {path} - Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
