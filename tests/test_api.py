import pytest
import pytest_asyncio  # For async fixtures
from httpx import AsyncClient

from app.main import app  # Import your FastAPI app instance

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client():
    """
    Async fixture to create an httpx.AsyncClient for testing the FastAPI app.
    The base_url should match where your app is served if testing an external server.
    For in-process testing with app=app, it's good practice but client.post(\"/api/v1/predict\") works.
    """
    # Ensure the FastAPI app's lifecycle events (startup/shutdown) are handled during tests.
    # For this, httpx.AsyncClient needs the app instance.
    async with AsyncClient(
        app=app, base_url="http://127.0.0.1:8081"
    ) as ac:  # Port matches main.py
        yield ac


async def test_predict_endpoint_date_error(client: AsyncClient):
    """
    Test the /api/v1/predict endpoint specifically with a date string
    that has been causing 'strftime' errors.
    """
    payload = {
        "title": "API Test Future Date",
        "genre_ids": [
            "g001"
        ],  # Placeholder: Ensure this ID exists in sample_genres.json
        "director_id": "p001",  # Placeholder: Ensure this ID exists as a director in sample_cast_and_crew.json
        "cast_ids": [
            "p002"
        ],  # Placeholder: Ensure this ID exists as an actor in sample_cast_and_crew.json
        "studio_id": "s001",  # Placeholder: Ensure this ID exists in sample_studios.json
        "budget_usd": 100000000,
        "runtime_minutes": 120,
        "release_date": "2025-05-23",  # The problematic date
        "screens_opening_day": 3000,
        "marketing_budget_est_usd": 50000000,
        "trailer_views_prerelease": 20000000,
    }

    print(f"\\nSending payload to /api/v1/predict: {payload}")

    # We expect this to potentially fail internally. Pytest will show the traceback.
    response = await client.post("/api/v1/predict", json=payload)

    # Print response details for debugging
    print(f"Response status: {response.status_code}")
    try:
        print(f"Response JSON: {response.json()}")
    except Exception as e:
        # If response is not JSON, print text and the error
        print(f"Response text (not JSON): {response.text}")
        print(f"Error decoding JSON response: {e}")

    # We are primarily interested in the traceback that pytest will provide on failure.
    # If the strftime error occurs and isn't caught by FastAPI's top-level error handling,
    # (e.g., if it's a non-HTTPException error), pytest itself will catch the unhandled exception
    # from the endpoint's synchronous code path if it's not properly awaited or handled.
    # For now, we won't assert a specific status code until we see the failure mode.
    # A 500 Internal Server Error would be a common outcome for unhandled exceptions.
    # assert response.status_code == 200 # Change this assertion once the error is fixed
