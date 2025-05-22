import pytest
from app.main import predict_button_callback # The full predict_button_callback

def test_full_predict_callback_basic_execution(capsys):
    """
    Tests the full predict_button_callback with dummy inputs to ensure it runs
    and returns the correct number of outputs.
    This is a basic check, not an end-to-end data validation.
    """
    # Provide dummy inputs for all 13 arguments
    dummy_title = "Test Movie"
    dummy_genre_ids = ["g1"] # Assuming it can handle a list
    dummy_director_id = "d1"
    dummy_cast_id1 = "c1"
    dummy_cast_id2 = "c2"
    dummy_cast_id3 = None # Test with a None value
    dummy_studio_id = "s1"
    dummy_budget_usd = 100000000
    dummy_runtime_minutes = 120
    dummy_release_date_str = "2025-01-15" # String date as Gradio would pass it
    dummy_screens_opening_day = 3000
    dummy_marketing_budget_est_usd = 50000000
    dummy_trailer_views_prerelease = 10000000

    outputs = predict_button_callback(
        dummy_title,
        dummy_genre_ids,
        dummy_director_id,
        dummy_cast_id1,
        dummy_cast_id2,
        dummy_cast_id3,
        dummy_studio_id,
        dummy_budget_usd,
        dummy_runtime_minutes,
        dummy_release_date_str, # Pass string directly
        dummy_screens_opening_day,
        dummy_marketing_budget_est_usd,
        dummy_trailer_views_prerelease
    )

    # Check that the DEBUG print statement (for release_date) was called
    captured = capsys.readouterr()
    assert f"DEBUG: release_date is '{dummy_release_date_str}'" in captured.out

    # Check that we get 7 outputs as defined in the full function
    assert isinstance(outputs, tuple), "Outputs should be a tuple"
    assert len(outputs) == 7, f"Expected 7 outputs, got {len(outputs)}"
    
    # Optional: Basic type checks for outputs if they are consistent
    # For example, the first output (predicted_revenue) should be a string
    # This depends on whether get_prediction mock returns an error or valid data
    # For now, just checking the count is a good first step after the change.
    # If predict_button_callback returns error values, their types might differ.
    print(f"Callback outputs: {outputs}") # For debugging the output types in test log

