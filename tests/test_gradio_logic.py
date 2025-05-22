"""Test module for Gradio logic components."""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filmquant_ui import predict_box_office


def test_predict_box_office_basic_execution():
    """
    Tests the predict_box_office function with dummy inputs to ensure it runs
    and returns the correct outputs.
    """
    # Provide dummy inputs matching current function signature
    dummy_title = "Test Movie"
    dummy_genres = "Action"
    dummy_director = "Christopher Nolan"
    dummy_actor1 = "Leonardo DiCaprio"
    dummy_actor2 = "Tom Hardy"
    dummy_actor3 = "Marion Cotillard"
    dummy_studio = "Universal Pictures"
    dummy_budget_millions = 200.0
    dummy_runtime = 148
    dummy_release_date = "2025-07-04"
    dummy_trailer_views = 2.0  # In millions

    outputs = predict_box_office(
        dummy_title,
        dummy_genres,
        dummy_director,
        dummy_actor1,
        dummy_actor2,
        dummy_actor3,
        dummy_studio,
        dummy_budget_millions,
        dummy_runtime,
        dummy_release_date,
        dummy_trailer_views,
    )

    # Check that we get the expected outputs
    assert isinstance(outputs, tuple), "Outputs should be a tuple"
    assert len(outputs) == 7, f"Expected 7 outputs, got {len(outputs)}"

    # Check that first output (predicted revenue) is a string
    predicted_revenue = outputs[0]
    assert isinstance(predicted_revenue, str), "Predicted revenue should be a string"

    print(f"Prediction outputs: {outputs}")  # For debugging
