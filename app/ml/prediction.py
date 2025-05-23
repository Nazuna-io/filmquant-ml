import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from app.static_data_loader import (
    get_cast_and_crew,
    get_genres,
    get_historical_films,
    get_studios,
)

# Placeholder for actual GCS client and XGBoost/Scikit-learn loading
# from google.cloud import storage
# import xgboost
# import joblib # For scikit-learn pipeline


# --- Mock Objects (to be replaced with actual model/processor loading) ---
class MockXGBoostBooster:
    """Mocks an XGBoost Booster object for MVP development."""

    def __init__(self, model_path):
        self.model_path = model_path
        # Simulate loading some feature names or properties if needed
        self.feature_names = [
            "budget_usd",
            "runtime_minutes",
            "screens_opening_day",
            "marketing_budget_est_usd",
            "marketing_budget_is_estimated",
            "trailer_views_prerelease",
            "genre_Action",
            "genre_Comedy",
            "genre_Drama",
            "genre_Sci-Fi",
            "genre_Thriller",
            "director_id_encoded",
            "actor1_id_encoded",
            "actor2_id_encoded",
            "actor3_id_encoded",
            "studio_id_encoded",
            "release_month",
            "release_quarter_1",
            "release_quarter_2",
            "release_quarter_3",
            "release_quarter_4",
            "avg_cast_notable_works",
            "director_notable_works",
        ]
        print(f"MockXGBoostBooster: Initialized with model from {model_path}")

    def predict(self, dmatrix):
        """Mocks the predict method. Assumes dmatrix is a DataFrame for simplicity."""
        print(f"MockXGBoostBooster: Predicting for {dmatrix.shape[0]} instances.")

        # More sophisticated mock prediction based on various features
        base_multiplier = 1.5  # Base revenue multiplier for budget

        # Get key features
        budget = dmatrix["budget_usd"].iloc[0] if "budget_usd" in dmatrix else 100000000

        # Adjust for marketing - marketing has slightly less impact if it's estimated
        marketing_multiplier = 1.0
        if "marketing_budget_est_usd" in dmatrix:
            marketing = dmatrix["marketing_budget_est_usd"].iloc[0]
            marketing_is_estimated = (
                dmatrix.get("marketing_budget_is_estimated", pd.Series([0])).iloc[0]
                == 1
            )

            # Less confidence in the impact of estimated marketing budgets
            if marketing_is_estimated:
                marketing_multiplier = 1.0 + (marketing / budget) * 0.25
            else:
                marketing_multiplier = 1.0 + (marketing / budget) * 0.4

        predicted_revenue = budget * base_multiplier * marketing_multiplier

        # Make confidence interval asymmetric - predicting revenue is harder on the upside
        lower_bound = predicted_revenue * 0.7
        upper_bound = predicted_revenue * 1.5

        return np.array([[lower_bound, predicted_revenue, upper_bound]])

    def get_score(self, importance_type="gain"):
        """Mocks feature importance scores."""
        print(f"MockXGBoostBooster: Getting feature scores (type: {importance_type})")
        # Ensure these feature names align with those produced by preprocess_input
        mock_scores = {
            "budget_usd": 150.0,
            "marketing_budget_est_usd": 130.0,  # Add marketing budget as an important feature
            "trailer_views_prerelease": 120.0,
            "screens_opening_day": 100.0,
            "avg_cast_notable_works": 80.0,
            "genre_Action": 70.0,
            "director_notable_works": 65.0,
            "runtime_minutes": 50.0,
        }
        return mock_scores


class MockFeatureProcessor:
    """Mocks a scikit-learn Pipeline object."""

    def __init__(self, processor_path):
        self.processor_path = processor_path
        self.fitted_ = True  # Simulate a fitted processor
        self.expected_features_ = []  # Will be set during preprocess_input for now
        print(f"MockFeatureProcessor: Initialized with processor from {processor_path}")

    def transform(self, X_df):
        """Mocks the transform method. Assumes X_df is a DataFrame."""
        print(f"MockFeatureProcessor: Transforming data.")
        # Simulate scaling for some numerical features - in reality, this would use fitted scalers
        X_transformed = X_df.copy()
        for col in [
            "budget_usd",
            "runtime_minutes",
            "screens_opening_day",
            "marketing_budget_est_usd",
            "trailer_views_prerelease",
            "avg_cast_notable_works",
            "director_notable_works",
        ]:
            if col in X_transformed:
                X_transformed[col] = X_transformed[col] / 100000  # Mock scaling
        return X_transformed


# --- Model and Processor Loading (LLM Instruction from 5.2) ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
MODEL_FILE = os.path.join(MODEL_DIR, "placeholder_filmquant_ml_revenue_model.ubj")
PROCESSOR_FILE = os.path.join(MODEL_DIR, "placeholder_feature_processor.pkl")

# Global placeholders for the loaded model and processor
_model = None
_feature_processor = None
_all_genres_list = []
_all_personnel_list = []
_all_studios_list = []


def load_xgboost_model_from_gcs(bucket_name=None, model_path=None):
    """Loads an XGBoost model. For MVP, loads a mock model from local placeholder."""
    # In future: download from GCS using bucket_name and model_path
    # client = storage.Client()
    # bucket = client.bucket(bucket_name)
    # blob = bucket.blob(model_path)
    # local_model_path = "/tmp/filmquant_ml_revenue_model.ubj"
    # blob.download_to_filename(local_model_path)
    # model = xgboost.Booster()
    # model.load_model(local_model_path)

    # MVP: Load mock model
    print(f"Loading MOCK XGBoost model from: {MODEL_FILE}")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model placeholder not found: {MODEL_FILE}")
    model = MockXGBoostBooster(MODEL_FILE)
    return model


def load_feature_processor_from_gcs(bucket_name=None, processor_path=None):
    """Loads a scikit-learn feature processor. For MVP, loads a mock processor."""
    # In future: download from GCS and unpickle
    # local_processor_path = "/tmp/feature_processor.pkl"
    # ... download logic ...
    # processor = joblib.load(local_processor_path)

    # MVP: Load mock processor
    print(f"Loading MOCK feature processor from: {PROCESSOR_FILE}")
    if not os.path.exists(PROCESSOR_FILE):
        raise FileNotFoundError(
            f"Feature processor placeholder not found: {PROCESSOR_FILE}"
        )
    processor = MockFeatureProcessor(PROCESSOR_FILE)
    return processor


def initialize_model_and_processor():
    """Initializes the model, processor, and reference data."""
    global _model, _feature_processor, _all_genres_list, _all_personnel_list, _all_studios_list
    if _model is None:
        _model = load_xgboost_model_from_gcs()  # Add GCS params if/when available
    if _feature_processor is None:
        _feature_processor = load_feature_processor_from_gcs()  # Add GCS params

    # Load reference data (idempotent, data is cached in static_data_loader)
    _all_genres_list = get_genres()
    _all_personnel_list = get_cast_and_crew()
    _all_studios_list = get_studios()


# --- Feature Preprocessing (LLM Instruction from 5.1) ---
def preprocess_input(
    raw_input_data: dict,
    feature_processor: MockFeatureProcessor,
    all_genres_list: list,
    all_personnel_list: list,
    all_studios_list: list,
) -> pd.DataFrame:
    """
    Transforms raw user input into a Pandas DataFrame suitable for XGBoost model prediction.
    Performs one-hot encoding for genres, label encoding for personnel (based on IDs),
    date feature extraction, calculation of avg_cast_notable_works and director_notable_works,
    and applies transformations from feature_processor.
    Ensures consistent feature ordering as expected by the model.
    """
    processed_data = {}

    # 1. Basic numerical features (will be scaled by feature_processor)
    processed_data["budget_usd"] = raw_input_data.get("budget_usd", 0)
    processed_data["runtime_minutes"] = raw_input_data.get("runtime_minutes", 0)
    processed_data["screens_opening_day"] = raw_input_data.get("screens_opening_day", 0)

    # Handle marketing budget - truly optional with fallback to estimated value
    marketing_budget = raw_input_data.get("marketing_budget_est_usd")
    if marketing_budget is None or marketing_budget == 0:
        # If no marketing budget provided, estimate based on production budget (common industry rule)
        production_budget = raw_input_data.get("budget_usd", 0)
        if production_budget >= 100000000:  # Blockbuster
            estimated_marketing = production_budget * 0.75  # 75% of production budget
        elif production_budget >= 30000000:  # Mid-tier
            estimated_marketing = production_budget * 0.95  # 95% of production budget
        else:  # Smaller film
            estimated_marketing = production_budget * 1.2  # 120% of production budget

        processed_data["marketing_budget_est_usd"] = estimated_marketing
        processed_data["marketing_budget_is_estimated"] = (
            1  # Flag to indicate this is an estimate
        )
    else:
        processed_data["marketing_budget_est_usd"] = marketing_budget
        processed_data["marketing_budget_is_estimated"] = 0  # User-provided value

    processed_data["trailer_views_prerelease"] = raw_input_data.get(
        "trailer_views_prerelease", 0
    )  # Handle missing

    # 2. Genre One-Hot Encoding
    genre_ids_map = {g["id"]: g["name"] for g in all_genres_list}
    input_genre_ids = raw_input_data.get("genre_ids", [])
    for genre_obj in all_genres_list:
        genre_name_slug = (
            genre_obj["name"].replace(" ", "_").replace("-", "_")
        )  # Simple slug for col name
        processed_data[f"genre_{genre_name_slug}"] = (
            1 if genre_obj["id"] in input_genre_ids else 0
        )

    # 3. Personnel Label Encoding (Simplified for MVP: map ID to an index)
    # A more robust approach would use a fitted LabelEncoder from the training set
    personnel_id_to_idx = {p["id"]: i for i, p in enumerate(all_personnel_list)}
    studio_id_to_idx = {s["id"]: i for i, s in enumerate(all_studios_list)}

    processed_data["director_id_encoded"] = personnel_id_to_idx.get(
        raw_input_data.get("director_id"), -1
    )  # -1 for unknown

    cast_ids = raw_input_data.get("cast_ids", [])
    for i in range(3):  # actor1, actor2, actor3
        actor_id = cast_ids[i] if i < len(cast_ids) else None
        processed_data[f"actor{i+1}_id_encoded"] = personnel_id_to_idx.get(actor_id, -1)

    processed_data["studio_id_encoded"] = studio_id_to_idx.get(
        raw_input_data.get("studio_id"), -1
    )

    # 4. Date Feature Extraction
    release_date_str = raw_input_data.get("release_date")
    if release_date_str:
        try:
            release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
            processed_data["release_month"] = release_date.month
            quarter = (release_date.month - 1) // 3 + 1
            for q_val in range(1, 5):
                processed_data[f"release_quarter_{q_val}"] = (
                    1 if quarter == q_val else 0
                )
        except ValueError:
            processed_data["release_month"] = (
                -1
            )  # Or some other indicator for invalid date
            for q_val in range(1, 5):
                processed_data[f"release_quarter_{q_val}"] = 0
    else:
        processed_data["release_month"] = -1
        for q_val in range(1, 5):
            processed_data[f"release_quarter_{q_val}"] = 0

    # 5. Calculated Aggregate Features (avg_cast_notable_works, director_notable_works)
    personnel_map = {p["id"]: p for p in all_personnel_list}
    director_obj = personnel_map.get(raw_input_data.get("director_id"))
    processed_data["director_notable_works"] = (
        director_obj.get("notable_works_count", 0) if director_obj else 0
    )

    avg_cast_notable_works = 0
    valid_cast_count = 0
    for actor_id in cast_ids[:3]:  # Max 3 actors
        actor_obj = personnel_map.get(actor_id)
        if actor_obj:
            avg_cast_notable_works += actor_obj.get("notable_works_count", 0)
            valid_cast_count += 1
    processed_data["avg_cast_notable_works"] = (
        avg_cast_notable_works / valid_cast_count if valid_cast_count > 0 else 0
    )

    # Convert to DataFrame
    df = pd.DataFrame([processed_data])

    # Apply scikit-learn pipeline transformations (mocked for now)
    df_transformed = feature_processor.transform(df)

    # Ensure consistent feature ordering (as expected by the model)
    # This list should come from the trained model's expectation or feature_processor
    # Using the one from MockXGBoostBooster for now
    expected_feature_order = (
        _model.feature_names if _model else feature_processor.expected_features_
    )

    # Reorder and add missing columns (with 0 or appropriate default)
    final_df = pd.DataFrame(columns=expected_feature_order)
    for col in expected_feature_order:
        if col in df_transformed:
            final_df[col] = df_transformed[col]
        else:
            # A more robust solution would know the default fill_value for each missing feature
            final_df[col] = 0

    feature_processor.expected_features_ = (
        expected_feature_order  # Store for consistency if needed
    )
    return final_df


# --- Post-Prediction Processing (LLM Instructions from 5.4) ---
def extract_quantile_predictions(predictions_array):
    """
    Extracts confidence interval and median prediction from XGBoost quantile predictions.
    Assumes predictions_array is [[q_low, q_median, q_high]] for a single input.
    """
    if (
        predictions_array is None
        or len(predictions_array) == 0
        or len(predictions_array[0]) < 3
    ):
        # Return default or raise error
        return 0, 0, 0

    pred_single_instance = predictions_array[0]
    confidence_interval_low_usd = pred_single_instance[0]
    predicted_revenue_usd = pred_single_instance[1]  # Median
    confidence_interval_high_usd = pred_single_instance[2]
    return (
        predicted_revenue_usd,
        confidence_interval_low_usd,
        confidence_interval_high_usd,
    )


def get_top_factors_xgboost(
    model: MockXGBoostBooster,
    feature_names: list,
    user_input_values_dict: dict,
    top_n=5,
    use_shap=False,
):
    """
    Gets top N influencing factors from an XGBoost model.
    If use_shap is True, it assumes a SHAP explainer is available (not implemented for MVP).
    Otherwise, uses model.get_score(). Maps to feature_names and associates with user_input_values_dict.
    """
    if use_shap:
        # Placeholder for SHAP value calculation (requires a SHAP explainer and appropriate data)
        print("SHAP values not implemented in MVP mock. Falling back to get_score.")
        importances = model.get_score(importance_type="gain")
    else:
        importances = model.get_score(
            importance_type="gain"
        )  # 'weight', 'cover', 'total_gain', 'total_cover'

    # Sort features by importance
    sorted_factors = sorted(importances.items(), key=lambda item: item[1], reverse=True)

    top_factors_output = []
    for feature_encoded_name, importance_score in sorted_factors[:top_n]:
        # Try to map encoded feature name back to a more original/human-readable form or value
        # This mapping can be complex. E.g., 'genre_Action' -> 'Genre: Action', 'director_id_encoded' -> 'Director Name'
        # For MVP, we'll use the feature_encoded_name and try to get a value from raw input if possible.

        # Simplistic mapping back for some known patterns
        original_feature_key = feature_encoded_name
        display_name = feature_encoded_name
        value_contributed = "N/A"

        if feature_encoded_name.startswith("genre_"):
            display_name = (
                f"Genre: {feature_encoded_name.split('genre_')[1].replace('_', ' ')}"
            )
            # Value for one-hot encoded is 1 if present, 0 if not. Look up from user_input.
            genre_id_for_name = None
            genre_name_part = feature_encoded_name.split("genre_")[1]
            for g in _all_genres_list:
                if g["name"].replace(" ", "_").replace("-", "_") == genre_name_part:
                    genre_id_for_name = g["id"]
                    break
            if genre_id_for_name and genre_id_for_name in user_input_values_dict.get(
                "genre_ids", []
            ):
                value_contributed = 1
            else:
                value_contributed = 0
        elif feature_encoded_name == "director_id_encoded":
            director_id = user_input_values_dict.get("director_id")
            director = next(
                (p["name"] for p in _all_personnel_list if p["id"] == director_id),
                "Unknown Director",
            )
            display_name = f"Director: {director}"
            value_contributed = director_id if director_id else "N/A"
        elif feature_encoded_name == "marketing_budget_est_usd":
            display_name = "Marketing Budget"
            value = user_input_values_dict.get("marketing_budget_est_usd", 0)

            # Check if marketing budget is estimated
            if "marketing_budget_est_usd" not in user_input_values_dict:
                display_name = "Marketing Budget (estimated)"
                # Calculate the estimated value
                production_budget = user_input_values_dict.get("budget_usd", 0)
                if production_budget >= 100000000:  # Blockbuster
                    estimated_value = production_budget * 0.75
                elif production_budget >= 30000000:  # Mid-tier
                    estimated_value = production_budget * 0.95
                else:  # Smaller film
                    estimated_value = production_budget * 1.2
                value_contributed = f"${estimated_value:,.0f} (estimated)"
            else:
                value_contributed = f"${value:,.0f}"
        elif feature_encoded_name in user_input_values_dict:
            value_contributed = user_input_values_dict[feature_encoded_name]
        elif feature_encoded_name == "avg_cast_notable_works":
            # This is a derived feature, its direct input isn't simple. Could show the calculated value before scaling.
            # For now, just N/A for simplicity or show the raw calculated value if we pass it through preprocess_input
            value_contributed = "Calculated"
        # Add more mappings as needed for other encoded/derived features

        top_factors_output.append(
            {
                "feature": display_name,
                "importance_score": float(importance_score),  # Ensure JSON serializable
                "value_contributed": value_contributed,
            }
        )
    return top_factors_output


def find_similar_films(
    predicted_film_data: dict, historical_films_list: list, top_n=3
) -> list:
    """
    Finds top N similar historical films based on primary genre match and budget proximity (+/- 25%).
    predicted_film_data: dict with 'primary_genre_id' (actual ID string) and 'budget_usd'.
    historical_films_list: loaded list from historical_films.json.
    """
    if not historical_films_list:
        return []

    predicted_budget = predicted_film_data.get("budget_usd")
    primary_genre_id = predicted_film_data.get(
        "primary_genre_id"
    )  # Assumes this is determined and passed in

    if predicted_budget is None or primary_genre_id is None:
        return []  # Not enough info to compare

    budget_min = predicted_budget * 0.75
    budget_max = predicted_budget * 1.25

    candidates = []
    for film in historical_films_list:
        # Match primary genre (first genre in the list for historical film, for simplicity)
        historical_primary_genre_id = film.get("genre_ids", [None])[0]
        if historical_primary_genre_id == primary_genre_id:
            film_budget = film.get("budget_usd")
            if film_budget is not None and budget_min <= film_budget <= budget_max:
                # Calculate budget proximity for sorting (smaller difference is better)
                film["budget_difference"] = abs(film_budget - predicted_budget)
                candidates.append(film)

    # Sort by release date (descending) first, then by budget proximity (ascending) as the primary sort key.
    # Python's sort is stable, so this achieves the desired multi-level sorting.
    candidates.sort(key=lambda x: x.get("release_date", "1900-01-01"), reverse=True)
    candidates.sort(key=lambda x: x["budget_difference"], reverse=False)

    similar_films_output = []
    for film in candidates[:top_n]:
        similar_films_output.append(
            {
                "title": film.get("title"),
                "predicted_or_actual_revenue_usd": film.get(
                    "actual_box_office_total_usd"
                ),
                "type": "historical_actual",  # As per schema
            }
        )
    return similar_films_output


# --- Main Prediction Orchestration ---
def get_prediction(raw_input_data: dict):
    """Orchestrates the full prediction pipeline."""
    initialize_model_and_processor()  # Ensure everything is loaded

    # 1. Preprocess input data
    # The 'all_genres_list' etc. are now global within this module after initialize
    processed_df = preprocess_input(
        raw_input_data,
        _feature_processor,
        _all_genres_list,
        _all_personnel_list,
        _all_studios_list,
    )

    # For XGBoost, typically convert to DMatrix, but mock model takes DataFrame
    # dinput = xgboost.DMatrix(processed_df)
    dinput = processed_df

    # 2. Make prediction
    model_predictions = _model.predict(dinput)

    # 3. Post-process predictions
    predicted_revenue, conf_low, conf_high = extract_quantile_predictions(
        model_predictions
    )

    # Domestic/International Split (as per user's clarification: 33% dom / 67% intl)
    domestic_share = 0.33
    international_share = 0.67

    # 4. Get Top Factors
    # feature_names should be the columns from processed_df, after transformation by feature_processor
    # For MVP mock, _model.feature_names is used by preprocess_input to set order.
    top_factors = get_top_factors_xgboost(
        _model, list(processed_df.columns), raw_input_data, top_n=5
    )

    # 5. Find Similar Films
    # Determine primary_genre_id for the input film (e.g., first in its genre_ids list)
    primary_genre_id_input = None
    if raw_input_data.get("genre_ids") and len(raw_input_data.get("genre_ids", [])) > 0:
        primary_genre_id_input = raw_input_data.get("genre_ids", [])[0]
    elif _all_genres_list:
        primary_genre_id_input = _all_genres_list[0]["id"]  # Fallback

    predicted_film_info_for_similarity = {
        "primary_genre_id": primary_genre_id_input,
        "budget_usd": raw_input_data.get("budget_usd"),
    }
    historical_film_data = get_historical_films()  # Load from static_data_loader
    comparable_films = find_similar_films(
        predicted_film_info_for_similarity, historical_film_data, top_n=3
    )

    output_schema = {
        "predicted_revenue_usd": predicted_revenue,
        "confidence_interval_low_usd": conf_low,
        "confidence_interval_high_usd": conf_high,
        "domestic_revenue_share_percent": domestic_share,
        "international_revenue_share_percent": international_share,
        "top_factors": top_factors,
        "comparable_films": comparable_films,
        "raw_input_data": raw_input_data,
    }

    return output_schema


if __name__ == "__main__":
    # Test the functions
    print("\n--- Testing ML Prediction Module ---")
    initialize_model_and_processor()  # Call this to load mock objects and reference data

    sample_raw_input = {
        "title": "Test Film Alpha",
        "genre_ids": [g["id"] for g in _all_genres_list[:2]],  # Use loaded genre IDs
        "director_id": (
            _all_personnel_list[0]["id"] if _all_personnel_list else "p_unknown"
        ),
        "cast_ids": [p["id"] for p in _all_personnel_list[1:3] if p["role"] == "actor"],
        "studio_id": _all_studios_list[0]["id"] if _all_studios_list else "s_unknown",
        "budget_usd": 75000000,
        "runtime_minutes": 110,
        "release_date": "2025-10-20",
        "screens_opening_day": 3200,
        "marketing_budget_est_usd": 20000000,
        "trailer_views_prerelease": 10000000,
    }
    print(f"\nSample Raw Input:\n{json.dumps(sample_raw_input, indent=2)}")

    # Test preprocess_input
    # print("\n--- Testing preprocess_input ---")
    # processed_df = preprocess_input(sample_raw_input, _feature_processor, _all_genres_list, _all_personnel_list, _all_studios_list)
    # print(f"Processed DataFrame columns: {processed_df.columns.tolist()}")
    # print(f"Processed DataFrame shape: {processed_df.shape}")
    # print(f"Processed DataFrame (first 5 cols):\n{processed_df.iloc[:, :5].to_string()}")

    # Test full prediction
    print("\n--- Testing get_prediction ---")
    prediction_output = get_prediction(sample_raw_input)
    print(f"Prediction Output:\n{json.dumps(prediction_output, indent=2)}")

    # Test find_similar_films directly
    print("\n--- Testing find_similar_films ---")
    pfi_sim = {
        "primary_genre_id": sample_raw_input["genre_ids"][0],
        "budget_usd": sample_raw_input["budget_usd"],
    }
    hist_films = get_historical_films()
    sim_films = find_similar_films(pfi_sim, hist_films)
    print(f"Similar Films found: {json.dumps(sim_films, indent=2)}")

    # Test top factors (using the result from get_prediction)
    # print("\n--- Testing top_factors (from get_prediction) ---")
    # print(json.dumps(prediction_output['top_factors'], indent=2))
