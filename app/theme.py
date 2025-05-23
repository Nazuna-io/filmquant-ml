"""
Apple-inspired theme for the FilmQuant ML Gradio UI with compatibility for Gradio 4.44.1
"""

import gradio as gr


def create_filmquant_ml_interface(callbacks=None):
    """
    Creates an enhanced Gradio interface for FilmQuant ML with Apple-inspired design elements.
    This function replicates the existing UI functionality but with improved visuals,
    using existing Gradio themes for compatibility.

    Args:
        callbacks: A dictionary containing callback functions for the UI
            - predict_button_callback: Function for prediction button
            - historical_film_selected_callback: Function for historical film selection

    Returns:
        gr.Blocks: The constructed Gradio interface
    """
    from app.static_data_loader import (
        get_cast_and_crew,
        get_genres,
        get_historical_films,
        get_studios,
    )

    # If callbacks are not provided, we're in demo mode - use empty functions
    if callbacks is None:

        def predict_button_callback(*args):
            return ["Sample prediction"] + [None] * 6

        def historical_film_selected_callback(*args):
            return [None] * 13

    else:
        predict_button_callback = callbacks.get("predict_button_callback")
        historical_film_selected_callback = callbacks.get(
            "historical_film_selected_callback"
        )

    # Use a built-in theme that resembles Apple's design
    apple_theme = gr.themes.Soft()

    # Load reference data for dropdowns
    def get_choices_from_data(data_list, name_key="name", id_key="id"):
        return [(item[name_key], item[id_key]) for item in data_list]

    genre_choices = get_choices_from_data(get_genres())
    personnel_choices = get_choices_from_data(get_cast_and_crew())
    director_choices = [
        (name, id)
        for name, id in personnel_choices
        if next(
            (
                p
                for p in get_cast_and_crew()
                if p["id"] == id and p["role"] == "director"
            ),
            None,
        )
    ]
    actor_choices = [
        (name, id)
        for name, id in personnel_choices
        if next(
            (p for p in get_cast_and_crew() if p["id"] == id and p["role"] == "actor"),
            None,
        )
    ]
    studio_choices = get_choices_from_data(get_studios())
    historical_film_choices = get_choices_from_data(
        get_historical_films(), name_key="title"
    )

    # Create the enhanced interface
    with gr.Blocks(
        theme=apple_theme, title="FilmQuant ML - Film Revenue Predictor"
    ) as filmquant_ml_interface:
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
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
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
                                elem_id="film-title",
                            )

                            genre_input = gr.Dropdown(
                                label="Genre(s)",
                                choices=genre_choices,
                                multiselect=True,
                                elem_id="film-genres",
                            )

                            director_input = gr.Dropdown(
                                label="Director",
                                choices=director_choices,
                                allow_custom_value=False,
                                elem_id="film-director",
                            )

                            with gr.Row():
                                cast_input1 = gr.Dropdown(
                                    label="Lead Actor 1",
                                    choices=actor_choices,
                                    allow_custom_value=False,
                                    elem_id="film-actor1",
                                )
                                cast_input2 = gr.Dropdown(
                                    label="Lead Actor 2",
                                    choices=actor_choices,
                                    allow_custom_value=False,
                                    elem_id="film-actor2",
                                )
                                cast_input3 = gr.Dropdown(
                                    label="Lead Actor 3",
                                    choices=actor_choices,
                                    allow_custom_value=False,
                                    elem_id="film-actor3",
                                )

                            studio_input = gr.Dropdown(
                                label="Studio",
                                choices=studio_choices,
                                allow_custom_value=False,
                                elem_id="film-studio",
                            )

                            budget_input = gr.Number(
                                label="Budget (USD)",
                                value=50000000,
                                step=1000000,
                                elem_id="film-budget",
                            )
                            runtime_input = gr.Number(
                                label="Runtime (minutes)",
                                value=120,
                                step=1,
                                elem_id="film-runtime",
                            )

                            release_date_input = gr.Textbox(
                                label="Release Date",
                                value="2025-07-04",
                                placeholder="YYYY-MM-DD",
                                elem_id="film-release-date",
                            )

                            screens_input = gr.Number(
                                label="Screens Opening Day",
                                value=3000,
                                step=100,
                                elem_id="film-screens",
                            )
                            marketing_budget_input = gr.Number(
                                label="Marketing Budget (USD, Optional)",
                                value=None,
                                step=1000000,
                                elem_id="film-marketing",
                            )

                            trailer_views_input = gr.Number(
                                label="Trailer Views Prerelease",
                                value=10000000,
                                step=100000,
                                elem_id="film-trailer-views",
                            )

                            predict_button = gr.Button(
                                "Predict Box Office Revenue",
                                variant="primary",
                                elem_id="predict-button",
                                size="lg",
                            )

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown(
                                """
                                <div style="margin-bottom: 15px;">
                                    <h3 style="font-size: 20px; font-weight: 600; margin-bottom: 10px;">Prediction Results</h3>
                                </div>
                                """
                            )

                            predicted_revenue_output = gr.Textbox(
                                label="Predicted Total Revenue",
                                elem_id="predicted-revenue",
                                interactive=False,
                            )

                            with gr.Row():
                                confidence_interval_output = gr.Textbox(
                                    label="Confidence Interval (90%)",
                                    lines=2,
                                    elem_id="confidence-interval",
                                    interactive=False,
                                )

                                with gr.Column():
                                    domestic_share_output = gr.Textbox(
                                        label="Est. Domestic Share",
                                        elem_id="domestic-share",
                                        interactive=False,
                                    )

                                    international_share_output = gr.Textbox(
                                        label="Est. International Share",
                                        elem_id="international-share",
                                        interactive=False,
                                    )

                            gr.Markdown(
                                """
                                <div style="margin-top: 20px; margin-bottom: 10px;">
                                    <h4 style="font-size: 16px; font-weight: 600;">Top Influencing Factors</h4>
                                </div>
                                """
                            )

                            top_factors_output = gr.DataFrame(
                                headers=[
                                    "feature",
                                    "importance_score",
                                    "value_contributed",
                                ],
                                col_count=(3, "fixed"),
                                interactive=False,
                                wrap=True,
                                elem_id="top-factors",
                            )

                            gr.Markdown(
                                """
                                <div style="margin-top: 20px; margin-bottom: 10px;">
                                    <h4 style="font-size: 16px; font-weight: 600;">Comparable Historical Films</h4>
                                </div>
                                """
                            )

                            comparable_films_output = gr.DataFrame(
                                headers=[
                                    "title",
                                    "predicted_or_actual_revenue_usd",
                                    "type",
                                ],
                                col_count=(3, "fixed"),
                                interactive=False,
                                wrap=True,
                                elem_id="comparable-films",
                            )

                            with gr.Accordion(
                                "Raw Input (for verification)", open=False
                            ):
                                raw_input_display_output = gr.JSON(
                                    label="Raw Input to Model (Post-Processed)",
                                    elem_id="raw-input",
                                )

            with gr.TabItem("Historical Validation Mode", id=1):
                with gr.Group():
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
                        elem_id="historical-film",
                    )

                    historical_actuals_output = gr.Textbox(
                        label="Historical Actual Revenue",
                        lines=3,
                        interactive=False,
                        elem_id="historical-actuals",
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
                title_input,
                genre_input,
                director_input,
                cast_input1,
                cast_input2,
                cast_input3,
                studio_input,
                budget_input,
                runtime_input,
                release_date_input,
                screens_input,
                marketing_budget_input,
                trailer_views_input,
            ],
            outputs=[
                predicted_revenue_output,
                confidence_interval_output,
                domestic_share_output,
                international_share_output,
                top_factors_output,
                comparable_films_output,
                raw_input_display_output,
            ],
        )

        historical_film_dropdown.change(
            fn=historical_film_selected_callback,
            inputs=[historical_film_dropdown],
            outputs=[
                title_input,
                genre_input,
                director_input,
                cast_input1,
                cast_input2,
                cast_input3,
                studio_input,
                budget_input,
                runtime_input,
                release_date_input,
                screens_input,
                marketing_budget_input,
                trailer_views_input,
                historical_actuals_output,
            ],
        )

    return filmquant_ml_interface


# If this module is run directly, show a demo of the theme
if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Soft(), title="Apple-like Theme Demo") as demo:
        gr.Markdown("# Apple-like Theme Demo")
        gr.Markdown("This is a demonstration of the Apple-inspired Gradio theme")

        with gr.Row():
            with gr.Column():
                gr.Textbox(label="Text Input Example", placeholder="Enter some text...")
                gr.Dropdown(
                    label="Dropdown Example",
                    choices=["Option 1", "Option 2", "Option 3"],
                )
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
