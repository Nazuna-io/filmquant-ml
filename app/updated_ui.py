#!/usr/bin/env python3
"""
Updated UI with Quick Select + Search Combo for Directors, Actors, Studios
All budget values displayed in millions for better UX
"""

import gradio as gr
import pandas as pd
from app.static_data_loader import get_genres, get_cast_and_crew, get_historical_films, get_studios

def create_updated_filmquant_ml_interface(callbacks):
    """Create the updated FilmQuant ML interface with improved UX"""
    
    # Get data for choices
    genres = get_genres()
    cast_and_crew = get_cast_and_crew()
    historical_films = get_historical_films()
    studios = get_studios()
    
    # Create choice lists
    genre_choices = [g['name'] for g in genres]
    
    # A-List Quick Select (top 17 from our analysis)
    a_list_directors = [
        "Christopher Nolan", "Martin Scorsese", "Greta Gerwig", "James Gunn",
        "Steven Spielberg", "Quentin Tarantino", "Denis Villeneuve", "Jordan Peele"
    ]
    
    a_list_actors = [
        "Leonardo DiCaprio", "Tom Cruise", "Margot Robbie", "Ryan Gosling",
        "Matt Damon", "Chris Pratt", "Keanu Reeves", "Harrison Ford",
        "Emma Stone", "Jennifer Lawrence", "Scarlett Johansson", "Jack Black", "Tom Hanks"
    ]
    
    major_studios = [
        "Walt Disney Pictures", "Universal Pictures", "Warner Bros. Pictures",
        "Marvel Studios (Walt Disney Studios)", "Paramount Pictures", "Sony Pictures",
        "Netflix", "Amazon Studios", "Apple Studios"
    ]
    
    # All available choices for search
    all_directors = list(set([person['name'] for person in cast_and_crew if person.get('role') == 'director']))
    all_actors = list(set([person['name'] for person in cast_and_crew if person.get('role') == 'actor']))
    all_studios = [studio['name'] for studio in studios]
    
    # Historical film choices
    historical_film_choices = [(f"{film['title']} ({film.get('release_date', 'Unknown')[:4]})", film['id']) for film in historical_films]
    
    predict_button_callback = callbacks["predict_button_callback"]
    historical_film_selected_callback = callbacks["historical_film_selected_callback"]
    
    def create_quick_select_component(label, a_list_options, all_options, elem_id):
        """Create a quick select + search component"""
        with gr.Group():
            gr.Markdown(f"### {label}")
            
            # Quick select buttons for A-listers
            gr.Markdown("**Quick Select (A-List):**")
            with gr.Row():
                buttons = []
                for i, option in enumerate(a_list_options):
                    if i % 4 == 0 and i > 0:  # New row every 4 buttons
                        with gr.Row():
                            pass
                    btn = gr.Button(option, size="sm", variant="secondary")
                    buttons.append(btn)
            
            # Search dropdown for all options
            gr.Markdown("**Or Search All:**")
            dropdown = gr.Dropdown(
                label=f"Search {label}",
                choices=all_options,
                allow_custom_value=False,
                filterable=True,
                elem_id=elem_id
            )
            
            return dropdown, buttons
    
    # Create the main interface
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="FilmQuant ML - Box Office Prediction",
        css="""
        .quick-select-btn {
            margin: 2px !important;
            font-size: 12px !important;
            padding: 4px 8px !important;
        }
        .millions-input {
            background: #f8fafc !important;
        }
        """
    ) as filmquant_ml_interface:
        
        gr.Markdown(
            """
            # 🎬 FilmQuant ML
            ## AI-Powered Box Office Prediction
            
            Predict domestic box office performance using our trained ML model with 90.5% accuracy.
            """
        )
        
        with gr.Tabs():
            with gr.TabItem("🎯 Predict New Film", id=0):
                with gr.Row():
                    # Left Column - Inputs
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Film Details")
                        
                        title_input = gr.Textbox(
                            label="Film Title", 
                            placeholder="e.g., My Awesome Blockbuster",
                            elem_id="film-title"
                        )
                        
                        genre_input = gr.Dropdown(
                            label="Genre(s)",
                            choices=genre_choices,
                            multiselect=True,
                            elem_id="film-genres"
                        )
                        
                        # Director Quick Select + Search
                        director_dropdown, director_buttons = create_quick_select_component(
                            "Director", a_list_directors, all_directors, "film-director"
                        )
                        director_input = director_dropdown
                        
                        # Actors Quick Select + Search  
                        gr.Markdown("### 🎭 Cast")
                        
                        # Actor 1
                        actor1_dropdown, actor1_buttons = create_quick_select_component(
                            "Lead Actor 1", a_list_actors, all_actors, "film-actor1"
                        )
                        cast_input1 = actor1_dropdown
                        
                        # Actor 2
                        actor2_dropdown, actor2_buttons = create_quick_select_component(
                            "Lead Actor 2", a_list_actors, all_actors, "film-actor2"
                        )
                        cast_input2 = actor2_dropdown
                        
                        # Actor 3
                        actor3_dropdown, actor3_buttons = create_quick_select_component(
                            "Lead Actor 3", a_list_actors, all_actors, "film-actor3"
                        )
                        cast_input3 = actor3_dropdown
                        
                        # Studio Quick Select + Search
                        studio_dropdown, studio_buttons = create_quick_select_component(
                            "Studio", major_studios, all_studios, "film-studio"
                        )
                        studio_input = studio_dropdown
                        
                        gr.Markdown("### 💰 Production Details")
                        
                        budget_input = gr.Number(
                            label="Budget (Millions USD)",
                            value=150.0,
                            step=5.0,
                            elem_id="film-budget",
                            elem_classes=["millions-input"]
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
                        
                        screens_input = gr.Number(
                            label="Opening Day Screen Count",
                            value=3500,
                            step=100,
                            elem_id="film-screens"
                        )
                        
                        marketing_budget_input = gr.Number(
                            label="Marketing Budget (Millions USD, Optional)",
                            value=50.0,
                            step=5.0,
                            elem_id="film-marketing",
                            elem_classes=["millions-input"]
                        )
                        
                        trailer_views_input = gr.Number(
                            label="Trailer Views (Pre-release)",
                            value=1000000,
                            step=100000,
                            elem_id="film-trailer-views"
                        )
                        
                        # Predict Button
                        predict_button = gr.Button(
                            "🚀 Predict Box Office", 
                            variant="primary", 
                            size="lg",
                            elem_id="predict-button"
                        )
                    
                    # Right Column - Results
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Prediction Results")
                        
                        predicted_revenue_output = gr.Textbox(
                            label="Predicted Domestic Box Office",
                            lines=2,
                            elem_id="predicted-revenue",
                            interactive=False
                        )
                        
                        confidence_interval_output = gr.Textbox(
                            label="Confidence Interval (90%)",
                            lines=2,
                            elem_id="confidence-interval",
                            interactive=False
                        )
                        
                        with gr.Row():
                            domestic_share_output = gr.Textbox(
                                label="Est. Domestic Share",
                                elem_id="domestic-share",
                                interactive=False
                            )
                            
                            international_share_output = gr.Textbox(
                                label="Est. International Share",
                                elem_id="international-share",
                                interactive=False
                            )
                        
                        gr.Markdown("### 🎯 Key Insights")
                        
                        top_factors_output = gr.DataFrame(
                            headers=["Factor", "Importance", "Impact"],
                            col_count=(3, "fixed"),
                            interactive=False,
                            wrap=True,
                            elem_id="top-factors"
                        )
                        
                        gr.Markdown("### 🎬 Similar Films")
                        
                        comparable_films_output = gr.DataFrame(
                            headers=["Film", "Box Office", "Type"],
                            col_count=(3,"fixed"),
                            interactive=False,
                            wrap=True,
                            elem_id="comparable-films"
                        )
                        
                        with gr.Accordion("🔍 Technical Details", open=False):
                            raw_input_display_output = gr.JSON(
                                label="Model Input Data",
                                elem_id="raw-input"
                            )
            
            with gr.TabItem("📈 Historical Validation", id=1):
                with gr.Group():
                    gr.Markdown(
                        """
                        ### 📈 Historical Film Analysis
                        Select a film from our training dataset to see actual performance and test prediction accuracy.
                        """
                    )
                    
                    historical_film_dropdown = gr.Dropdown(
                        label="Select Historical Film",
                        choices=historical_film_choices,
                        allow_custom_value=False,
                        elem_id="historical-film"
                    )
                    
                    historical_actuals_output = gr.Textbox(
                        label="Actual Performance",
                        lines=4,
                        interactive=False,
                        elem_id="historical-actuals"
                    )
                    
                    gr.Markdown(
                        """
                        **Note:** Selected film data will populate the prediction form. 
                        You can modify values and re-predict to see model accuracy.
                        """
                    )

        # Wire up quick select buttons
        def create_button_callbacks(buttons, dropdown):
            """Create callbacks for quick select buttons"""
            for button in buttons:
                button.click(
                    fn=lambda x, btn_value=button.value: btn_value,
                    inputs=[],
                    outputs=[dropdown]
                )
        
        # Set up all button callbacks
        create_button_callbacks(director_buttons, director_input)
        create_button_callbacks(actor1_buttons, cast_input1)
        create_button_callbacks(actor2_buttons, cast_input2) 
        create_button_callbacks(actor3_buttons, cast_input3)
        create_button_callbacks(studio_buttons, studio_input)
        
        # Main prediction callback (convert millions to actual values)
        def predict_with_million_conversion(*args):
            # Convert budget values from millions to actual values for the model
            args_list = list(args)
            if args_list[7] is not None:  # budget_input
                args_list[7] = args_list[7] * 1_000_000  # Convert to actual USD
            if args_list[11] is not None:  # marketing_budget_input 
                args_list[11] = args_list[11] * 1_000_000  # Convert to actual USD
            
            # Call original prediction function
            return predict_button_callback(*args_list)
        
        # Wire up main callbacks
        predict_button.click(
            fn=predict_with_million_conversion,
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

        # Historical selection callback (convert actual values to millions for display)
        def historical_with_million_conversion(film_id):
            result = historical_film_selected_callback(film_id)
            if result and len(result) >= 8:
                result_list = list(result)
                # Convert budget from actual to millions for display
                if result_list[7] is not None:  # budget 
                    result_list[7] = result_list[7] / 1_000_000
                if len(result_list) > 11 and result_list[11] is not None:  # marketing budget
                    result_list[11] = result_list[11] / 1_000_000
                return tuple(result_list)
            return result

        historical_film_dropdown.change(
            fn=historical_with_million_conversion,
            inputs=[historical_film_dropdown],
            outputs=[
                title_input, genre_input, director_input, cast_input1, cast_input2, cast_input3, studio_input,
                budget_input, runtime_input, release_date_input, screens_input,
                marketing_budget_input, trailer_views_input, historical_actuals_output
            ]
        )
        
    return filmquant_ml_interface
