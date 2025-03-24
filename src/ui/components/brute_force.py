"""
Brute force component for the facial authentication application.

This module provides the component for the brute force method page, where users
can evaluate the brute force method and test authentication.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from .common import (
    create_card, create_progress_bar, create_image_display, 
    create_confusion_matrix_display
)

def create_brute_force_component():
    """
    Create the brute force component.
    
    Returns:
        html.Div: Brute force component container
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H2("Brute Force Method", className="mb-3"),
                html.P(
                    """
                    This method compares face images directly by computing distances between pixel values.
                    It's simple but can be effective for small datasets.
                    """,
                    className="mb-4"
                )
            ], width=12)
        ]),
        
        dbc.Row([
            # Parameters column
            dbc.Col([
                create_card("Parameters", [
                    html.P("Configure the brute force method parameters:"),
                    
                    html.Label("Distance Metric:"),
                    dbc.Select(
                        id="bf-metric-select",
                        options=[
                            {"label": "Euclidean Distance (L2)", "value": "L2"},
                            {"label": "Manhattan Distance (L1)", "value": "L1"},
                            {"label": "Cosine Distance", "value": "cosine"}
                        ],
                        value="L2",
                        className="mb-3"
                    ),
                    
                    html.Label("Authentication Threshold:"),
                    dcc.Slider(
                        id="bf-threshold-slider",
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.5,
                        marks={i/10: str(i/10) for i in range(0, 11)},
                        className="mb-3"
                    ),
                    
                    html.Div(id="bf-threshold-display", className="mb-3"),
                    
                    dbc.Button(
                        "Find Best Threshold", 
                        id="bf-find-threshold-btn", 
                        color="secondary", 
                        className="me-2"
                    ),
                    dbc.Button(
                        "Evaluate Performance", 
                        id="bf-evaluate-btn", 
                        color="primary"
                    )
                ], id="bf-params-card")
            ], width=4),
            
            # Evaluation results column
            dbc.Col([
                create_card("Evaluation", [
                    create_progress_bar("bf-progress"),
                    
                    html.Div([
                        html.P("No evaluation results yet.", id="bf-results-text"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H5("Performance Metrics", className="text-center"),
                                    html.Div(id="bf-metrics")
                                ])
                            ], width=6),
                            
                            dbc.Col([
                                html.Div([
                                    html.H5("Confusion Matrix", className="text-center"),
                                    create_confusion_matrix_display("bf-confusion-matrix")
                                ])
                            ], width=6)
                        ]),
                    ], id="bf-results-container")
                ], id="bf-results-card")
            ], width=8)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card("Authentication Test", [
                    html.P("Test authentication with a random face image:"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Select Random Image", 
                                id="bf-select-image-btn", 
                                color="secondary", 
                                className="mb-3 me-2"
                            ),
                            dbc.Button(
                                "Authenticate", 
                                id="bf-authenticate-btn", 
                                color="primary", 
                                className="mb-3",
                                disabled=True
                            )
                        ], width=12)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H5("Test Image", className="text-center mb-2"),
                            create_image_display("bf-test-image", height="200px", width="200px")
                        ], width=4),
                        
                        dbc.Col([
                            html.H5("Authentication Result", className="text-center mb-2"),
                            html.Div([
                                html.P("No image selected", id="bf-auth-result")
                            ], className="d-flex align-items-center justify-content-center h-100")
                        ], width=8)
                    ])
                ], id="bf-auth-test-card")
            ], width=6),
            
            dbc.Col([
                create_card("Match Details", [
                    html.Div([
                        html.H5("Best Match in Gallery", className="text-center mb-2"),
                        create_image_display("bf-best-match-image"),
                        html.P("No match available", id="bf-match-details", className="text-center mt-2")
                    ], id="bf-match-container"),
                    
                    html.Div([
                        html.H5("Other Images of Same Person", className="text-center mb-3"),
                        html.Div(id="bf-same-person-images", className="d-flex flex-wrap justify-content-center")
                    ], id="bf-same-person-container", style={"display": "none"})
                ], id="bf-match-card")
            ], width=6)
        ]),
        
        # Stores for the brute force state
        dcc.Store(id="bf-results-store"),
        dcc.Store(id="bf-test-image-store")
    ]) 