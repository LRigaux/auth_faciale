"""
Eigenfaces component for the facial authentication application.

This module provides the component for the eigenfaces method page, where users
can evaluate the eigenfaces method and test authentication.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from .common import (
    create_card, create_progress_bar, create_image_display, 
    create_confusion_matrix_display
)

def create_eigenfaces_component():
    """
    Create the eigenfaces component.
    
    Returns:
        html.Div: Eigenfaces component container
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H2("Eigenfaces Method", className="mb-3"),
                html.P(
                    """
                    This method uses Principal Component Analysis (PCA) to reduce dimensionality
                    and extract the most important features from face images.
                    """,
                    className="mb-4"
                )
            ], width=12)
        ]),
        
        dbc.Row([
            # Parameters column
            dbc.Col([
                create_card("Parameters", [
                    html.P("Configure the eigenfaces method parameters:"),
                    
                    html.Label("Number of Components:"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="ef-n-components-input",
                            type="number",
                            min=1,
                            max=100,
                            step=1,
                            value=10
                        ),
                        dbc.InputGroupText("components")
                    ], className="mb-3"),
                    
                    html.Label("Explained Variance Threshold:"),
                    dcc.Slider(
                        id="ef-variance-slider",
                        min=0.5,
                        max=0.99,
                        step=0.01,
                        value=0.95,
                        marks={i/100: str(i/100) for i in range(50, 100, 10)},
                        className="mb-3"
                    ),
                    
                    html.Div(id="ef-variance-display", className="mb-3"),
                    
                    html.Label("Authentication Threshold:"),
                    dcc.Slider(
                        id="ef-threshold-slider",
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.5,
                        marks={i/10: str(i/10) for i in range(0, 11)},
                        className="mb-3"
                    ),
                    
                    html.Div(id="ef-threshold-display", className="mb-3"),
                    
                    dbc.Button(
                        "Find Best Threshold", 
                        id="ef-find-threshold-btn", 
                        color="secondary", 
                        className="me-2"
                    ),
                    dbc.Button(
                        "Evaluate Performance", 
                        id="ef-evaluate-btn", 
                        color="primary"
                    )
                ], id="ef-params-card")
            ], width=4),
            
            # Evaluation results column
            dbc.Col([
                create_card("Evaluation", [
                    create_progress_bar("ef-progress"),
                    
                    html.Div([
                        html.P("No evaluation results yet.", id="ef-results-text"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H5("Performance Metrics", className="text-center"),
                                    html.Div(id="ef-metrics")
                                ])
                            ], width=6),
                            
                            dbc.Col([
                                html.Div([
                                    html.H5("Confusion Matrix", className="text-center"),
                                    create_confusion_matrix_display("ef-confusion-matrix")
                                ])
                            ], width=6)
                        ]),
                    ], id="ef-results-container")
                ], id="ef-results-card")
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
                                id="ef-select-image-btn", 
                                color="secondary", 
                                className="mb-3 me-2"
                            ),
                            dbc.Button(
                                "Authenticate", 
                                id="ef-authenticate-btn", 
                                color="primary", 
                                className="mb-3",
                                disabled=True
                            )
                        ], width=12)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H5("Test Image", className="text-center mb-2"),
                            create_image_display("ef-test-image", height="200px", width="200px")
                        ], width=4),
                        
                        dbc.Col([
                            html.H5("Authentication Result", className="text-center mb-2"),
                            html.Div([
                                html.P("No image selected", id="ef-auth-result")
                            ], className="d-flex align-items-center justify-content-center h-100")
                        ], width=8)
                    ])
                ], id="ef-auth-test-card")
            ], width=6),
            
            dbc.Col([
                create_card("Match Details", [
                    html.Div([
                        html.H5("Best Match in Gallery", className="text-center mb-2"),
                        create_image_display("ef-best-match-image"),
                        html.P("No match available", id="ef-match-details", className="text-center mt-2")
                    ], id="ef-match-container"),
                    
                    html.Div([
                        html.H5("Other Images of Same Person", className="text-center mb-3"),
                        html.Div(id="ef-same-person-images", className="d-flex flex-wrap justify-content-center")
                    ], id="ef-same-person-container", style={"display": "none"})
                ], id="ef-match-card")
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card("Eigenfaces Visualization", [
                    html.P("Visualization of the learned eigenfaces after training:"),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Mean Face", className="text-center mb-2"),
                            create_image_display("ef-mean-face")
                        ], width=4),
                        dbc.Col([
                            html.H5("Top Eigenfaces", className="text-center mb-2"),
                            html.Div(id="ef-eigenfaces-container", className="d-flex flex-wrap justify-content-center")
                        ], width=8)
                    ])
                ], id="ef-visualization-card")
            ], width=12)
        ]),
        
        # Stores for the eigenfaces state
        dcc.Store(id="ef-results-store"),
        dcc.Store(id="ef-test-image-store"),
        dcc.Store(id="ef-model-store")
    ]) 