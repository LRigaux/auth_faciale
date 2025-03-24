"""
Home component for the facial authentication application.

This module provides the component for the home page, where users
can load datasets and view basic information.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from .common import create_card, create_progress_bar, create_image_display

def create_home_component():
    """
    Create the home component.
    
    Returns:
        dbc.Container: Home component container
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H1("Facial Authentication System", className="mb-4"),
                html.P(
                    """
                    Welcome to the Facial Authentication System. This application demonstrates and compares 
                    two different approaches to facial authentication: brute force and eigenfaces.
                    """,
                    className="lead"
                ),
                html.P(
                    """
                    Start by loading a dataset below, then explore the different methods using the accordion sections.
                    """
                )
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card("Dataset Selection", [
                    html.P("Select a dataset to load:"),
                    dbc.Select(
                        id="dataset-select",
                        options=[
                            {"label": "Dataset 1 (small synthetic dataset)", "value": "1"},
                            {"label": "Dataset 2 (large synthetic dataset)", "value": "2"}
                        ],
                        value="1",
                        className="mb-3"
                    ),
                    dbc.Button("Load Dataset", id="load-dataset-btn", color="primary", className="mb-3"),
                    html.Div([
                        create_progress_bar("dataset-progress"),
                        dcc.Loading([
                            html.Div(id="dataset-loading-output")
                        ], type="circle")
                    ]),
                ], id="dataset-card")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card("Dataset Information", [
                    html.Div([
                        html.P("No dataset loaded yet.", id="dataset-info")
                    ], id="dataset-info-container"),
                    
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Sample Gallery Image", className="text-center mb-2"),
                                create_image_display("sample-gallery-img")
                            ], width=6),
                            dbc.Col([
                                html.H5("Sample Probe Image", className="text-center mb-2"),
                                create_image_display("sample-probe-img")
                            ], width=6)
                        ])
                    ], id="dataset-samples-container", style={"display": "none"})
                ], id="dataset-info-card")
            ], width=12)
        ]),
        
        # Hidden div to store dataset state
        dcc.Store(id="dataset-store")
    ]) 