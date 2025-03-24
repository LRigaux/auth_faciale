"""
Main layout for the facial authentication application.

This module provides the main layout structure for the Dash application,
including the accordion with different sections.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from src.ui.components import (
    create_header, create_footer,
    create_home_component, create_brute_force_component,
    create_eigenfaces_component, create_comparison_component
)

def create_layout():
    """
    Create the main application layout.
    
    Returns:
        dbc.Container: Main application container
    """
    return dbc.Container([
        # Header
        create_header(),
        
        # Main content
        html.Div([
            dbc.Accordion([
                dbc.AccordionItem(
                    create_home_component(),
                    title="Home",
                    item_id="home"
                ),
                dbc.AccordionItem(
                    create_brute_force_component(),
                    title="Brute Force Method",
                    item_id="brute-force"
                ),
                dbc.AccordionItem(
                    create_eigenfaces_component(),
                    title="Eigenfaces Method",
                    item_id="eigenfaces"
                ),
                dbc.AccordionItem(
                    create_comparison_component(),
                    title="Methods Comparison",
                    item_id="comparison"
                )
            ], id="main-accordion", active_item="home", always_open=True)
        ], className="my-4"),
        
        # Footer
        create_footer(),
        
        # Interval for checking progress
        dcc.Interval(id="progress-interval", interval=1000, n_intervals=0),
        
        # Global application state
        dcc.Store(id="app-state")
    ], fluid=True) 