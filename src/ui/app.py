"""
Main Dash application for the facial authentication system.

This module initializes the Dash application with the main layout and
registers the callbacks for all components.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import os
from pathlib import Path

# Create necessary directories if they don't exist
os.makedirs("data/dataset1", exist_ok=True)
os.makedirs("data/dataset2", exist_ok=True)
os.makedirs("assets/img", exist_ok=True)

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets')
)

# Set app title
app.title = "Facial Authentication System"

# Import layout and callbacks after app initialization to avoid circular imports
from src.ui.layout import create_layout
from src.ui.callbacks import register_callbacks

# Configure the app layout
app.layout = create_layout()

# Register all callbacks
register_callbacks(app)

# Server instance for deployment
server = app.server

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8050) 