"""
Common UI components for the facial authentication application.

This module provides reusable UI components shared across different
parts of the application.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from typing import List, Dict, Any, Optional

def create_header():
    """
    Create the application header.
    
    Returns:
        dbc.Navbar: Bootstrap navbar component
    """
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Facial Authentication System", className="ms-2"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Home", href="#home", id="nav-home")),
                dbc.NavItem(dbc.NavLink("Brute Force", href="#brute-force", id="nav-brute-force")),
                dbc.NavItem(dbc.NavLink("Eigenfaces", href="#eigenfaces", id="nav-eigenfaces")),
                dbc.NavItem(dbc.NavLink("Comparison", href="#comparison", id="nav-comparison")),
            ], className="ms-auto", navbar=True),
        ]),
        color="primary",
        dark=True,
        className="mb-4"
    )

def create_footer():
    """
    Create the application footer.
    
    Returns:
        dbc.Container: Bootstrap container for the footer
    """
    return dbc.Container([
        html.Hr(),
        html.P(
            ["Facial Authentication System - Created with ", 
            html.A("Dash", href="https://dash.plotly.com/", target="_blank"),
            " - 2023"],
            className="text-center text-muted"
        )
    ], fluid=True, className="mt-4")

def create_card(title: str, children: List[dash.development.base_component.Component], 
               id: Optional[str] = None, collapsed: bool = False):
    """
    Create a Bootstrap card with optional collapse functionality.
    
    Args:
        title: Card title
        children: Card content components
        id: Component ID (optional)
        collapsed: Whether the card should be initially collapsed
        
    Returns:
        dbc.Card: Bootstrap card component
    """
    card_props = {"className": "mb-3"}
    if id:
        card_props["id"] = id
        
    if collapsed:
        return dbc.Card([
            dbc.CardHeader(
                dbc.Button(
                    title,
                    color="link",
                    id=f"{id}-header" if id else None,
                    className="text-decoration-none d-block text-left p-0"
                )
            ),
            dbc.Collapse(
                dbc.CardBody(children),
                id=f"{id}-collapse" if id else None,
                is_open=not collapsed
            )
        ], **card_props)
    else:
        return dbc.Card([
            dbc.CardHeader(title),
            dbc.CardBody(children)
        ], **card_props)

def create_progress_bar(id: str, label: Optional[str] = None):
    """
    Create a progress bar with optional label.
    
    Args:
        id: Component ID
        label: Progress bar label (optional)
        
    Returns:
        dbc.Progress: Bootstrap progress component
    """
    return html.Div([
        html.P(label or "", id=f"{id}-label", className="mb-1"),
        dbc.Progress(id=id, value=0, striped=True, animated=True, className="mb-3")
    ])

def create_image_display(id: str, height: str = "200px", width: str = "200px"):
    """
    Create an image display component.
    
    Args:
        id: Component ID
        height: Image height
        width: Image width
        
    Returns:
        html.Div: Div containing the image
    """
    return html.Div([
        html.Img(
            id=id,
            src="",
            style={
                "height": height,
                "width": width,
                "objectFit": "contain",
                "border": "1px solid #ddd",
                "borderRadius": "4px",
                "padding": "5px"
            },
            className="mx-auto d-block"
        )
    ], className="text-center")

def create_confusion_matrix_display(id: str, animate: bool = True):
    """
    Create a confusion matrix display component.
    
    Args:
        id: Component ID
        animate: Whether to animate the confusion matrix
        
    Returns:
        dcc.Graph: Plotly graph component for the confusion matrix
    """
    return dcc.Graph(
        id=id,
        config={'displayModeBar': False},
        figure=create_empty_confusion_matrix()
    )

def create_empty_confusion_matrix():
    """
    Create an empty confusion matrix figure.
    
    Returns:
        go.Figure: Empty confusion matrix figure
    """
    return go.Figure(
        data=go.Heatmap(
            z=[[0, 0], [0, 0]],
            x=['Not Authenticated', 'Authenticated'],
            y=['Non-enrolled', 'Enrolled'],
            colorscale='Blues',
            showscale=False,
            text=[['TN: 0 (0%)', 'FP: 0 (0%)'], ['FN: 0 (0%)', 'TP: 0 (0%)']],
            texttemplate="%{text}",
            textfont={"size": 16}
        ),
        layout=go.Layout(
            title='Confusion Matrix',
            xaxis=dict(title='Predicted'),
            yaxis=dict(title='Actual', autorange='reversed'),
            margin=dict(l=60, r=30, t=60, b=60),
            height=400
        )
    )

def create_confusion_matrix_figure(confusion_matrix: List[List[int]]):
    """
    Create a confusion matrix figure from matrix data.
    
    Args:
        confusion_matrix: 2x2 confusion matrix
        
    Returns:
        go.Figure: Confusion matrix figure
    """
    # Convert to numpy array
    cm = np.array(confusion_matrix)
    
    # Calculate percentages
    total = np.sum(cm)
    if total > 0:
        percentages = cm / total * 100
    else:
        percentages = np.zeros_like(cm)
    
    # Create text annotations with counts and percentages
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    
    tn_pct, fp_pct = percentages[0, 0], percentages[0, 1]
    fn_pct, tp_pct = percentages[1, 0], percentages[1, 1]
    
    annotations = [
        [f'TN: {tn} ({tn_pct:.1f}%)', f'FP: {fp} ({fp_pct:.1f}%)'],
        [f'FN: {fn} ({fn_pct:.1f}%)', f'TP: {tp} ({tp_pct:.1f}%)']
    ]
    
    return go.Figure(
        data=go.Heatmap(
            z=cm,
            x=['Not Authenticated', 'Authenticated'],
            y=['Non-enrolled', 'Enrolled'],
            colorscale='Blues',
            showscale=False,
            text=annotations,
            texttemplate="%{text}",
            textfont={"size": 16}
        ),
        layout=go.Layout(
            title='Confusion Matrix',
            xaxis=dict(title='Predicted'),
            yaxis=dict(title='Actual', autorange='reversed'),
            margin=dict(l=60, r=30, t=60, b=60),
            height=400
        )
    ) 