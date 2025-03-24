"""
Comparison component for the facial authentication application.

This module provides the component for comparing different authentication methods.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from .common import create_card, create_confusion_matrix_display

def create_comparison_component():
    """
    Create the methods comparison component.
    
    Returns:
        html.Div: Comparison component container
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H2("Methods Comparison", className="mb-3"),
                html.P(
                    """
                    Compare the performance of brute force and eigenfaces methods for facial authentication.
                    Evaluate both methods on the same dataset to see which performs better.
                    """,
                    className="mb-4"
                ),
                dbc.Button(
                    "Compare Methods", 
                    id="comp-evaluate-btn", 
                    color="primary", 
                    className="mb-4"
                ),
                html.Div(id="comp-results-text", className="mb-3")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card("Performance Metrics Comparison", [
                    html.P("Comparison of key performance metrics between methods:"),
                    
                    html.Div([
                        html.P("No comparison data available. Please evaluate both methods first.", className="text-center"),
                    ], id="comparison-placeholder"),
                    
                    html.Div(id="comp-metrics-table", className="mb-3"),
                    
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Brute Force Metrics"),
                                html.Div(id="comp-bf-metrics")
                            ], width=6),
                            dbc.Col([
                                html.H5("Eigenfaces Metrics"),
                                html.Div(id="comp-ef-metrics")
                            ], width=6)
                        ])
                    ], className="mb-3"),
                    
                    html.Div([
                        dbc.Table(
                            [
                                html.Thead(html.Tr([
                                    html.Th("Metric"),
                                    html.Th("Brute Force"),
                                    html.Th("Eigenfaces"),
                                    html.Th("Difference")
                                ])),
                                html.Tbody([
                                    html.Tr([
                                        html.Td("Accuracy"),
                                        html.Td(id="comp-bf-accuracy"),
                                        html.Td(id="comp-ef-accuracy"),
                                        html.Td(id="comp-diff-accuracy")
                                    ]),
                                    html.Tr([
                                        html.Td("Precision"),
                                        html.Td(id="comp-bf-precision"),
                                        html.Td(id="comp-ef-precision"),
                                        html.Td(id="comp-diff-precision")
                                    ]),
                                    html.Tr([
                                        html.Td("Recall"),
                                        html.Td(id="comp-bf-recall"),
                                        html.Td(id="comp-ef-recall"),
                                        html.Td(id="comp-diff-recall")
                                    ]),
                                    html.Tr([
                                        html.Td("F1-Score"),
                                        html.Td(id="comp-bf-f1"),
                                        html.Td(id="comp-ef-f1"),
                                        html.Td(id="comp-diff-f1")
                                    ]),
                                    html.Tr([
                                        html.Td("Execution Time"),
                                        html.Td(id="comp-bf-time"),
                                        html.Td(id="comp-ef-time"),
                                        html.Td(id="comp-time-ratio")
                                    ])
                                ])
                            ],
                            bordered=True,
                            hover=True,
                            responsive=True,
                            striped=True,
                            className="mb-3"
                        ),
                        
                        html.Div([
                            html.H5("Overall Winner", className="text-center"),
                            html.Div(id="comp-winner", className="text-center mb-3")
                        ])
                    ], id="comparison-table-container", style={"display": "none"})
                ], id="comparison-metrics-card")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card("ROC Curves", [
                    html.P("Comparison of ROC curves for both methods:"),
                    dcc.Graph(
                        id="comp-roc-curve",
                        config={'displayModeBar': False}
                    )
                ])
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card("Confusion Matrices", [
                    html.P("Comparison of confusion matrices:"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H5("Brute Force", className="text-center mb-2"),
                            create_confusion_matrix_display("comp-bf-confusion-matrix")
                        ], width=6),
                        
                        dbc.Col([
                            html.H5("Eigenfaces", className="text-center mb-2"),
                            create_confusion_matrix_display("comp-ef-confusion-matrix")
                        ], width=6)
                    ])
                ], id="comparison-confusion-card")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card("Performance Visualization", [
                    html.P("Graphical comparison of performance metrics:"),
                    
                    dcc.Graph(
                        id="comp-metrics-chart",
                        config={'displayModeBar': False},
                        figure=create_empty_comparison_chart()
                    ),
                    
                    dcc.Graph(
                        id="comp-time-chart",
                        config={'displayModeBar': False},
                        figure=create_empty_time_chart()
                    )
                ], id="comparison-charts-card")
            ], width=12)
        ]),
        
        # Store for comparison data
        dcc.Store(id="comparison-store")
    ])

def create_empty_comparison_chart():
    """
    Create an empty metrics comparison chart.
    
    Returns:
        go.Figure: Empty metrics comparison chart
    """
    return go.Figure(
        data=[
            go.Bar(name='Brute Force', x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], y=[0, 0, 0, 0]),
            go.Bar(name='Eigenfaces', x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], y=[0, 0, 0, 0])
        ],
        layout=go.Layout(
            title='Performance Metrics Comparison',
            yaxis=dict(title='Score', range=[0, 1]),
            barmode='group',
            height=400
        )
    )

def create_empty_time_chart():
    """
    Create an empty execution time comparison chart.
    
    Returns:
        go.Figure: Empty time comparison chart
    """
    return go.Figure(
        data=[
            go.Bar(name='Execution Time', x=['Brute Force', 'Eigenfaces'], y=[0, 0])
        ],
        layout=go.Layout(
            title='Execution Time Comparison',
            yaxis=dict(title='Time (seconds)'),
            height=400
        )
    ) 