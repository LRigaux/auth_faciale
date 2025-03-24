"""
Callbacks for the facial authentication application.

This module provides all the callbacks for the Dash application,
organized by component or feature.
"""

import dash
from dash import Input, Output, State, callback, html, dcc
import dash_bootstrap_components as dbc
import numpy as np
import base64
import io
from PIL import Image
import json
import plotly.graph_objects as go
import time
from typing import Dict, List, Any, Tuple, Optional

from src.core.dataset import FaceDataset, load_dataset
from src.core import brute_force as bf
from src.core import eigenfaces as ef
from src.core import evaluation as eval_module
from src.ui.components.common import create_confusion_matrix_figure

# Global state
app_state = {
    "progress": {},
    "cancel": {}
}

def register_callbacks(app):
    """
    Register all callbacks for the application.
    
    Args:
        app: Dash application instance
    """
    register_navigation_callbacks(app)
    register_home_callbacks(app)
    register_brute_force_callbacks(app)
    register_eigenfaces_callbacks(app)
    register_comparison_callbacks(app)

def register_navigation_callbacks(app):
    """
    Register callbacks for navigation.
    
    Args:
        app: Dash application instance
    """
    @app.callback(
        Output("main-accordion", "active_item"),
        [
            Input("nav-home", "n_clicks"),
            Input("nav-brute-force", "n_clicks"),
            Input("nav-eigenfaces", "n_clicks"),
            Input("nav-comparison", "n_clicks")
        ],
        prevent_initial_call=True
    )
    def navigate(home_clicks, bf_clicks, ef_clicks, comp_clicks):
        # Determine which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return "home"
            
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "nav-home":
            return "home"
        elif button_id == "nav-brute-force":
            return "brute-force"
        elif button_id == "nav-eigenfaces":
            return "eigenfaces"
        elif button_id == "nav-comparison":
            return "comparison"
            
        return "home"

def register_home_callbacks(app):
    """
    Register callbacks for the home component.
    
    Args:
        app: Dash application instance
    """
    @app.callback(
        [
            Output("dataset-progress", "value"),
            Output("dataset-progress-label", "children"),
            Output("dataset-loading-output", "children"),
            Output("dataset-store", "data"),
            Output("dataset-info", "children"),
            Output("dataset-samples-container", "style"),
            Output("sample-gallery-img", "src"),
            Output("sample-probe-img", "src")
        ],
        [
            Input("load-dataset-btn", "n_clicks"),
            Input("progress-interval", "n_intervals")
        ],
        [
            State("dataset-select", "value"),
            State("dataset-store", "data")
        ],
        prevent_initial_call=True
    )
    def handle_dataset_loading(n_clicks, n_intervals, dataset_num, current_data):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Initialize progress tracking
        if "dataset_loading" not in app_state["progress"]:
            app_state["progress"]["dataset_loading"] = {"value": 0, "message": ""}
        
        progress = app_state["progress"]["dataset_loading"]
        
        # If load button clicked, start loading
        if trigger_id == "load-dataset-btn" and n_clicks:
            # Start asynchronous loading
            progress["value"] = 5
            progress["message"] = "Initializing dataset loading..."
            
            try:
                # Update progress
                def update_progress(value, message):
                    progress["value"] = value
                    progress["message"] = message
                
                # Load dataset
                dataset_num = int(dataset_num) if dataset_num else 1
                dataset = load_dataset(dataset_num, progress_callback=update_progress)
                
                # Process and create display images
                gallery_img = array_to_base64(dataset.gallery[0])
                probe_img = array_to_base64(dataset.probes[0])
                
                # Preprocess the data
                gallery_processed, probes_processed = dataset.preprocess_images(
                    method='normalize', 
                    flatten=True,
                    progress_callback=update_progress
                )
                
                # Prepare dataset info
                info_html = html.Div([
                    html.P([
                        html.Strong("Dataset: "), f"Dataset {dataset_num}"
                    ]),
                    html.P([
                        html.Strong("Gallery size: "), f"{dataset.n_gallery} images"
                    ]),
                    html.P([
                        html.Strong("Probe size: "), f"{dataset.n_probes} images"
                    ]),
                    html.P([
                        html.Strong("Enrolled probes: "), f"{dataset.n_enrolled_probes} images"
                    ]),
                    html.P([
                        html.Strong("Non-enrolled probes: "), f"{dataset.n_non_enrolled_probes} images"
                    ]),
                    html.P([
                        html.Strong("Image dimensions: "), f"{dataset.image_shape[0]}x{dataset.image_shape[1]}"
                    ])
                ])
                
                # Serialize for storage
                serialized_dataset = {
                    "gallery": gallery_processed.tolist(),
                    "probes": probes_processed.tolist(),
                    "gallery_ids": dataset.gallery_ids,
                    "probe_ids": dataset.probe_ids,
                    "ground_truth": [bool(gt) for gt in dataset.ground_truth],
                    "gallery_shape": dataset.gallery.shape,
                    "probes_shape": dataset.probes.shape,
                    "image_shape": dataset.image_shape
                }
                
                # Set progress to complete
                progress["value"] = 100
                progress["message"] = "Dataset loaded successfully!"
                
                return (
                    progress["value"],
                    progress["message"],
                    None,
                    serialized_dataset,
                    info_html,
                    {"display": "block"},
                    gallery_img,
                    probe_img
                )
                
            except Exception as e:
                # Handle errors
                progress["value"] = 0
                progress["message"] = f"Error: {str(e)}"
                
                return (
                    progress["value"],
                    progress["message"],
                    html.Div(f"Error loading dataset: {str(e)}", className="text-danger"),
                    None,
                    "No dataset loaded yet.",
                    {"display": "none"},
                    "",
                    ""
                )
        
        # If interval triggered, just update progress
        elif trigger_id == "progress-interval":
            if current_data:
                # Dataset already loaded
                return (
                    100,
                    "Dataset loaded successfully!",
                    None,
                    current_data,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update
                )
            else:
                # Still loading or not started
                return (
                    progress["value"],
                    progress["message"],
                    None,
                    None,
                    dash.no_update,
                    {"display": "none"},
                    dash.no_update,
                    dash.no_update
                )
        
        # Default return
        return (0, "", None, None, dash.no_update, {"display": "none"}, "", "")

def register_brute_force_callbacks(app):
    """
    Register callbacks for the brute force component.
    
    Args:
        app: Dash application instance
    """
    @app.callback(
        Output("bf-threshold-display", "children"),
        Input("bf-threshold-slider", "value")
    )
    def update_bf_threshold_display(value):
        return f"Current threshold: {value:.2f}"

    @app.callback(
        [
            Output("bf-progress", "value"),
            Output("bf-progress-label", "children"),
            Output("bf-results-text", "children"),
            Output("bf-results-store", "data"),
            Output("bf-metrics", "children"),
            Output("bf-confusion-matrix", "figure")
        ],
        [
            Input("bf-evaluate-btn", "n_clicks"),
            Input("bf-find-threshold-btn", "n_clicks"),
            Input("progress-interval", "n_intervals")
        ],
        [
            State("dataset-store", "data"),
            State("bf-metric-select", "value"),
            State("bf-threshold-slider", "value"),
            State("bf-results-store", "data")
        ],
        prevent_initial_call=True
    )
    def handle_bf_evaluation(eval_clicks, find_clicks, n_intervals, 
                            dataset_data, metric, threshold, current_results):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Initialize progress tracking
        if "bf_evaluation" not in app_state["progress"]:
            app_state["progress"]["bf_evaluation"] = {"value": 0, "message": ""}
        
        progress = app_state["progress"]["bf_evaluation"]
        
        # If dataset not loaded, return error
        if not dataset_data:
            return (
                0, "", 
                "Please load a dataset first",
                None, None, 
                create_confusion_matrix_figure([[0, 0], [0, 0]])
            )
        
        # Deserialize dataset
        gallery, probes, gallery_ids, probe_ids, ground_truth, image_shape = deserialize_dataset(dataset_data)
        
        # Split probes into enrolled and non-enrolled
        enrolled_indices = [i for i, gt in enumerate(ground_truth) if gt]
        non_enrolled_indices = [i for i, gt in enumerate(ground_truth) if not gt]
        
        enrolled_probes = probes[enrolled_indices]
        non_enrolled_probes = probes[non_enrolled_indices]
        
        # If evaluate button clicked
        if trigger_id == "bf-evaluate-btn" and eval_clicks:
            # Start evaluation
            progress["value"] = 5
            progress["message"] = "Starting brute force evaluation..."
            
            try:
                # Update progress
                def update_progress(value, message):
                    progress["value"] = value
                    progress["message"] = message
                
                # Evaluate performance
                results = bf.evaluate_performance(
                    gallery=gallery,
                    enrolled_probes=enrolled_probes,
                    non_enrolled_probes=non_enrolled_probes,
                    threshold=threshold,
                    metric=metric,
                    progress_callback=update_progress
                )
                
                # Create metrics display
                metrics_html = create_metrics_table(results['performance'])
                
                # Create confusion matrix figure
                conf_mat_fig = create_confusion_matrix_figure(results['confusion_matrix'])
                
                # Create results text
                results_text = html.Div([
                    html.P([
                        "Evaluation completed in ",
                        html.Strong(f"{results['execution_time']:.2f} seconds")
                    ]),
                    html.P([
                        "Threshold: ",
                        html.Strong(f"{results['threshold']:.4f}")
                    ]),
                    html.P([
                        "Metric: ",
                        html.Strong(f"{results['metric']}")
                    ])
                ])
                
                # Set progress to complete
                progress["value"] = 100
                progress["message"] = "Evaluation completed successfully!"
                
                return (
                    progress["value"],
                    progress["message"],
                    results_text,
                    results,
                    metrics_html,
                    conf_mat_fig
                )
            
            except Exception as e:
                # Handle errors
                progress["value"] = 0
                progress["message"] = f"Error: {str(e)}"
                
                return (
                    progress["value"],
                    progress["message"],
                    f"Error during evaluation: {str(e)}",
                    None,
                    None,
                    create_confusion_matrix_figure([[0, 0], [0, 0]])
                )
        
        # If find best threshold button clicked
        elif trigger_id == "bf-find-threshold-btn" and find_clicks:
            # Start threshold finding
            progress["value"] = 5
            progress["message"] = "Finding best threshold..."
            
            try:
                # Update progress
                def update_progress(value, message):
                    progress["value"] = value
                    progress["message"] = message
                
                # Find best threshold
                best_threshold, results = bf.find_best_threshold(
                    gallery=gallery,
                    enrolled_probes=enrolled_probes,
                    non_enrolled_probes=non_enrolled_probes,
                    metric=metric,
                    progress_callback=update_progress
                )
                
                # Create metrics display
                metrics_html = create_metrics_table(results['performance'])
                
                # Create confusion matrix figure
                conf_mat_fig = create_confusion_matrix_figure(results['confusion_matrix'])
                
                # Create results text
                results_text = html.Div([
                    html.P([
                        "Best threshold found: ",
                        html.Strong(f"{best_threshold:.4f}")
                    ]),
                    html.P([
                        "Evaluation completed in ",
                        html.Strong(f"{results['execution_time']:.2f} seconds")
                    ]),
                    html.P([
                        "Metric: ",
                        html.Strong(f"{results['metric']}")
                    ])
                ])
                
                # Set progress to complete
                progress["value"] = 100
                progress["message"] = "Best threshold found successfully!"
                
                return (
                    progress["value"],
                    progress["message"],
                    results_text,
                    results,
                    metrics_html,
                    conf_mat_fig
                )
            
            except Exception as e:
                # Handle errors
                progress["value"] = 0
                progress["message"] = f"Error: {str(e)}"
                
                return (
                    progress["value"],
                    progress["message"],
                    f"Error finding best threshold: {str(e)}",
                    None,
                    None,
                    create_confusion_matrix_figure([[0, 0], [0, 0]])
                )
        
        # If interval triggered, just update progress
        elif trigger_id == "progress-interval":
            if current_results:
                # Evaluation already completed
                return (
                    100,
                    "Evaluation completed",
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update
                )
            else:
                # Still evaluating or not started
                return (
                    progress["value"],
                    progress["message"],
                    dash.no_update,
                    None,
                    None,
                    dash.no_update
                )
        
        # Default return
        return (0, "", dash.no_update, None, None, dash.no_update)
    
    @app.callback(
        [
            Output("bf-test-image", "src"),
            Output("bf-authenticate-btn", "disabled"),
            Output("bf-test-image-store", "data")
        ],
        Input("bf-select-image-btn", "n_clicks"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def select_random_bf_image(n_clicks, dataset_data):
        if not dataset_data or not n_clicks:
            return "", True, None
            
        # Deserialize dataset
        gallery, probes, gallery_ids, probe_ids, ground_truth, image_shape = deserialize_dataset(dataset_data)
        
        # Select random probe
        import random
        probe_idx = random.randint(0, len(probes) - 1)
        probe = probes[probe_idx]
        probe_id = probe_ids[probe_idx]
        is_enrolled = ground_truth[probe_idx]
        
        # Reshape for display
        probe_image = probe.reshape(image_shape)
        
        # Convert to base64
        img_src = array_to_base64(probe_image)
        
        # Store probe data
        probe_data = {
            "probe": probe.tolist(),
            "probe_id": probe_id,
            "is_enrolled": is_enrolled,
            "probe_idx": probe_idx
        }
        
        return img_src, False, probe_data
    
    @app.callback(
        [
            Output("bf-auth-result", "children"),
            Output("bf-best-match-image", "src"),
            Output("bf-match-details", "children"),
            Output("bf-same-person-container", "style"),
            Output("bf-same-person-images", "children")
        ],
        Input("bf-authenticate-btn", "n_clicks"),
        [
            State("dataset-store", "data"),
            State("bf-test-image-store", "data"),
            State("bf-metric-select", "value"),
            State("bf-threshold-slider", "value"),
            State("bf-results-store", "data")
        ],
        prevent_initial_call=True
    )
    def authenticate_bf_image(n_clicks, dataset_data, test_image_data, metric, threshold, results_data):
        if not n_clicks or not dataset_data or not test_image_data:
            return "No image selected", "", "", {"display": "none"}, []
            
        # Get threshold from results if available
        if results_data and 'threshold' in results_data:
            threshold = results_data['threshold']
            
        # Deserialize data
        gallery, probes, gallery_ids, probe_ids, ground_truth, image_shape = deserialize_dataset(dataset_data)
        
        probe = np.array(test_image_data["probe"])
        probe_id = test_image_data["probe_id"]
        is_enrolled = test_image_data["is_enrolled"]
        
        try:
            # Authenticate
            is_authenticated, closest_idx, min_distance = bf.authenticate(probe, gallery, threshold, metric)
            
            # Get best match
            best_match = gallery[closest_idx]
            best_match_id = gallery_ids[closest_idx]
            
            # Reshape for display
            best_match_image = best_match.reshape(image_shape)
            
            # Convert to base64
            best_match_src = array_to_base64(best_match_image)
            
            # Authentication result text
            if is_authenticated:
                result_html = html.Div([
                    html.H4("Authenticated ✅", className="text-success"),
                    html.P([
                        "The person was ",
                        html.Strong("successfully authenticated"),
                        " with a distance of ",
                        html.Strong(f"{min_distance:.4f}")
                    ]),
                    html.P([
                        "Ground truth: ",
                        html.Strong("Enrolled" if is_enrolled else "Not enrolled", 
                                   className="text-success" if is_enrolled else "text-danger")
                    ])
                ])
            else:
                result_html = html.Div([
                    html.H4("Not Authenticated ❌", className="text-danger"),
                    html.P([
                        "Authentication failed with a distance of ",
                        html.Strong(f"{min_distance:.4f}"),
                        " (threshold: ",
                        html.Strong(f"{threshold:.4f}"),
                        ")"
                    ]),
                    html.P([
                        "Ground truth: ",
                        html.Strong("Enrolled" if is_enrolled else "Not enrolled", 
                                   className="text-success" if is_enrolled else "text-danger")
                    ])
                ])
            
            # Match details
            match_details = f"Best match: ID {best_match_id} (Distance: {min_distance:.4f})"
            
            # Find other images of the same person
            same_person_images = []
            
            if is_authenticated:
                # Find all probes with the same ID as the best match
                same_id_indices = [i for i, pid in enumerate(probe_ids) if pid == best_match_id]
                
                # Create image components
                for idx in same_id_indices[:4]:  # Limit to 4 images
                    same_person_probe = probes[idx].reshape(image_shape)
                    img_src = array_to_base64(same_person_probe)
                    
                    img_div = html.Div([
                        html.Img(src=img_src, style={"height": "150px", "width": "150px", 
                                                     "margin": "5px", "border": "1px solid #ddd"})
                    ])
                    
                    same_person_images.append(img_div)
            
            # Show same person container only if authenticated
            same_person_style = {"display": "block"} if is_authenticated and same_person_images else {"display": "none"}
            
            return result_html, best_match_src, match_details, same_person_style, same_person_images
            
        except Exception as e:
            return f"Error: {str(e)}", "", "", {"display": "none"}, []

def register_eigenfaces_callbacks(app):
    """
    Register callbacks for the eigenfaces component.
    
    Args:
        app: Dash application instance
    """
    @app.callback(
        Output("ef-n-components-display", "children"),
        Input("ef-n-components-input", "value")
    )
    def update_ef_components_display(value):
        return f"Number of eigenvectors: {value}"
    
    @app.callback(
        Output("ef-variance-display", "children"),
        Input("ef-variance-slider", "value")
    )
    def update_ef_variance_display(value):
        return f"Variance explained: {value:.1f}%"
        
    @app.callback(
        Output("ef-threshold-display", "children"),
        Input("ef-threshold-slider", "value")
    )
    def update_ef_threshold_display(value):
        return f"Current threshold: {value:.2f}"
    
    @app.callback(
        [
            Output("ef-progress", "value"),
            Output("ef-progress-label", "children"),
            Output("ef-results-text", "children"),
            Output("ef-results-store", "data"),
            Output("ef-metrics", "children"),
            Output("ef-confusion-matrix", "figure"),
            Output("ef-eigenfaces-container", "children")
        ],
        [
            Input("ef-evaluate-btn", "n_clicks"),
            Input("ef-find-threshold-btn", "n_clicks"),
            Input("progress-interval", "n_intervals")
        ],
        [
            State("dataset-store", "data"),
            State("ef-n-components-input", "value"),
            State("ef-threshold-slider", "value"),
            State("ef-results-store", "data")
        ],
        prevent_initial_call=True
    )
    def handle_eigenfaces_training(train_clicks, find_clicks, n_intervals, 
                                  dataset_data, n_components, threshold, current_results):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Initialize progress tracking
        if "eigenfaces_training" not in app_state["progress"]:
            app_state["progress"]["eigenfaces_training"] = {"value": 0, "message": ""}
        
        progress = app_state["progress"]["eigenfaces_training"]
        
        # If dataset not loaded, return error
        if not dataset_data:
            return 0, "", "Please load a dataset first", None, None, create_confusion_matrix_figure([[0, 0], [0, 0]]), None
        
        # Deserialize dataset
        gallery, probes, gallery_ids, probe_ids, ground_truth, image_shape = deserialize_dataset(dataset_data)
        
        # Split probes into enrolled and non-enrolled
        enrolled_indices = [i for i, gt in enumerate(ground_truth) if gt]
        non_enrolled_indices = [i for i, gt in enumerate(ground_truth) if not gt]
        
        enrolled_probes = probes[enrolled_indices]
        non_enrolled_probes = probes[non_enrolled_indices]
        
        # If evaluate button clicked
        if trigger_id == "ef-evaluate-btn" and train_clicks:
            # Start training
            progress["value"] = 5
            progress["message"] = "Starting eigenfaces training..."
            
            try:
                # Update progress
                def update_progress(value, message):
                    progress["value"] = value
                    progress["message"] = message
                
                # Train eigenfaces model
                eigenfaces_model, results = ef.train_and_evaluate(
                    gallery=gallery,
                    enrolled_probes=enrolled_probes,
                    non_enrolled_probes=non_enrolled_probes,
                    n_components=n_components,
                    threshold=threshold,
                    progress_callback=update_progress
                )
                
                # Create metrics display
                metrics_html = create_metrics_table(results['performance'])
                
                # Create confusion matrix figure
                conf_mat_fig = create_confusion_matrix_figure(results['confusion_matrix'])
                
                # Create eigenfaces gallery
                ef_gallery = []
                for i in range(min(9, n_components)):
                    # Reshape eigenface for display
                    eigenface = eigenfaces_model.eigenfaces[i].reshape(image_shape)
                    img_src = array_to_base64(eigenface)
                    
                    ef_gallery.append(
                        html.Div([
                            html.Img(src=img_src, style={"height": "100px", "width": "100px", 
                                                         "margin": "5px", "border": "1px solid #ddd"}),
                            html.P(f"Eigenface {i+1}", style={"fontSize": "12px", "textAlign": "center"})
                        ], className="col-md-4")
                    )
                
                eigenfaces_grid = html.Div(ef_gallery, className="row")
                
                # Create results text
                results_text = html.Div([
                    html.P([
                        "Training completed in ",
                        html.Strong(f"{results['execution_time']:.2f} seconds")
                    ]),
                    html.P([
                        "Components: ",
                        html.Strong(f"{n_components}")
                    ]),
                    html.P([
                        "Threshold: ",
                        html.Strong(f"{results['threshold']:.4f}")
                    ])
                ])
                
                # Serialize model for storage
                serialized_model = {
                    "eigenfaces": eigenfaces_model.eigenfaces.tolist(),
                    "mean_face": eigenfaces_model.mean_face.tolist(),
                    "weights": eigenfaces_model.weights.tolist(),
                    "gallery_ids": gallery_ids,
                    "threshold": threshold,
                    "n_components": n_components,
                    "performance": results['performance']
                }
                
                # Add model to results
                results['model'] = serialized_model
                
                # Set progress to complete
                progress["value"] = 100
                progress["message"] = "Training completed successfully!"
                
                return progress["value"], progress["message"], results_text, results, metrics_html, conf_mat_fig, eigenfaces_grid
            
            except Exception as e:
                # Handle errors
                progress["value"] = 0
                progress["message"] = f"Error: {str(e)}"
                
                return progress["value"], progress["message"], f"Error during training: {str(e)}", None, None, create_confusion_matrix_figure([[0, 0], [0, 0]]), None
        
        # If find best threshold button clicked
        elif trigger_id == "ef-find-threshold-btn" and find_clicks:
            # Start threshold finding
            progress["value"] = 5
            progress["message"] = "Finding best threshold..."
            
            try:
                # Update progress
                def update_progress(value, message):
                    progress["value"] = value
                    progress["message"] = message
                
                # Find best threshold
                best_threshold, results, eigenfaces_model = ef.find_best_threshold(
                    gallery=gallery,
                    enrolled_probes=enrolled_probes,
                    non_enrolled_probes=non_enrolled_probes,
                    n_components=n_components,
                    progress_callback=update_progress
                )
                
                # Create metrics display
                metrics_html = create_metrics_table(results['performance'])
                
                # Create confusion matrix figure
                conf_mat_fig = create_confusion_matrix_figure(results['confusion_matrix'])
                
                # Create eigenfaces gallery
                ef_gallery = []
                for i in range(min(9, n_components)):
                    # Reshape eigenface for display
                    eigenface = eigenfaces_model.eigenfaces[i].reshape(image_shape)
                    img_src = array_to_base64(eigenface)
                    
                    ef_gallery.append(
                        html.Div([
                            html.Img(src=img_src, style={"height": "100px", "width": "100px", 
                                                         "margin": "5px", "border": "1px solid #ddd"}),
                            html.P(f"Eigenface {i+1}", style={"fontSize": "12px", "textAlign": "center"})
                        ], className="col-md-4")
                    )
                
                eigenfaces_grid = html.Div(ef_gallery, className="row")
                
                # Create results text
                results_text = html.Div([
                    html.P([
                        "Best threshold found: ",
                        html.Strong(f"{best_threshold:.4f}")
                    ]),
                    html.P([
                        "Completed in ",
                        html.Strong(f"{results['execution_time']:.2f} seconds")
                    ]),
                    html.P([
                        "Components: ",
                        html.Strong(f"{n_components}")
                    ])
                ])
                
                # Serialize model for storage
                serialized_model = {
                    "eigenfaces": eigenfaces_model.eigenfaces.tolist(),
                    "mean_face": eigenfaces_model.mean_face.tolist(),
                    "weights": eigenfaces_model.weights.tolist(),
                    "gallery_ids": gallery_ids,
                    "threshold": best_threshold,
                    "n_components": n_components,
                    "performance": results['performance']
                }
                
                # Add model to results
                results['model'] = serialized_model
                
                # Set progress to complete
                progress["value"] = 100
                progress["message"] = "Best threshold found successfully!"
                
                return progress["value"], progress["message"], results_text, results, metrics_html, conf_mat_fig, eigenfaces_grid
            
            except Exception as e:
                # Handle errors
                progress["value"] = 0
                progress["message"] = f"Error: {str(e)}"
                
                return progress["value"], progress["message"], f"Error finding best threshold: {str(e)}", None, None, create_confusion_matrix_figure([[0, 0], [0, 0]]), None
        
        # If interval triggered, just update progress
        elif trigger_id == "progress-interval":
            if current_results:
                # Training already completed
                return 100, "Training completed", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            else:
                # Still training or not started
                return progress["value"], progress["message"], dash.no_update, None, None, dash.no_update, None
        
        # Default return
        return 0, "", dash.no_update, None, None, dash.no_update, None

    @app.callback(
        [
            Output("ef-test-image", "src"),
            Output("ef-authenticate-btn", "disabled"),
            Output("ef-test-image-store", "data")
        ],
        Input("ef-select-image-btn", "n_clicks"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def select_random_ef_image(n_clicks, dataset_data):
        if not dataset_data or not n_clicks:
            return "", True, None
            
        # Deserialize dataset
        gallery, probes, gallery_ids, probe_ids, ground_truth, image_shape = deserialize_dataset(dataset_data)
        
        # Select random probe
        import random
        probe_idx = random.randint(0, len(probes) - 1)
        probe = probes[probe_idx]
        probe_id = probe_ids[probe_idx]
        is_enrolled = ground_truth[probe_idx]
        
        # Reshape for display
        probe_image = probe.reshape(image_shape)
        
        # Convert to base64
        img_src = array_to_base64(probe_image)
        
        # Store probe data
        probe_data = {
            "probe": probe.tolist(),
            "probe_id": probe_id,
            "is_enrolled": is_enrolled,
            "probe_idx": probe_idx
        }
        
        return img_src, False, probe_data
    
    @app.callback(
        [
            Output("ef-auth-result", "children"),
            Output("ef-best-match-image", "src"),
            Output("ef-match-details", "children"),
            Output("ef-same-person-container", "style"),
            Output("ef-same-person-images", "children")
        ],
        Input("ef-authenticate-btn", "n_clicks"),
        [
            State("dataset-store", "data"),
            State("ef-test-image-store", "data"),
            State("ef-results-store", "data")
        ],
        prevent_initial_call=True
    )
    def authenticate_ef_image(n_clicks, dataset_data, test_image_data, results_data):
        if not n_clicks or not dataset_data or not test_image_data or not results_data:
            return "No image or model selected", "", "", {"display": "none"}, []
            
        # Get model and threshold from results
        if 'model' not in results_data:
            return "No eigenfaces model trained", "", "", {"display": "none"}, []
            
        model_data = results_data['model']
        threshold = model_data.get('threshold', 0.8)
            
        # Deserialize data
        gallery, probes, gallery_ids, probe_ids, ground_truth, image_shape = deserialize_dataset(dataset_data)
        
        probe = np.array(test_image_data["probe"])
        probe_id = test_image_data["probe_id"]
        is_enrolled = test_image_data["is_enrolled"]
        
        try:
            # Reconstruct the eigenfaces model
            from src.core.eigenfaces import EigenfacesModel
            eigenfaces_model = EigenfacesModel()
            eigenfaces_model.eigenfaces = np.array(model_data['eigenfaces'])
            eigenfaces_model.mean_face = np.array(model_data['mean_face'])
            eigenfaces_model.weights = np.array(model_data['weights'])
            eigenfaces_model.gallery_ids = model_data['gallery_ids']
            eigenfaces_model.threshold = threshold
            
            # Authenticate
            is_authenticated, closest_idx, min_distance = ef.authenticate(
                probe, eigenfaces_model, threshold
            )
            
            # Get best match
            if closest_idx is not None:
                best_match_id = eigenfaces_model.gallery_ids[closest_idx]
                
                # Get the corresponding gallery image
                gallery_idx = gallery_ids.index(best_match_id)
                best_match = gallery[gallery_idx]
                
                # Reshape for display
                best_match_image = best_match.reshape(image_shape)
                
                # Convert to base64
                best_match_src = array_to_base64(best_match_image)
            else:
                best_match_id = "None"
                best_match_src = ""
            
            # Authentication result text
            if is_authenticated:
                result_html = html.Div([
                    html.H4("Authenticated ✅", className="text-success"),
                    html.P([
                        "The person was ",
                        html.Strong("successfully authenticated"),
                        " with a distance of ",
                        html.Strong(f"{min_distance:.4f}")
                    ]),
                    html.P([
                        "Ground truth: ",
                        html.Strong("Enrolled" if is_enrolled else "Not enrolled", 
                                   className="text-success" if is_enrolled else "text-danger")
                    ])
                ])
            else:
                result_html = html.Div([
                    html.H4("Not Authenticated ❌", className="text-danger"),
                    html.P([
                        "Authentication failed with a distance of ",
                        html.Strong(f"{min_distance:.4f}"),
                        " (threshold: ",
                        html.Strong(f"{threshold:.4f}"),
                        ")"
                    ]),
                    html.P([
                        "Ground truth: ",
                        html.Strong("Enrolled" if is_enrolled else "Not enrolled", 
                                   className="text-success" if is_enrolled else "text-danger")
                    ])
                ])
            
            # Match details
            match_details = f"Best match: ID {best_match_id} (Distance: {min_distance:.4f})"
            
            # Find other images of the same person
            same_person_images = []
            
            if is_authenticated:
                # Find all probes with the same ID as the best match
                same_id_indices = [i for i, pid in enumerate(probe_ids) if pid == best_match_id]
                
                # Create image components
                for idx in same_id_indices[:4]:  # Limit to 4 images
                    same_person_probe = probes[idx].reshape(image_shape)
                    img_src = array_to_base64(same_person_probe)
                    
                    img_div = html.Div([
                        html.Img(src=img_src, style={"height": "150px", "width": "150px", 
                                                     "margin": "5px", "border": "1px solid #ddd"})
                    ])
                    
                    same_person_images.append(img_div)
            
            # Show same person container only if authenticated
            same_person_style = {"display": "block"} if is_authenticated and same_person_images else {"display": "none"}
            
            return result_html, best_match_src, match_details, same_person_style, same_person_images
            
        except Exception as e:
            return f"Error: {str(e)}", "", "", {"display": "none"}, []

def register_comparison_callbacks(app):
    """
    Register callbacks for the comparison component.
    
    Args:
        app: Dash application instance
    """
    @app.callback(
        [
            Output("comp-bf-metrics", "children"),
            Output("comp-ef-metrics", "children"),
            Output("comp-results-text", "children"),
            Output("comp-metrics-table", "children"),
            Output("comp-roc-curve", "figure")
        ],
        Input("comp-evaluate-btn", "n_clicks"),
        [
            State("dataset-store", "data"),
            State("bf-results-store", "data"),
            State("ef-results-store", "data")
        ],
        prevent_initial_call=True
    )
    def compare_methods(n_clicks, dataset_data, bf_results, ef_results):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        # Check if all data is available
        if not dataset_data:
            return None, None, "Please load a dataset first", None, {}
            
        if not bf_results:
            return None, None, "Please evaluate brute force method first", None, {}
            
        if not ef_results:
            return None, None, "Please train eigenfaces model first", None, {}
            
        try:
            # Deserialize dataset
            gallery, probes, gallery_ids, probe_ids, ground_truth, image_shape = deserialize_dataset(dataset_data)
            
            # Get metrics from both methods
            bf_metrics = bf_results.get('performance', {})
            ef_metrics = ef_results.get('performance', {})
            
            # Create metrics tables
            bf_metrics_table = create_metrics_table(bf_metrics)
            ef_metrics_table = create_metrics_table(ef_metrics)
            
            # Create comparison table
            comparison_rows = []
            for metric_name in set(bf_metrics.keys()).intersection(set(ef_metrics.keys())):
                bf_value = bf_metrics.get(metric_name, 0)
                ef_value = ef_metrics.get(metric_name, 0)
                
                # Determine which is better
                if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'true_positive_rate', 'true_negative_rate']:
                    # Higher is better
                    bf_better = bf_value > ef_value
                elif metric_name in ['false_positive_rate', 'false_negative_rate', 'equal_error_rate']:
                    # Lower is better
                    bf_better = bf_value < ef_value
                else:
                    # Default: higher is better
                    bf_better = bf_value > ef_value
                
                bf_formatted = f"{bf_value:.4f}" if isinstance(bf_value, float) else str(bf_value)
                ef_formatted = f"{ef_value:.4f}" if isinstance(ef_value, float) else str(ef_value)
                
                comparison_rows.append(
                    html.Tr([
                        html.Td(metric_name.replace('_', ' ').title()),
                        html.Td(bf_formatted, style={"fontWeight": "bold", "color": "green"} if bf_better else {}),
                        html.Td(ef_formatted, style={"fontWeight": "bold", "color": "green"} if not bf_better else {})
                    ])
                )
            
            comparison_table = dbc.Table(
                [
                    html.Thead(
                        html.Tr([
                            html.Th("Metric"),
                            html.Th("Brute Force"),
                            html.Th("Eigenfaces")
                        ])
                    ),
                    html.Tbody(comparison_rows)
                ],
                bordered=True,
                hover=True,
                striped=True
            )
            
            # Create results text
            results_text = html.Div([
                html.H4("Comparison Results"),
                html.P([
                    "Brute Force: ",
                    html.Strong(f"threshold = {bf_results.get('threshold', 0):.4f}")
                ]),
                html.P([
                    "Eigenfaces: ",
                    html.Strong(f"components = {ef_results.get('model', {}).get('n_components', 0)}"),
                    ", ",
                    html.Strong(f"threshold = {ef_results.get('threshold', 0):.4f}")
                ])
            ])
            
            # Create ROC curve
            roc_fig = create_roc_curve_comparison(bf_results, ef_results)
            
            return bf_metrics_table, ef_metrics_table, results_text, comparison_table, roc_fig
            
        except Exception as e:
            return None, None, f"Error during comparison: {str(e)}", None, {}
    
def create_roc_curve_comparison(bf_results, ef_results):
    """
    Create a ROC curve comparison figure.
    
    Args:
        bf_results: Brute force results dictionary
        ef_results: Eigenfaces results dictionary
        
    Returns:
        plotly.graph_objects.Figure: ROC curve comparison figure
    """
    fig = go.Figure()
    
    # Add Brute Force ROC curve if available
    if bf_results and 'roc_curve' in bf_results:
        roc_data = bf_results['roc_curve']
        fig.add_trace(go.Scatter(
            x=roc_data['fpr'],
            y=roc_data['tpr'],
            mode='lines',
            name='Brute Force',
            line=dict(color='blue', width=2)
        ))
    
    # Add Eigenfaces ROC curve if available
    if ef_results and 'roc_curve' in ef_results:
        roc_data = ef_results['roc_curve']
        fig.add_trace(go.Scatter(
            x=roc_data['fpr'],
            y=roc_data['tpr'],
            mode='lines',
            name='Eigenfaces',
            line=dict(color='red', width=2)
        ))
    
    # Add random guess line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='grey', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.01, y=0.99),
        width=600,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Update axes
    fig.update_xaxes(range=[0, 1], constrain="domain")
    fig.update_yaxes(range=[0, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    
    return fig

def array_to_base64(img_array):
    """
    Convert numpy array to base64 encoded image.
    
    Args:
        img_array: Numpy array containing image data
        
    Returns:
        str: Base64 encoded image string
    """
    # Ensure array is properly scaled and in the right format
    if img_array.max() <= 1.0:
        # Améliorer le contraste pour les images sombres
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    elif img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    # Ensure the image has the right shape for PIL
    if len(img_array.shape) == 1:
        # Handle flattened images by trying to convert to 2D
        img_height = int(np.sqrt(img_array.shape[0]))
        img_array = img_array.reshape(img_height, img_height)
    
    # Améliorer le contraste si l'image est trop sombre
    if img_array.mean() < 50:  # Si l'image est très sombre
        # Étirer l'histogramme pour améliorer le contraste
        min_val = img_array.min()
        max_val = img_array.max()
        if max_val > min_val:  # Éviter division par zéro
            img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # Convert to PIL Image with explicit mode
    if len(img_array.shape) == 2:
        # Grayscale image
        img = Image.fromarray(img_array, mode='L')
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # RGB image
        img = Image.fromarray(img_array, mode='RGB')
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # RGBA image
        img = Image.fromarray(img_array, mode='RGBA')
    else:
        # Try to convert as grayscale if shape is unknown
        img = Image.fromarray(img_array, mode='L')
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    
    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{img_str}"

def deserialize_dataset(serialized):
    """
    Deserialize dataset from JSON.
    
    Args:
        serialized: JSON serialized dataset
        
    Returns:
        Tuple: (gallery, probes, gallery_ids, probe_ids, ground_truth, image_shape)
    """
    if not serialized:
        return None, None, None, None, None, None
        
    gallery = np.array(serialized["gallery"])
    probes = np.array(serialized["probes"])
    gallery_ids = serialized["gallery_ids"]
    probe_ids = serialized["probe_ids"]
    ground_truth = serialized["ground_truth"]
    image_shape = tuple(serialized["image_shape"])
    
    return gallery, probes, gallery_ids, probe_ids, ground_truth, image_shape 

def create_metrics_table(metrics):
    """
    Create a Bootstrap table to display performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
        
    Returns:
        html.Table: Bootstrap styled table with metrics
    """
    rows = []
    for metric_name, metric_value in metrics.items():
        formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
        rows.append(
            html.Tr([
                html.Td(metric_name.replace('_', ' ').title()),
                html.Td(formatted_value)
            ])
        )
    
    table = dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Metric"),
                    html.Th("Value")
                ])
            ),
            html.Tbody(rows)
        ],
        bordered=True,
        hover=True,
        striped=True,
        size="sm"
    )
    
    return table 