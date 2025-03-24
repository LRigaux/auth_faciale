"""
Page spécifique pour la méthode Eigenfaces.

Ce module fournit la mise en page et les callbacks pour la page Eigenfaces.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from pathlib import Path

from src.ui.components.common import (
    create_card, create_progress_card, create_dataset_selector,
    create_authentication_test_card, create_evaluation_results_card,
    create_authentication_results_card, create_timing_card,
    create_visualization_card, create_modal, create_figure_display
)
from src.ui.utils.auth_functions import (
    fig_to_base64, ensure_directory, generate_performance_summary,
    format_duration
)
from src.ui.utils.async_utils import (
    run_async, update_global_state, get_global_state,
    register_figure, get_figure, has_figure
)

# Préfixe pour les identifiants des composants
PREFIX = "eigenfaces"

# Structure de la page
def create_layout():
    """
    Crée la mise en page de la page Eigenfaces.
    
    Returns:
        dbc.Container: Mise en page de la page
    """
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Méthode Eigenfaces", className="mb-4"),
                html.P([
                    "Cette méthode utilise l'Analyse en Composantes Principales (PCA) pour ",
                    "réduire la dimensionnalité des images avant de réaliser l'authentification."
                ], className="lead mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_dataset_selector(),
                create_eigenfaces_params_card(),
                create_timing_card(),
            ], width=4),
            
            dbc.Col([
                create_progress_card(),
                create_evaluation_results_card(),
                create_visualization_card(),
            ], width=8)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_authentication_test_card()
            ], width=6),
            
            dbc.Col([
                create_authentication_results_card()
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H3("Visualisations", className="mb-3"),
                dbc.Tabs([
                    dbc.Tab([
                        html.Div([
                            html.P("Les eigenfaces sont les vecteurs propres de la matrice de covariance des images, qui représentent les directions de plus grande variance dans l'espace des visages."),
                            create_figure_display(f"{PREFIX}-eigenfaces-fig")
                        ], className="mt-3")
                    ], label="Eigenfaces"),
                    
                    dbc.Tab([
                        html.Div([
                            html.P("Cette visualisation montre la variance expliquée par chaque composante principale et la variance cumulée."),
                            create_figure_display(f"{PREFIX}-variance-fig")
                        ], className="mt-3")
                    ], label="Variance Expliquée"),
                    
                    dbc.Tab([
                        html.Div([
                            html.P("Cette visualisation montre la distribution des distances pour les utilisateurs enregistrés et non enregistrés."),
                            create_figure_display(f"{PREFIX}-distances-fig")
                        ], className="mt-3")
                    ], label="Distribution des Distances"),
                    
                    dbc.Tab([
                        html.Div([
                            html.P("Cette visualisation montre la matrice de confusion pour les résultats de l'authentification."),
                            create_figure_display(f"{PREFIX}-confmat-fig")
                        ], className="mt-3")
                    ], label="Matrice de Confusion"),
                    
                    dbc.Tab([
                        html.Div([
                            html.P("Cette visualisation montre les performances pour différentes valeurs de rayon."),
                            create_figure_display(f"{PREFIX}-radius-perf-fig")
                        ], className="mt-3")
                    ], label="Performances / Rayon"),
                    
                    dbc.Tab([
                        html.Div([
                            html.P("Cette visualisation montre la qualité de reconstruction des images originales."),
                            create_figure_display(f"{PREFIX}-reconstruction-fig")
                        ], className="mt-3")
                    ], label="Reconstruction")
                ], id=f"{PREFIX}-visualization-tabs")
            ], width=12)
        ]),
        
        # Stockage client-side
        dcc.Store(id=f"{PREFIX}-results-store"),
        dcc.Store(id=f"{PREFIX}-probe-info-store"),
        
        # Interval pour les mises à jour
        dcc.Interval(id=f"{PREFIX}-progress-interval", interval=1000),
        
        # Modal
        create_modal(f"{PREFIX}-info", "Information Eigenfaces")
    ], fluid=True)

def create_eigenfaces_params_card():
    """
    Crée une carte avec les paramètres spécifiques à la méthode Eigenfaces.
    
    Returns:
        dbc.Card: Composant carte avec les paramètres
    """
    return create_card(
        "Paramètres Eigenfaces",
        [
            html.Div([
                html.Label("Nombre de composantes:"),
                dcc.Input(
                    id=f"{PREFIX}-n-components",
                    type="number",
                    min=1,
                    max=300,
                    step=1,
                    value=100,
                    className="form-control"
                ),
                html.Small("Laissez vide pour déterminer automatiquement", className="text-muted")
            ], className="mb-3"),
            
            html.Div([
                html.Label("Seuil de variance expliquée:"),
                dcc.Slider(
                    id=f"{PREFIX}-variance-threshold",
                    min=0.7,
                    max=0.99,
                    step=0.01,
                    value=0.95,
                    marks={0.7: '70%', 0.8: '80%', 0.9: '90%', 0.95: '95%', 0.99: '99%'}
                ),
            ], className="mb-3"),
            
            dbc.Button("Évaluer", id=f"{PREFIX}-evaluate-btn", color="success", disabled=True)
        ]
    )

# Callbacks spécifiques à la page Eigenfaces
@callback(
    [
        Output(f"{PREFIX}-evaluate-btn", "disabled"),
        Output("loading-time", "children"),
        Output("preprocessing-time", "children")
    ],
    [Input("load-data-btn", "n_clicks")],
    [State("dataset-select", "value")],
    prevent_initial_call=True
)
def on_load_data(n_clicks, dataset_num):
    """
    Callback pour le chargement des données.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Lancer le chargement des données de façon asynchrone
    from src.ui.utils.auth_functions import load_dataset_with_timing
    from src.ui.utils.async_utils import run_async
    
    @run_async
    def async_load_data(dataset_num, progress_callback=None):
        return load_dataset_with_timing(dataset_num, progress_callback)
    
    async_load_data(dataset_num)
    
    # Le bouton sera activé par le callback de progression
    return True, "", ""

@callback(
    [
        Output("progress-bar", "value"),
        Output("progress-text", "children"),
        Output(f"{PREFIX}-evaluate-btn", "disabled", allow_duplicate=True),
        Output("loading-time", "children", allow_duplicate=True),
        Output("preprocessing-time", "children", allow_duplicate=True),
        Output("select-image-btn", "disabled")
    ],
    [Input(f"{PREFIX}-progress-interval", "n_intervals")],
    prevent_initial_call=True
)
def update_progress_eigenfaces(n_intervals):
    """
    Callback pour la mise à jour de la progression.
    """
    from src.ui.utils.async_utils import get_progress_update, get_result_update, is_processing
    from src.ui.utils.auth_functions import format_duration
    
    # Récupérer la mise à jour de progression
    progress_value, progress_message = get_progress_update()
    
    # Récupérer le résultat éventuel
    status, result = get_result_update()
    
    if progress_value is None and status is None and not is_processing():
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Valeurs par défaut
    loading_time = ""
    preprocessing_time = ""
    enable_evaluate = dash.no_update
    enable_select_image = dash.no_update
    
    # Traiter le résultat si disponible
    if status == "success" and isinstance(result, dict):
        # Chargement de données
        if 'dataset' in result:
            # Mettre à jour l'état global
            update_global_state('dataset', result['dataset'])
            update_global_state('gallery_processed', result['gallery_processed'])
            update_global_state('probes_processed', result['probes_processed'])
            
            # Mise à jour des temps
            stats = result.get('statistics', {})
            loading_time = format_duration(stats.get('loading_time', 0))
            preprocessing_time = format_duration(stats.get('preprocessing_time', 0))
            
            # Activer les boutons
            enable_evaluate = False
            enable_select_image = False
    
    # Mettre à jour la progression
    if progress_value is not None:
        return progress_value, progress_message, enable_evaluate, loading_time, preprocessing_time, enable_select_image
    
    return dash.no_update, dash.no_update, enable_evaluate, loading_time, preprocessing_time, enable_select_image

@callback(
    [
        Output(f"{PREFIX}-results-store", "data"),
        Output("evaluation-result-text", "children"),
        Output("evaluation-time", "children")
    ],
    [Input(f"{PREFIX}-evaluate-btn", "n_clicks")],
    [
        State(f"{PREFIX}-n-components", "value"),
        State(f"{PREFIX}-variance-threshold", "value")
    ],
    prevent_initial_call=True
)
def on_evaluate_eigenfaces(n_clicks, n_components, variance_threshold):
    """
    Callback pour l'évaluation des performances de la méthode Eigenfaces.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Seuil de variance expliquée par défaut
    if variance_threshold is None:
        variance_threshold = 0.95
    
    @run_async
    def async_evaluate_eigenfaces(n_components, variance_threshold, progress_callback=None):
        """
        Évaluation asynchrone des performances de la méthode Eigenfaces.
        """
        import src.eigenfaces.authentication as eigenfaces
        from src.ui.utils.async_utils import get_global_state
        
        # Récupérer les données
        dataset = get_global_state('dataset')
        gallery_processed = get_global_state('gallery_processed')
        probes_processed = get_global_state('probes_processed')
        
        if dataset is None or gallery_processed is None or probes_processed is None:
            return {'error': "Veuillez d'abord charger les données"}
        
        # Convertir n_components en None si vide
        if n_components == "" or n_components <= 0:
            n_components = None
            
        progress_callback(5, "Préparation des données pour l'évaluation...")
        
        # Séparer les probes en "enregistrés" et "non enregistrés"
        enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if gt]
        non_enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if not gt]
        
        enrolled_probes = probes_processed[enrolled_indices]
        non_enrolled_probes = probes_processed[non_enrolled_indices]
        
        progress_callback(10, "Recherche du rayon optimal et évaluation...")
        
        # Créer le dossier pour enregistrer les figures
        figures_dir = ensure_directory("figures/eigenfaces")
        
        # Trouver le meilleur rayon et évaluer les performances
        radius, model, results = eigenfaces.find_best_radius(
            gallery_processed, 
            enrolled_probes, 
            non_enrolled_probes,
            n_components,
            progress_callback
        )
        
        # Ajouter des informations sur le modèle
        results['model_info'] = {
            'n_components': model.n_components,
            'variance_explained': np.sum(model.pca.explained_variance_ratio_)
        }
        
        # Stocker dans l'état global
        update_global_state('eigenfaces_model', model)
        update_global_state('last_evaluation_results', results)
        update_global_state('last_evaluated_method', 'eigenfaces')
        
        progress_callback(80, "Génération des visualisations...")
        
        # Générer et enregistrer les figures
        if hasattr(model, 'pca') and model.pca is not None:
            
            # Visualisation des eigenfaces
            try:
                img_shape = (int(np.sqrt(gallery_processed.shape[1])), int(np.sqrt(gallery_processed.shape[1])))
                fig_eigenfaces = model.visualize_eigenfaces(img_shape, n_eigenfaces=8)
                register_figure(f"{PREFIX}-eigenfaces-fig", fig_to_base64(fig_eigenfaces))
            except Exception as e:
                print(f"Erreur lors de la visualisation des eigenfaces: {e}")
            
            # Visualisation de la variance expliquée
            try:
                fig_variance = model.visualize_variance()
                register_figure(f"{PREFIX}-variance-fig", fig_to_base64(fig_variance))
            except Exception as e:
                print(f"Erreur lors de la visualisation de la variance: {e}")
            
            # Visualisation de la qualité de reconstruction
            try:
                fig_reconstruction = model.get_reconstruction_quality(gallery_processed[:5])
                register_figure(f"{PREFIX}-reconstruction-fig", fig_to_base64(fig_reconstruction))
            except Exception as e:
                print(f"Erreur lors de la visualisation de la reconstruction: {e}")
        
        # Visualisation des performances en fonction du rayon
        try:
            fig_radius_perf = eigenfaces.visualize_radius_performances(results)
            register_figure(f"{PREFIX}-radius-perf-fig", fig_to_base64(fig_radius_perf))
        except Exception as e:
            print(f"Erreur lors de la visualisation des performances en fonction du rayon: {e}")
        
        # Visualisation de la distribution des distances
        try:
            fig_distances = eigenfaces.visualize_distances_distribution(results)
            register_figure(f"{PREFIX}-distances-fig", fig_to_base64(fig_distances))
        except Exception as e:
            print(f"Erreur lors de la visualisation de la distribution des distances: {e}")
        
        # Visualisation de la matrice de confusion
        try:
            fig_confmat = eigenfaces.visualize_confusion_matrix(results)
            register_figure(f"{PREFIX}-confmat-fig", fig_to_base64(fig_confmat))
        except Exception as e:
            print(f"Erreur lors de la visualisation de la matrice de confusion: {e}")
        
        progress_callback(100, "Évaluation terminée")
        
        return results
    
    # Lancer l'évaluation asynchrone
    async_evaluate_eigenfaces(n_components, variance_threshold)
    
    # Le résultat sera mis à jour par le callback de progression
    return dash.no_update, "Évaluation en cours...", ""

# Callback pour la mise à jour des visualisations
@callback(
    [
        Output(f"{PREFIX}-eigenfaces-fig", "src"),
        Output(f"{PREFIX}-variance-fig", "src"),
        Output(f"{PREFIX}-distances-fig", "src"),
        Output(f"{PREFIX}-confmat-fig", "src"),
        Output(f"{PREFIX}-radius-perf-fig", "src"),
        Output(f"{PREFIX}-reconstruction-fig", "src"),
        Output("evaluation-result-text", "children", allow_duplicate=True),
        Output("evaluation-time", "children", allow_duplicate=True)
    ],
    [Input(f"{PREFIX}-progress-interval", "n_intervals")],
    prevent_initial_call=True
)
def update_visualizations(n_intervals):
    """
    Callback pour la mise à jour des visualisations.
    """
    from src.ui.utils.async_utils import get_result_update, is_processing
    from src.ui.utils.auth_functions import format_duration, generate_performance_summary
    
    # Récupérer le résultat éventuel
    status, result = get_result_update()
    
    if status is None and not is_processing():
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Vérifier si c'est un résultat d'évaluation Eigenfaces
    if status == "success" and isinstance(result, dict):
        if 'radius' in result and 'performance' in result and not 'dataset' in result:
            # Récupérer le temps d'exécution
            execution_time = format_duration(result.get('execution_time', 0))
            
            # Générer le résumé des performances
            summary = generate_performance_summary(result, 'eigenfaces')
            
            # Récupérer les figures enregistrées
            eigenfaces_fig = get_figure(f"{PREFIX}-eigenfaces-fig")
            variance_fig = get_figure(f"{PREFIX}-variance-fig")
            distances_fig = get_figure(f"{PREFIX}-distances-fig")
            confmat_fig = get_figure(f"{PREFIX}-confmat-fig")
            radius_perf_fig = get_figure(f"{PREFIX}-radius-perf-fig")
            reconstruction_fig = get_figure(f"{PREFIX}-reconstruction-fig")
            
            return (
                eigenfaces_fig or dash.no_update,
                variance_fig or dash.no_update,
                distances_fig or dash.no_update,
                confmat_fig or dash.no_update,
                radius_perf_fig or dash.no_update,
                reconstruction_fig or dash.no_update,
                summary,
                execution_time
            )
    
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Callback pour l'actualisation individuelle des figures
for fig_type in ["eigenfaces", "variance", "distances", "confmat", "radius-perf", "reconstruction"]:
    @callback(
        Output(f"{PREFIX}-{fig_type}-fig", "src", allow_duplicate=True),
        Input(f"refresh-{PREFIX}-{fig_type}-fig", "n_clicks"),
        prevent_initial_call=True
    )
    def refresh_figure(n_clicks, fig_type=fig_type):
        """
        Callback pour l'actualisation d'une figure.
        """
        if not n_clicks:
            return dash.no_update
            
        figure_data = get_figure(f"{PREFIX}-{fig_type}-fig")
        return figure_data or dash.no_update 