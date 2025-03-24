"""
Interface graphique pour le système d'authentification faciale avec Dash.

Ce module fournit une interface web interactive pour tester et comparer
les différentes méthodes d'authentification.
"""

import sys
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import base64
import io
from PIL import Image
import json
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dash.exceptions import PreventUpdate

# Ajouter le répertoire parent au chemin de recherche pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Initialiser l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Pour le déploiement

# Files d'attente pour la communication entre threads
progress_queue = queue.Queue()
result_queue = queue.Queue()

# Variables globales pour stocker l'état
global_state = {
    'dataset': None,
    'gallery_processed': None,
    'probes_processed': None,
    'eigenfaces_model': None,
    'cnn_model': None,
    'current_probe': None,
    'current_probe_identity': None,
    'current_ground_truth': None,
    'is_processing': False
}

# Layout de l'application
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Système d'Authentification Faciale", className="text-center my-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Configuration"),
                dbc.CardBody([
                    html.Div([
                        html.Label("Dataset:"),
                        dbc.RadioItems(
                            id="dataset-select",
                            options=[
                                {"label": "Dataset 1", "value": 1},
                                {"label": "Dataset 2", "value": 2}
                            ],
                            value=1,
                            inline=True
                        )
                    ], className="mb-3"),
                    
                    html.Div([
                        html.Label("Méthode:"),
                        dbc.RadioItems(
                            id="method-select",
                            options=[
                                {"label": "Force Brute", "value": "brute_force"},
                                {"label": "Eigenfaces", "value": "eigenfaces"},
                                {"label": "CNN", "value": "cnn"}
                            ],
                            value="brute_force",
                            inline=True
                        )
                    ], className="mb-3"),
                    
                    dbc.Button("Charger les données", id="load-data-btn", color="primary", className="me-2"),
                    dbc.Button("Évaluer les performances", id="evaluate-btn", color="success", disabled=True)
                ])
            ], className="mb-4")
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Progression"),
                dbc.CardBody([
                    dbc.Progress(id="progress-bar", value=0, style={"height": "20px", "marginBottom": "10px"}),
                    html.P(id="progress-text", children="")
                ])
            ], className="mb-4")
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Test d'authentification"),
                dbc.CardBody([
                    html.Div([
                        html.Img(id="probe-image", style={
                            "width": "150px", 
                            "height": "150px", 
                            "marginBottom": "10px",
                            "border": "1px solid #ddd"
                        }),
                        html.Div([
                            dbc.Button("Sélectionner une image", id="select-image-btn", color="info", className="me-2", disabled=True),
                            dbc.Button("Authentifier", id="authenticate-btn", color="danger", disabled=True)
                        ], style={"marginTop": "10px"})
                    ], style={"textAlign": "center"})
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Résultats"),
                dbc.CardBody(
                    html.Pre(id="result-text", style={
                        "whiteSpace": "pre-wrap", 
                        "wordBreak": "break-all",
                        "maxHeight": "200px",
                        "overflowY": "auto"
                    })
                )
            ])
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Visualisation"),
                dbc.CardBody([
                    dbc.ButtonGroup([
                        dbc.Button("Visualiser les métriques", id="viz-metrics-btn", color="secondary", className="me-2", disabled=True),
                        dbc.Button("Visualiser les Eigenfaces", id="viz-eigenfaces-btn", color="secondary", className="me-2", disabled=True),
                        dbc.Button("Visualiser les activations CNN", id="viz-cnn-btn", color="secondary", disabled=True)
                    ])
                ])
            ], className="mt-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="visualization-graph")
        ], width=12)
    ]),
    
    # Interval pour mise à jour de la progression asynchrone
    dcc.Interval(id='progress-interval', interval=1000, n_intervals=0, disabled=False),
    
    # Stockage client-side de certaines données légères
    dcc.Store(id='probe-info-store'),
    dcc.Store(id='evaluation-results-store'),
    
    # Modals pour les messages
    dbc.Modal(
        [
            dbc.ModalHeader("Information"),
            dbc.ModalBody(id="modal-body"),
            dbc.ModalFooter(
                dbc.Button("Fermer", id="close-modal", className="ms-auto", n_clicks=0)
            ),
        ],
        id="info-modal",
        is_open=False,
    ),
], fluid=True)

# Importation différée des modules pour éviter les problèmes d'initialisation
def load_dependencies():
    global brute_force, eigenfaces, deep_learning, load_dataset, evaluate_authentication_method, compare_methods
    try:
        from src.utils import load_dataset, evaluate_authentication_method, compare_methods
        import src.brute_force as brute_force
        import src.eigenfaces as eigenfaces
        import src.deep_learning as deep_learning
        return True
    except Exception as e:
        print(f"Erreur lors du chargement des dépendances: {e}")
        return False

# Fonctions utilitaires pour les opérations asynchrones
def run_async(func, callback=None):
    """Exécute une fonction de manière asynchrone dans un thread séparé"""
    def wrapper(*args, **kwargs):
        global global_state
        global_state['is_processing'] = True
        
        # Vider les files d'attente pour éviter des résultats résiduels
        while not progress_queue.empty():
            try:
                progress_queue.get_nowait()
            except queue.Empty:
                break
                
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                break
        
        def update_progress(percentage, message):
            try:
                progress_queue.put((percentage, message), block=False)
            except queue.Full:
                pass
        
        def task_done():
            global global_state
            global_state['is_processing'] = False
            if callback:
                callback()
        
        def run_func():
            try:
                result = func(*args, **kwargs, progress_callback=update_progress)
                try:
                    result_queue.put(("success", result), block=False)
                except queue.Full:
                    pass
            except Exception as e:
                print(f"Erreur dans la fonction asynchrone: {e}")
                try:
                    result_queue.put(("error", str(e)), block=False)
                except queue.Full:
                    pass
            finally:
                task_done()
        
        threading.Thread(target=run_func, daemon=True).start()
    
    return wrapper

# Fonction asynchrone pour charger les données
@run_async
def async_load_dataset(dataset_num, progress_callback=None):
    """Charge les données du dataset de manière asynchrone"""
    # Charger les dépendances si ce n'est pas déjà fait
    if not 'load_dataset' in globals():
        if not load_dependencies():
            return {'error': "Impossible de charger les dépendances nécessaires"}
    
    progress_callback(10, f"Chargement du dataset {dataset_num}...")
    
    # Charger le dataset
    dataset = load_dataset(dataset_num)
    progress_callback(50, "Prétraitement des images...")
    
    # Prétraiter les images
    gallery_processed, probes_processed = dataset.preprocess_images()
    progress_callback(90, "Finalisation...")
    
    # Mettre à jour l'état global
    global global_state
    global_state['dataset'] = dataset
    global_state['gallery_processed'] = gallery_processed
    global_state['probes_processed'] = probes_processed
    global_state['eigenfaces_model'] = None
    global_state['cnn_model'] = None
    
    progress_callback(100, "Chargement terminé")
    return {
        'success': True,
        'message': f"Dataset {dataset_num} chargé avec succès!",
        'gallery_shape': gallery_processed.shape,
        'probes_shape': probes_processed.shape
    }

# Fonction asynchrone pour évaluer les performances
@run_async
def async_evaluate_performance(method, progress_callback=None):
    """Évalue les performances d'une méthode de manière asynchrone"""
    # Charger les dépendances si ce n'est pas déjà fait
    if not 'evaluate_authentication_method' in globals():
        if not load_dependencies():
            return {'error': "Impossible de charger les dépendances nécessaires"}
    
    global global_state
    dataset = global_state['dataset']
    gallery_processed = global_state['gallery_processed']
    probes_processed = global_state['probes_processed']
    
    # Vérifier si nous avons déjà calculé les résultats pour cette méthode
    if 'last_evaluated_method' in global_state and global_state['last_evaluated_method'] == method:
        if 'last_evaluation_results' in global_state:
            results = global_state['last_evaluation_results']
            # Si nous avons déjà les résultats complets, les retourner immédiatement
            if 'metrics' in results and 'output' in results:
                progress_callback(100, f"Résultats précédemment calculés pour {method}")
                return results
    
    progress_callback(10, f"Préparation pour l'évaluation de la méthode {method}...")
    
    # Séparer les probes en "enregistrés" et "non enregistrés"
    enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if gt]
    non_enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if not gt]
    
    enrolled_probes = probes_processed[enrolled_indices]
    non_enrolled_probes = probes_processed[non_enrolled_indices]
    
    progress_callback(20, "Calcul du rayon optimal...")
    
    results = {}
    
    if method == "brute_force":
        # Trouver le meilleur rayon
        radius = brute_force.find_best_radius(
            gallery_processed, 
            enrolled_probes, 
            non_enrolled_probes
        )
        results['radius'] = radius
        
        progress_callback(60, "Évaluation des performances...")
        
        # Évaluer les performances
        metrics = evaluate_authentication_method(
            lambda p, g, **kwargs: brute_force.authenticate(p, g, radius),
            gallery_processed,
            probes_processed,
            dataset.ground_truth
        )
        
        results['metrics'] = metrics.to_dict()
        results['output'] = f"Rayon optimal: {radius:.4f}\n\n{metrics}"
        
    elif method == "eigenfaces":
        # Préparer les données pour Eigenfaces (aplatir les images)
        gallery_flat = gallery_processed.reshape(gallery_processed.shape[0], -1)
        enrolled_flat = enrolled_probes.reshape(enrolled_probes.shape[0], -1)
        non_enrolled_flat = non_enrolled_probes.reshape(non_enrolled_probes.shape[0], -1)
        
        progress_callback(40, "Calcul des eigenfaces et du rayon optimal...")
        
        try:
            # Trouver le meilleur rayon et entraîner le modèle
            radius, eigenfaces_model = eigenfaces.find_best_radius(
                gallery_flat,
                enrolled_flat,
                non_enrolled_flat
            )
            global_state['eigenfaces_model'] = eigenfaces_model
            results['radius'] = radius
            results['n_components'] = eigenfaces_model.n_components
            
            progress_callback(70, "Évaluation des performances...")
            
            # Évaluer les performances
            metrics = evaluate_authentication_method(
                lambda p, g, **kwargs: eigenfaces.authenticate(p, g, radius, eigenfaces_model),
                gallery_flat,
                probes_processed.reshape(probes_processed.shape[0], -1),
                dataset.ground_truth
            )
            
            results['metrics'] = metrics.to_dict()
            results['output'] = f"Rayon optimal: {radius:.4f}\nNombre de composantes utilisées: {eigenfaces_model.n_components}\n\n{metrics}"
        except Exception as e:
            results['error'] = str(e)
            results['output'] = f"Erreur lors de l'évaluation de la méthode Eigenfaces: {str(e)}"
            print(f"Erreur pour eigenfaces: {e}")
        
    elif method == "cnn":
        progress_callback(50, "Initialisation du modèle CNN...")
        
        # Créer un modèle CNN (dans un cas réel, on l'entraînerait ici)
        cnn_model = deep_learning.CNNAuthenticator()
        global_state['cnn_model'] = cnn_model
        
        results['output'] = "Méthode CNN non implémentée complètement dans cette démo interactive.\nDans un cas réel, le modèle serait entraîné ici."
    
    # Stocker les résultats dans l'état global pour visualisation ultérieure
    global_state['last_evaluation_results'] = results
    global_state['last_evaluated_method'] = method
    
    progress_callback(100, "Évaluation terminée")
    return results

# Fonction asynchrone pour l'authentification
@run_async
def async_authenticate(method, progress_callback=None):
    """Authentifie une image de manière asynchrone"""
    # Charger les dépendances si ce n'est pas déjà fait
    if not all(m in globals() for m in ['brute_force', 'eigenfaces', 'deep_learning']):
        if not load_dependencies():
            return {'error': "Impossible de charger les dépendances nécessaires"}
    
    global global_state
    gallery_processed = global_state['gallery_processed']
    current_probe = global_state['current_probe']
    dataset = global_state['dataset']
    
    if current_probe is None:
        return {'error': "Aucune image sélectionnée pour l'authentification"}
    
    progress_callback(10, f"Préparation pour l'authentification avec la méthode {method}...")
    
    result = {
        'authenticated': False,
        'method': method
    }
    
    if method == "brute_force":
        # Utiliser le rayon déjà calculé s'il existe
        if 'last_evaluation_results' in global_state and global_state.get('last_evaluated_method') == method:
            radius = global_state['last_evaluation_results'].get('radius')
            if radius is not None:
                progress_callback(40, f"Utilisation du rayon précédemment calculé: {radius:.4f}...")
            else:
                # Recalculer le rayon
                progress_callback(30, "Calcul du rayon optimal...")
                
                # Aplatir les images
                probe_flat = current_probe.flatten() / 255.0
                gallery_flat = gallery_processed.reshape(gallery_processed.shape[0], -1)
                
                # Séparer les probes
                enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if gt]
                non_enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if not gt]
                
                enrolled_probes = global_state['probes_processed'][enrolled_indices].reshape(len(enrolled_indices), -1)
                non_enrolled_probes = global_state['probes_processed'][non_enrolled_indices].reshape(len(non_enrolled_indices), -1)
                
                radius = brute_force.find_best_radius(
                    gallery_flat,
                    enrolled_probes,
                    non_enrolled_probes
                )
        else:
            # Recalculer le rayon
            progress_callback(30, "Calcul du rayon optimal...")
            
            # Aplatir les images
            probe_flat = current_probe.flatten() / 255.0
            gallery_flat = gallery_processed.reshape(gallery_processed.shape[0], -1)
            
            # Séparer les probes
            enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if gt]
            non_enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if not gt]
            
            enrolled_probes = global_state['probes_processed'][enrolled_indices].reshape(len(enrolled_indices), -1)
            non_enrolled_probes = global_state['probes_processed'][non_enrolled_indices].reshape(len(non_enrolled_indices), -1)
            
            radius = brute_force.find_best_radius(
                gallery_flat,
                enrolled_probes,
                non_enrolled_probes
            )
        
        progress_callback(70, "Authentification...")
        
        # Aplatir l'image si ce n'est pas déjà fait
        if len(current_probe.shape) > 1:
            probe_flat = current_probe.flatten() / 255.0
        else:
            probe_flat = current_probe / 255.0
            
        gallery_flat = gallery_processed.reshape(gallery_processed.shape[0], -1)
        
        # Authentifier
        authenticated = brute_force.authenticate(probe_flat, gallery_flat, radius)
        result['authenticated'] = authenticated
        
    elif method == "eigenfaces":
        eigenfaces_model = global_state.get('eigenfaces_model')
        if eigenfaces_model is None:
            result['error'] = "Le modèle Eigenfaces n'est pas encore entraîné. Évaluez d'abord les performances."
            progress_callback(100, "Erreur")
            return result
        
        # Utiliser le rayon déjà calculé s'il existe
        if 'last_evaluation_results' in global_state and global_state.get('last_evaluated_method') == method:
            radius = global_state['last_evaluation_results'].get('radius')
            if radius is not None:
                progress_callback(40, f"Utilisation du rayon précédemment calculé: {radius:.4f}...")
            else:
                # Recalculer le rayon
                progress_callback(30, "Calcul du rayon optimal...")
                
                # Aplatir l'image
                probe_flat = current_probe.flatten() / 255.0
                gallery_flat = gallery_processed.reshape(gallery_processed.shape[0], -1)
                
                # Séparer les probes
                enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if gt]
                non_enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if not gt]
                
                enrolled_probes = global_state['probes_processed'][enrolled_indices].reshape(len(enrolled_indices), -1)
                non_enrolled_probes = global_state['probes_processed'][non_enrolled_indices].reshape(len(non_enrolled_indices), -1)
                
                radius, _ = eigenfaces.find_best_radius(
                    gallery_flat,
                    enrolled_probes,
                    non_enrolled_probes
                )
        else:
            # Recalculer le rayon
            progress_callback(30, "Calcul du rayon optimal...")
            
            # Aplatir l'image
            probe_flat = current_probe.flatten() / 255.0
            gallery_flat = gallery_processed.reshape(gallery_processed.shape[0], -1)
            
            # Séparer les probes
            enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if gt]
            non_enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if not gt]
            
            enrolled_probes = global_state['probes_processed'][enrolled_indices].reshape(len(enrolled_indices), -1)
            non_enrolled_probes = global_state['probes_processed'][non_enrolled_indices].reshape(len(non_enrolled_indices), -1)
            
            radius, _ = eigenfaces.find_best_radius(
                gallery_flat,
                enrolled_probes,
                non_enrolled_probes
            )
        
        progress_callback(70, "Authentification...")
        
        # Aplatir l'image si ce n'est pas déjà fait
        if len(current_probe.shape) > 1:
            probe_flat = current_probe.flatten() / 255.0
        else:
            probe_flat = current_probe / 255.0
            
        gallery_flat = gallery_processed.reshape(gallery_processed.shape[0], -1)
        
        # Authentifier
        authenticated = eigenfaces.authenticate(probe_flat, gallery_flat, radius, eigenfaces_model)
        result['authenticated'] = authenticated
        
    elif method == "cnn":
        result['error'] = "Méthode CNN non implémentée dans cette démo interactive."
    
    progress_callback(100, "Authentification terminée")
    return result

# Callback pour charger les données
@app.callback(
    [
        Output("result-text", "children"),
        Output("evaluate-btn", "disabled"),
        Output("select-image-btn", "disabled"),
        Output("authenticate-btn", "disabled"),
        Output("modal-body", "children"),
        Output("info-modal", "is_open")
    ],
    Input("load-data-btn", "n_clicks"),
    State("dataset-select", "value"),
    prevent_initial_call=True
)
def load_data(n_clicks, dataset_num):
    if n_clicks is None:
        raise PreventUpdate
    
    # Lancer le chargement asynchrone
    async_load_dataset(dataset_num)
    
    return (
        "Chargement des données en cours...", 
        True,  # Désactiver le bouton d'évaluation pendant le chargement
        True,  # Désactiver le bouton de sélection d'image pendant le chargement
        True,  # Désactiver le bouton d'authentification pendant le chargement
        "",
        False
    )

# Callback pour mettre à jour la progression et récupérer les résultats
@app.callback(
    [
        Output("progress-bar", "value"),
        Output("progress-text", "children"),
        Output("result-text", "children", allow_duplicate=True),
        Output("evaluate-btn", "disabled", allow_duplicate=True),
        Output("select-image-btn", "disabled", allow_duplicate=True),
        Output("authenticate-btn", "disabled", allow_duplicate=True),
        Output("viz-metrics-btn", "disabled"),
        Output("viz-eigenfaces-btn", "disabled"),
        Output("viz-cnn-btn", "disabled"),
        Output("evaluation-results-store", "data")
    ],
    Input("progress-interval", "n_intervals"),
    prevent_initial_call=True
)
def update_progress(n_intervals):
    # Éviter les mises à jour inutiles
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Vérifier s'il y a une mise à jour de progression
    progress_value = 0
    progress_message = ""
    progress_updated = False
    
    try:
        if not progress_queue.empty():
            progress_value, progress_message = progress_queue.get_nowait()
            progress_updated = True
    except queue.Empty:
        pass
    
    # Vérifier s'il y a un résultat
    result_text = dash.no_update
    enable_evaluate = global_state.get('dataset') is not None
    enable_select_image = global_state.get('dataset') is not None
    enable_authenticate = global_state.get('current_probe') is not None
    enable_viz_metrics = 'last_evaluation_results' in global_state and 'metrics' in global_state['last_evaluation_results']
    enable_viz_eigenfaces = global_state.get('eigenfaces_model') is not None
    enable_viz_cnn = global_state.get('cnn_model') is not None
    evaluation_results = dash.no_update
    result_updated = False
    
    try:
        if not result_queue.empty():
            status, result = result_queue.get_nowait()
            result_updated = True
            
            if status == "success":
                if 'output' in result:
                    result_text = result['output']
                    
                    # Si c'est un résultat d'évaluation, stocker les métriques
                    if 'metrics' in result:
                        evaluation_results = result
                        enable_viz_metrics = True
                elif 'message' in result:
                    result_text = result['message']
                    if 'gallery_shape' in result and 'probes_shape' in result:
                        gallery_shape = result['gallery_shape']
                        probes_shape = result['probes_shape']
                        result_text += f"\nGallery: {gallery_shape[0]} images\nProbes: {probes_shape[0]} images"
                elif 'authenticated' in result:
                    method = result.get('method', 'inconnue')
                    authenticated = result.get('authenticated', False)
                    result_text = f"Résultat d'authentification ({method}):\nAccès {'autorisé' if authenticated else 'refusé'}"
            elif status == "error":
                result_text = f"Erreur: {result}"
    except queue.Empty:
        pass
    
    # Si aucune mise à jour n'est nécessaire et que le traitement est terminé, ne rien faire
    if not progress_updated and not result_updated and not global_state['is_processing']:
        raise PreventUpdate
    
    # Si nous sommes ici, c'est qu'il y a eu une mise à jour ou que le traitement est en cours
    return (
        progress_value,
        progress_message,
        result_text,
        not enable_evaluate,
        not enable_select_image,
        not enable_authenticate,
        not enable_viz_metrics,
        not enable_viz_eigenfaces,
        not enable_viz_cnn,
        evaluation_results
    )

# Callback pour l'évaluation des performances
@app.callback(
    [
        Output("result-text", "children", allow_duplicate=True),
        Output("authenticate-btn", "disabled", allow_duplicate=True),
        Output("viz-metrics-btn", "disabled", allow_duplicate=True)
    ],
    Input("evaluate-btn", "n_clicks"),
    State("method-select", "value"),
    prevent_initial_call=True
)
def evaluate_performance(n_clicks, method):
    if n_clicks is None:
        raise PreventUpdate
    
    # Lancer l'évaluation asynchrone
    async_evaluate_performance(method)
    
    return "Évaluation des performances en cours...", True, True

# Callback pour sélectionner une image
@app.callback(
    [
        Output("probe-image", "src"),
        Output("probe-info-store", "data"),
        Output("result-text", "children", allow_duplicate=True),
        Output("authenticate-btn", "disabled", allow_duplicate=True)
    ],
    Input("select-image-btn", "n_clicks"),
    prevent_initial_call=True
)
def select_image(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    
    global global_state
    if global_state['dataset'] is None:
        return dash.no_update, dash.no_update, "Veuillez d'abord charger les données.", True
    
    # Choisir une image aléatoire des probes
    dataset = global_state['dataset']
    idx = np.random.randint(0, len(dataset.probes))
    current_probe = dataset.probes[idx]
    current_probe_identity = dataset.probe_identities[idx]
    current_ground_truth = dataset.ground_truth[idx]
    
    # Mettre à jour l'état global
    global_state['current_probe'] = current_probe
    global_state['current_probe_identity'] = current_probe_identity
    global_state['current_ground_truth'] = current_ground_truth
    
    # Convertir l'image en base64 pour l'afficher
    img = Image.fromarray(current_probe)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Stocker les informations de l'image
    probe_info = {
        'identity': current_probe_identity,
        'ground_truth': current_ground_truth
    }
    
    return (
        f"data:image/png;base64,{img_str}",
        probe_info,
        f"Image sélectionnée.\nIdentité: {current_probe_identity}\nDevrait être authentifié: {current_ground_truth}",
        False  # Activer le bouton d'authentification
    )

# Callback pour l'authentification
@app.callback(
    [
        Output("result-text", "children", allow_duplicate=True),
        Output("authenticate-btn", "disabled", allow_duplicate=True)
    ],
    Input("authenticate-btn", "n_clicks"),
    State("method-select", "value"),
    prevent_initial_call=True
)
def authenticate(n_clicks, method):
    if n_clicks is None:
        raise PreventUpdate
    
    global global_state
    if global_state.get('current_probe') is None:
        return "Veuillez d'abord sélectionner une image.", False
    
    # Lancer l'authentification asynchrone
    async_authenticate(method)
    
    return "Authentification en cours...", True

# Callback pour visualiser les eigenfaces
@app.callback(
    Output("visualization-graph", "figure"),
    Input("viz-eigenfaces-btn", "n_clicks"),
    prevent_initial_call=True
)
def visualize_eigenfaces(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    
    global global_state
    eigenfaces_model = global_state.get('eigenfaces_model')
    if eigenfaces_model is None:
        return go.Figure()
    
    # Créer une figure pour afficher les eigenfaces
    fig = make_subplots(
        rows=2, 
        cols=5,
        subplot_titles=["Visage moyen"] + [f"Eigenface {i+1}" for i in range(9)],
        specs=[[{"type": "heatmap"}]*5]*2
    )
    
    # Ajouter le visage moyen
    mean_face = eigenfaces_model.mean_face.reshape((150, 150))
    fig.add_trace(
        go.Heatmap(z=mean_face, colorscale='Greys', showscale=False),
        row=1, col=1
    )
    
    # Ajouter les eigenfaces
    for i in range(min(9, eigenfaces_model.n_components)):
        row, col = (i+1) // 5 + 1, (i+1) % 5 + 1
        if col == 0:
            col = 5
        eigenface = eigenfaces_model.eigenfaces[:, i].reshape((150, 150))
        fig.add_trace(
            go.Heatmap(z=eigenface, colorscale='RdBu', showscale=False),
            row=row, col=col
        )
    
    # Mise en page
    fig.update_layout(
        height=600,
        title_text="Visualisation des Eigenfaces",
        template="plotly_white"
    )
    
    return fig

# Callback pour visualiser les métriques
@app.callback(
    Output("visualization-graph", "figure", allow_duplicate=True),
    Input("viz-metrics-btn", "n_clicks"),
    prevent_initial_call=True
)
def visualize_metrics(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    
    global global_state
    if 'last_evaluation_results' not in global_state or 'metrics' not in global_state['last_evaluation_results']:
        return go.Figure()
    
    # Récupérer les métriques
    metrics = global_state['last_evaluation_results']['metrics']
    method = global_state.get('last_evaluated_method', 'inconnue')
    
    # Créer un graphique de comparaison des métriques
    fig = go.Figure()
    
    # Convertir les métriques en liste pour le graphique
    metric_names = ['Précision', 'Rappel', 'Exactitude', 'Spécificité', 'F1-Score']
    metric_values = [
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('accuracy', 0),
        metrics.get('specificity', 0),
        metrics.get('f1_score', 0)
    ]
    
    fig.add_trace(go.Bar(
        x=metric_names,
        y=metric_values,
        name=f'Méthode {method}',
        text=[f'{val:.2f}' for val in metric_values],
        textposition='auto',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ))
    
    fig.update_layout(
        title=f"Métriques de performance - Méthode {method}",
        xaxis_title="Métrique",
        yaxis_title="Valeur",
        yaxis=dict(range=[0, 1]),
        template="plotly_white"
    )
    
    return fig

# Point d'entrée pour lancer l'application
if __name__ == "__main__":
    # Essayer de précharger les dépendances avant de lancer l'application
    try:
        load_dependencies()
    except Exception as e:
        print(f"Avertissement: Certaines dépendances n'ont pas pu être chargées: {e}")
        print("Les dépendances seront chargées à la demande.")
    
    app.run(debug=True)
