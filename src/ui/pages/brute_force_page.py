"""
Page spécifique pour la méthode Force Brute.

Ce module fournit la mise en page et les callbacks pour la page Force Brute.
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
    format_duration, array_to_base64
)
from src.ui.utils.async_utils import (
    run_async, update_global_state, get_global_state,
    register_figure, get_figure, has_figure
)

# Préfixe pour les identifiants des composants
PREFIX = "brute-force"

# Structure de la page
def create_layout():
    """
    Crée la mise en page de la page Force Brute.
    
    Returns:
        dbc.Container: Mise en page de la page
    """
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Méthode Force Brute", className="mb-4"),
                html.P([
                    "Cette méthode compare directement l'image de test avec toutes les images ",
                    "de la galerie en calculant une distance entre les vecteurs de pixels."
                ], className="lead mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_dataset_selector(),
                create_brute_force_params_card(),
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
                            html.P("Cette visualisation montre la distribution des distances pour les utilisateurs enregistrés et non enregistrés."),
                            html.Div(id=f"{PREFIX}-distances-container", className="mt-3")
                        ], className="mt-3")
                    ], label="Distribution des Distances"),
                    
                    dbc.Tab([
                        html.Div([
                            html.P("Cette visualisation montre la matrice de confusion pour les résultats de l'authentification."),
                            html.Div(id=f"{PREFIX}-confmat-container", className="mt-3")
                        ], className="mt-3")
                    ], label="Matrice de Confusion"),
                    
                    dbc.Tab([
                        html.Div([
                            html.P("Cette visualisation montre d'autres photos de la personne identifiée."),
                            html.Div(id=f"{PREFIX}-person-photos-container", className="mt-3")
                        ], className="mt-3")
                    ], label="Photos de la personne")
                ], id=f"{PREFIX}-visualization-tabs")
            ], width=12)
        ]),
        
        # Stockage client-side
        dcc.Store(id=f"{PREFIX}-results-store"),
        dcc.Store(id=f"{PREFIX}-probe-info-store"),
        
        # Interval pour les mises à jour
        dcc.Interval(id=f"{PREFIX}-progress-interval", interval=1000),
        
        # Modal pour afficher des informations supplémentaires
        create_modal(f"{PREFIX}-info", "Information Force Brute")
    ], fluid=True)

def create_brute_force_params_card():
    """
    Crée une carte avec les paramètres spécifiques à la méthode Force Brute.
    
    Returns:
        dbc.Card: Composant carte avec les paramètres
    """
    return create_card(
        "Paramètres Force Brute",
        [
            html.Div([
                html.Label("Norme de distance:"),
                dbc.RadioItems(
                    id=f"{PREFIX}-norm",
                    options=[
                        {"label": "L1 (Manhattan)", "value": "L1"},
                        {"label": "L2 (Euclidienne)", "value": "L2"},
                        {"label": "Inf (Chebyshev)", "value": "inf"}
                    ],
                    value="L2",
                    inline=True
                )
            ], className="mb-3"),
            
            dbc.Button("Évaluer", id=f"{PREFIX}-evaluate-btn", color="success", disabled=True)
        ]
    )

# Callbacks spécifiques à la page Force Brute
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
def update_progress_brute_force(n_intervals):
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
    [State(f"{PREFIX}-norm", "value")],
    prevent_initial_call=True
)
def on_evaluate_brute_force(n_clicks, norm):
    """
    Callback pour l'évaluation des performances de la méthode Force Brute.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    
    @run_async
    def async_evaluate_brute_force(norm, progress_callback=None):
        """
        Évaluation asynchrone des performances de la méthode Force Brute.
        """
        import src.brute_force.authentication as brute_force
        import matplotlib.pyplot as plt
        import seaborn as sns
        from src.ui.utils.async_utils import get_global_state
        
        # Récupérer les données
        dataset = get_global_state('dataset')
        gallery_processed = get_global_state('gallery_processed')
        probes_processed = get_global_state('probes_processed')
        
        if dataset is None or gallery_processed is None or probes_processed is None:
            return {'error': "Veuillez d'abord charger les données"}
        
        progress_callback(5, "Préparation des données pour l'évaluation...")
        
        # Séparer les probes en "enregistrés" et "non enregistrés"
        enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if gt]
        non_enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if not gt]
        
        enrolled_probes = probes_processed[enrolled_indices]
        non_enrolled_probes = probes_processed[non_enrolled_indices]
        
        progress_callback(10, "Recherche du rayon optimal...")
        
        start_time = time.time()
        
        # Aplatir les images pour le calcul des distances
        gallery_flat = gallery_processed.reshape(gallery_processed.shape[0], -1)
        enrolled_flat = enrolled_probes.reshape(enrolled_probes.shape[0], -1)
        non_enrolled_flat = non_enrolled_probes.reshape(non_enrolled_probes.shape[0], -1)
        
        # Calculer les distances minimales pour chaque probe
        progress_callback(20, "Calcul des distances pour les utilisateurs enregistrés...")
        
        enrolled_min_distances = []
        enrolled_closest_ids = []
        
        for i, probe in enumerate(enrolled_flat):
            distances = brute_force.compute_distances(gallery_flat, probe, norm)
            min_idx = np.argmin(distances)
            enrolled_min_distances.append(distances[min_idx])
            enrolled_closest_ids.append(dataset.gallery_ids[min_idx])
            
            if progress_callback and i % 10 == 0:
                progress_pct = 20 + (i / len(enrolled_flat)) * 30
                progress_callback(int(progress_pct), f"Traitement de la probe {i+1}/{len(enrolled_flat)}...")
        
        progress_callback(50, "Calcul des distances pour les utilisateurs non enregistrés...")
        
        non_enrolled_min_distances = []
        non_enrolled_closest_ids = []
        
        for i, probe in enumerate(non_enrolled_flat):
            distances = brute_force.compute_distances(gallery_flat, probe, norm)
            min_idx = np.argmin(distances)
            non_enrolled_min_distances.append(distances[min_idx])
            non_enrolled_closest_ids.append(dataset.gallery_ids[min_idx])
            
            if progress_callback and i % 10 == 0:
                progress_pct = 50 + (i / len(non_enrolled_flat)) * 30
                progress_callback(int(progress_pct), f"Traitement de la probe {i+1}/{len(non_enrolled_flat)}...")
        
        enrolled_min_distances = np.array(enrolled_min_distances)
        non_enrolled_min_distances = np.array(non_enrolled_min_distances)
        
        # Trouver le rayon optimal
        progress_callback(80, "Recherche du rayon optimal...")
        
        best_radius = 0
        best_metrics = None
        best_confmat = None
        
        # Tester différentes valeurs de rayon
        radius_candidates = np.percentile(enrolled_min_distances, np.arange(0, 101, 5))
        metrics_by_radius = []
        
        for radius in radius_candidates:
            # Calculer les métriques pour ce rayon
            tp = np.sum(enrolled_min_distances <= radius)
            fp = np.sum(non_enrolled_min_distances <= radius)
            tn = len(non_enrolled_min_distances) - fp
            fn = len(enrolled_min_distances) - tp
            
            # Calculer les métriques de performance
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'radius': radius,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1
            }
            
            metrics_by_radius.append(metrics)
            
            # Mettre à jour le meilleur rayon si nécessaire
            if best_metrics is None or metrics['accuracy'] > best_metrics['accuracy']:
                best_radius = radius
                best_metrics = metrics
                best_confmat = np.array([[tn, fp], [fn, tp]])
        
        execution_time = time.time() - start_time
        
        # Créer les visualisations
        progress_callback(90, "Création des visualisations...")
        
        # Visualisation de la distribution des distances
        fig_distances = plt.figure(figsize=(10, 6))
        ax = fig_distances.add_subplot(111)
        
        sns.histplot(enrolled_min_distances, bins=30, alpha=0.5, label='Utilisateurs enregistrés', ax=ax)
        sns.histplot(non_enrolled_min_distances, bins=30, alpha=0.5, label='Utilisateurs non enregistrés', ax=ax)
        
        ax.axvline(best_radius, color='red', linestyle='--', label=f'Rayon optimal: {best_radius:.2f}')
        ax.set_xlabel('Distance minimale')
        ax.set_ylabel('Nombre de probes')
        ax.set_title('Distribution des distances minimales')
        ax.legend()
        
        # Enregistrer la figure
        register_figure(f"{PREFIX}-distances-fig", fig_to_base64(fig_distances))
        
        # Visualisation de la matrice de confusion
        fig_confmat = plt.figure(figsize=(8, 6))
        ax = fig_confmat.add_subplot(111)
        
        sns.heatmap(best_confmat, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Négatif', 'Positif'], 
                   yticklabels=['Négatif', 'Positif'], ax=ax)
        
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Vérité')
        ax.set_title('Matrice de confusion')
        
        # Enregistrer la figure
        register_figure(f"{PREFIX}-confmat-fig", fig_to_base64(fig_confmat))
        
        # Préparer les résultats
        results = {
            'radius': best_radius,
            'norm': norm,
            'performance': best_metrics,
            'confusion_matrix': best_confmat.tolist(),
            'enrolled_distances': enrolled_min_distances.tolist(),
            'non_enrolled_distances': non_enrolled_min_distances.tolist(),
            'execution_time': execution_time,
            'enrolled_closest_ids': enrolled_closest_ids,
            'non_enrolled_closest_ids': non_enrolled_closest_ids
        }
        
        # Mettre à jour l'état global
        update_global_state('last_evaluation_results', results)
        update_global_state('last_evaluated_method', 'brute_force')
        
        progress_callback(100, "Évaluation terminée")
        
        return results
    
    # Lancer l'évaluation asynchrone
    async_evaluate_brute_force(norm)
    
    # Le résultat sera mis à jour par le callback de progression
    return dash.no_update, "Évaluation en cours...", ""

@callback(
    [
        Output(f"{PREFIX}-distances-container", "children"),
        Output(f"{PREFIX}-confmat-container", "children"),
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
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Vérifier si c'est un résultat d'évaluation Force Brute
    if status == "success" and isinstance(result, dict):
        if 'radius' in result and 'performance' in result and 'norm' in result and not 'dataset' in result:
            # Récupérer le temps d'exécution
            execution_time = format_duration(result.get('execution_time', 0))
            
            # Générer le résumé des performances
            summary = generate_performance_summary(result, 'brute_force')
            
            # Récupérer les figures enregistrées
            distances_fig = get_figure(f"{PREFIX}-distances-fig")
            confmat_fig = get_figure(f"{PREFIX}-confmat-fig")
            
            # Créer les conteneurs pour les visualisations
            distances_container = html.Img(src=distances_fig, style={"width": "100%"}) if distances_fig else html.Div("Aucune visualisation disponible")
            confmat_container = html.Img(src=confmat_fig, style={"width": "100%"}) if confmat_fig else html.Div("Aucune visualisation disponible")
            
            return distances_container, confmat_container, summary, execution_time
    
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

@callback(
    [
        Output(f"{PREFIX}-probe-info-store", "data"),
        Output("probe-image", "src")
    ],
    [Input("select-image-btn", "n_clicks")],
    prevent_initial_call=True
)
def on_select_image(n_clicks):
    """
    Callback pour la sélection d'une image aléatoire.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    # Récupérer le dataset et les données
    dataset = get_global_state('dataset')
    probes_processed = get_global_state('probes_processed')
    
    if dataset is None or probes_processed is None:
        return dash.no_update, dash.no_update
    
    # Sélectionner une image aléatoire
    try:
        probe, probe_id, is_enrolled = dataset.get_random_probe()
        
        # Trouver l'indice de cette probe dans le dataset
        probe_idx = next((i for i, pid in enumerate(dataset.probe_ids) if pid == probe_id), None)
        
        if probe_idx is not None:
            # Récupérer l'image prétraitée
            probe_processed = probes_processed[probe_idx]
            
            # Convertir l'image en base64
            probe_img_b64 = array_to_base64(probe_processed)
            
            # Stocker les informations sur la probe
            probe_info = {
                'probe_id': int(probe_id),
                'is_enrolled': bool(is_enrolled),
                'probe_idx': int(probe_idx)
            }
            
            # Mettre à jour l'état global
            update_global_state('current_probe', probe_processed)
            update_global_state('current_probe_identity', probe_id)
            update_global_state('current_ground_truth', is_enrolled)
            
            return probe_info, probe_img_b64
    except Exception as e:
        print(f"Erreur lors de la sélection de l'image: {e}")
    
    return dash.no_update, dash.no_update

@callback(
    [
        Output("authentication-result-text", "children"),
        Output("authentication-time", "children"),
        Output(f"{PREFIX}-person-photos-container", "children")
    ],
    [Input("authenticate-btn", "n_clicks")],
    [
        State(f"{PREFIX}-norm", "value"),
        State(f"{PREFIX}-results-store", "data"),
        State(f"{PREFIX}-probe-info-store", "data")
    ],
    prevent_initial_call=True
)
def on_authenticate(n_clicks, norm, results, probe_info):
    """
    Callback pour l'authentification d'une image.
    """
    if not n_clicks or probe_info is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Récupérer les données nécessaires
    gallery_processed = get_global_state('gallery_processed')
    current_probe = get_global_state('current_probe')
    dataset = get_global_state('dataset')
    
    if gallery_processed is None or current_probe is None or dataset is None:
        return "Erreur: Données manquantes", "", dash.no_update
    
    # Récupérer le rayon optimal si disponible
    radius = results.get('radius') if results else None
    
    if radius is None:
        return "Erreur: Veuillez d'abord évaluer les performances", "", dash.no_update
    
    # Effectuer l'authentification
    import src.brute_force.authentication as brute_force
    import time
    
    start_time = time.time()
    
    # Préparer les données
    probe_flat = current_probe.flatten()
    gallery_flat = gallery_processed.reshape(gallery_processed.shape[0], -1)
    
    # Calculer les distances
    distances = brute_force.compute_distances(gallery_flat, probe_flat, norm)
    
    # Trouver l'image la plus proche
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    
    # Vérifier si la distance est inférieure au rayon
    is_authenticated = min_distance <= radius
    closest_person_id = dataset.gallery_ids[min_idx]
    
    execution_time = time.time() - start_time
    
    # Préparer le résultat
    if is_authenticated:
        result_text = f"Authentification réussie! ✅\n\n"
        result_text += f"Distance minimale: {min_distance:.4f} (rayon: {radius:.4f})\n"
        result_text += f"Personne identifiée: ID {closest_person_id}\n"
        
        # Vérifier si l'authentification est correcte
        probe_id = probe_info.get('probe_id')
        is_enrolled = probe_info.get('is_enrolled')
        
        if is_enrolled:
            # Pour les utilisateurs enregistrés, vérifier si l'ID correspond
            if probe_id == closest_person_id:
                result_text += "\nL'identité correspond à la vérité terrain ✓"
            else:
                result_text += f"\nErreur d'identité! L'ID correct est {probe_id} ✗"
        else:
            # Pour les utilisateurs non enregistrés, toute authentification est une erreur
            result_text += "\nErreur! Cette personne ne devrait pas être authentifiée ✗"
    else:
        result_text = f"Authentification échouée ❌\n\n"
        result_text += f"Distance minimale: {min_distance:.4f} (rayon: {radius:.4f})\n"
        
        # Vérifier si le rejet est correct
        is_enrolled = probe_info.get('is_enrolled')
        
        if is_enrolled:
            # Pour les utilisateurs enregistrés, tout rejet est une erreur
            result_text += "\nErreur! Cette personne devrait être authentifiée ✗"
        else:
            # Pour les utilisateurs non enregistrés, le rejet est correct
            result_text += "\nLe rejet est correct (personne non enregistrée) ✓"
    
    # Mettre à jour le temps d'authentification
    auth_time = format_duration(execution_time)
    
    # Afficher d'autres photos de la personne identifiée
    person_photos = create_person_photos_display(dataset, closest_person_id)
    
    return result_text, auth_time, person_photos

def create_person_photos_display(dataset, person_id):
    """
    Crée un affichage avec plusieurs photos de la personne identifiée.
    
    Args:
        dataset: Dataset contenant les images
        person_id: ID de la personne à afficher
        
    Returns:
        html.Div: Composant affichant les photos de la personne
    """
    # Obtenir l'image de galerie de la personne
    gallery_img = dataset.get_gallery_person(person_id)
    gallery_img_b64 = array_to_base64(gallery_img)
    
    # Obtenir d'autres images de la personne (probes)
    probe_imgs = dataset.get_probe_person(person_id)
    probe_imgs_b64 = [array_to_base64(img) for img in probe_imgs[:5]]  # Limiter à 5 images max
    
    # Créer l'affichage
    return html.Div([
        html.H4(f"Photos de la personne (ID: {person_id})"),
        
        html.Div([
            html.Div([
                html.H5("Image de référence (galerie)"),
                html.Img(src=gallery_img_b64, style={"width": "150px", "height": "150px", "border": "1px solid #ddd"})
            ], className="mb-3"),
            
            html.Div([
                html.H5("Autres photos de cette personne"),
                html.Div(
                    [html.Img(src=img_b64, style={"width": "120px", "height": "120px", "margin": "5px", "border": "1px solid #ddd"}) 
                     for img_b64 in probe_imgs_b64] if probe_imgs_b64 else "Aucune autre photo disponible",
                    style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"}
                )
            ])
        ])
    ]) 