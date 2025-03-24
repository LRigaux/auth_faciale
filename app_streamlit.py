"""
Application d'authentification faciale avec Streamlit.

Cette application permet de tester et comparer différentes méthodes d'authentification faciale.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import time
import io
import base64
import os
import random
from typing import Dict, List, Any, Optional, Tuple

# Import des modules métier
from src.core.dataset import FaceDataset, load_dataset
from src.core import brute_force as bf
from src.core import eigenfaces as ef
from src.core import evaluation as eval_module

# Configuration de la page
st.set_page_config(
    page_title="Facial Authentication",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .auth-result-success {
        color: #0f5132;
        background-color: #d1e7dd;
        border-radius: 0.375rem;
        padding: 1rem;
        font-weight: bold;
    }
    .auth-result-failure {
        color: #842029;
        background-color: #f8d7da;
        border-radius: 0.375rem;
        padding: 1rem;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Fonctions utilitaires
def array_to_image(img_array):
    """Convertit un tableau numpy en image PIL."""
    # Normalisation si nécessaire
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    elif img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    # Gestion des images aplaties
    if len(img_array.shape) == 1:
        # Image aplatie - essayer de la reconvertir en 2D
        img_size = int(np.sqrt(img_array.shape[0]))
        img_array = img_array.reshape(img_size, img_size)
    
    # Amélioration du contraste des images sombres
    if img_array.mean() < 50:
        min_val = img_array.min()
        max_val = img_array.max()
        if max_val > min_val:
            img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # Conversion en image PIL
    if len(img_array.shape) == 2:
        return Image.fromarray(img_array, mode='L')
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        return Image.fromarray(img_array, mode='RGB')
    else:
        return Image.fromarray(img_array, mode='L')

def create_confusion_matrix_figure(cm):
    """Crée une figure Plotly pour la matrice de confusion."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Négatif prédit", "Positif prédit"],
        y=["Négatif réel", "Positif réel"],
        colorscale="Blues",
        showscale=False,
        text=[[str(val) for val in row] for row in cm],
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    fig.update_layout(
        margin=dict(l=30, r=30, t=50, b=30),
        height=300
    )
    return fig

def create_roc_curve(data):
    """Crée une figure Plotly pour la courbe ROC."""
    fig = go.Figure()
    
    # Ajouter la courbe ROC
    if data and 'roc_curve' in data:
        roc_data = data['roc_curve']
        fig.add_trace(go.Scatter(
            x=roc_data['fpr'],
            y=roc_data['tpr'],
            mode='lines',
            name=f"AUC = {roc_data.get('auc', 0):.3f}",
            line=dict(color='blue', width=2)
        ))
    
    # Ajouter la ligne de référence (aléatoire)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='grey', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Courbe ROC",
        xaxis_title="Taux de faux positifs",
        yaxis_title="Taux de vrais positifs",
        legend=dict(x=0.01, y=0.99),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Mettre à jour les axes
    fig.update_xaxes(range=[0, 1], constrain="domain")
    fig.update_yaxes(range=[0, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    
    return fig

def create_metrics_table(data):
    """Crée un tableau de métriques formaté."""
    if not data or not isinstance(data, dict):
        return pd.DataFrame()
    
    # Convertir en DataFrame
    df = pd.DataFrame(list(data.items()), columns=["Métrique", "Valeur"])
    
    # Formater les noms de métriques
    df["Métrique"] = df["Métrique"].apply(lambda x: x.replace("_", " ").title())
    
    # Formater les valeurs numériques
    df["Valeur"] = df["Valeur"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    
    return df

def create_comparison_df(bf_results, ef_results):
    """Crée un DataFrame pour comparer les performances des deux méthodes."""
    if not bf_results or not ef_results:
        return pd.DataFrame()
    
    bf_metrics = bf_results.get('performance', {})
    ef_metrics = ef_results.get('performance', {})
    
    # Trouver les métriques communes
    common_metrics = set(bf_metrics.keys()).intersection(set(ef_metrics.keys()))
    
    # Créer les données de comparaison
    data = []
    for metric in common_metrics:
        bf_value = bf_metrics.get(metric, 0)
        ef_value = ef_metrics.get(metric, 0)
        diff = ef_value - bf_value
        
        # Déterminer quelle méthode est meilleure
        if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'true_positive_rate']:
            # Pour ces métriques, plus c'est élevé, mieux c'est
            better = "Eigenfaces" if ef_value > bf_value else "Brute Force" if bf_value > ef_value else "Égal"
        elif metric in ['false_positive_rate', 'false_negative_rate', 'equal_error_rate']:
            # Pour ces métriques, plus c'est bas, mieux c'est
            better = "Eigenfaces" if ef_value < bf_value else "Brute Force" if bf_value < ef_value else "Égal"
        else:
            better = "N/A"
        
        data.append({
            "Métrique": metric.replace("_", " ").title(),
            "Brute Force": f"{bf_value:.4f}",
            "Eigenfaces": f"{ef_value:.4f}",
            "Différence": f"{diff:.4f}",
            "Meilleure méthode": better
        })
    
    return pd.DataFrame(data)

def main():
    """Fonction principale de l'application."""
    # Initialiser l'état de session si nécessaire
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
        st.session_state.precomputed = {
            'gallery_processed': None,
            'probes_processed': None
        }
        st.session_state.bf_results = None
        st.session_state.ef_results = None
        st.session_state.test_probe = None
        st.session_state.test_probe_id = None
        st.session_state.test_probe_enrolled = None
    
    # Navigation latérale
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Aller à",
        ["Accueil", "Brute Force", "Eigenfaces", "Comparaison"]
    )
    
    # Afficher la page sélectionnée
    if page == "Accueil":
        home_page()
    elif page == "Brute Force":
        brute_force_page()
    elif page == "Eigenfaces":
        eigenfaces_page()
    else:
        comparison_page()

def home_page():
    """Page d'accueil avec chargement du dataset."""
    st.title("Authentification Faciale")
    
    st.markdown("""
    ### Démonstration de méthodes d'authentification faciale
    
    Cette application permet de tester et comparer deux méthodes d'authentification faciale :
    
    1. **Brute Force** : Comparaison directe des images par calcul de distances
    2. **Eigenfaces** : Méthode basée sur l'analyse en composantes principales (PCA)
    
    Commencez par charger un dataset puis explorez les différentes méthodes.
    """)
    
    # Sélection et chargement du dataset
    st.header("Chargement des données")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        dataset_num = st.selectbox(
            "Sélectionner un dataset", 
            [1, 2], 
            index=0,
            help="Dataset 1: 373 personnes, Dataset 2: Dataset synthétique plus grand"
        )
    
    with col2:
        load_btn = st.button("Charger le dataset", use_container_width=True)
    
    # Charger le dataset si le bouton est cliqué
    if load_btn:
        progress = st.progress(0)
        status_text = st.empty()
        
        def update_progress(value, message):
            progress.progress(value / 100)
            status_text.info(message)
        
        try:
            status_text.info("Chargement du dataset...")
            dataset = load_dataset(dataset_num, progress_callback=update_progress)
            
            # Prétraitement des données
            update_progress(70, "Prétraitement des données...")
            gallery_processed, probes_processed = dataset.preprocess_images(
                method='normalize', 
                flatten=True,
                progress_callback=lambda v, m: update_progress(70 + v * 0.2, m)
            )
            
            # Stockage des données dans la session
            st.session_state.dataset = dataset
            st.session_state.precomputed = {
                'gallery_processed': gallery_processed,
                'probes_processed': probes_processed
            }
            st.session_state.test_probe = None
            st.session_state.bf_results = None
            st.session_state.ef_results = None
            
            update_progress(100, "Dataset chargé avec succès!")
            st.success(f"Dataset {dataset_num} chargé avec succès!")
            
            # Force le rafraîchissement de la page pour afficher les infos du dataset
            st.rerun()
            
        except Exception as e:
            st.error(f"Erreur lors du chargement du dataset: {str(e)}")
            progress.progress(0)
    
    # Affichage des informations du dataset s'il est chargé
    if st.session_state.dataset is not None:
        dataset = st.session_state.dataset
        
        st.header("Informations sur le dataset")
        
        # Affichage des métriques dans des cartes
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gallery", f"{dataset.n_gallery} images")
        with col2:
            st.metric("Probes", f"{dataset.n_probes} images")
        with col3:
            st.metric("Dimensions", f"{dataset.image_shape[0]}x{dataset.image_shape[1]}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probes enrolled", f"{dataset.n_enrolled_probes} images")
        with col2:
            st.metric("Probes non-enrolled", f"{dataset.n_non_enrolled_probes} images")
        
        # Affichage d'exemples d'images
        st.subheader("Exemples d'images")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Image de gallery**")
            # Sélectionner aléatoirement une image de gallery
            idx = random.randint(0, dataset.n_gallery - 1)
            img = dataset.gallery[idx]
            st.image(array_to_image(img), width=200)
            st.caption(f"ID: {dataset.gallery_ids[idx]}")
        
        with col2:
            st.markdown("**Image probe**")
            # Sélectionner aléatoirement une image probe
            idx = random.randint(0, dataset.n_probes - 1)
            img = dataset.probes[idx]
            st.image(array_to_image(img), width=200)
            is_enrolled = dataset.ground_truth[idx]
            st.caption(f"ID: {dataset.probe_ids[idx]} {'(Enrolled)' if is_enrolled else '(Non-enrolled)'}")
        
        # Instructions pour continuer
        st.info("Naviguez vers les onglets 'Brute Force' ou 'Eigenfaces' pour tester les méthodes d'authentification.")
        
    else:
        # Affichage si aucun dataset n'est chargé
        st.info("Veuillez charger un dataset pour commencer.")

def brute_force_page():
    """Page pour tester la méthode brute force."""
    st.title("Méthode Brute Force")
    
    # Vérifier si un dataset est chargé
    if st.session_state.dataset is None:
        st.warning("Veuillez d'abord charger un dataset dans l'onglet 'Accueil'.")
        return
    
    dataset = st.session_state.dataset
    
    # Récupérer les données prétraitées
    gallery = st.session_state.precomputed['gallery_processed']
    probes = st.session_state.precomputed['probes_processed']
    
    # Section paramètres
    st.header("Paramètres")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Métrique de distance
        metric = st.selectbox(
            "Métrique de distance",
            ["L1", "L2", "cosine"],
            index=1,
            help="L1: Manhattan, L2: Euclidienne, cosine: Similarité cosinus"
        )
    
    with col2:
        # Seuil d'authentification
        threshold = st.slider(
            "Seuil d'authentification",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Seuil de distance en dessous duquel une probe est considérée comme authentifiée"
        )
    
    # Boutons d'action
    col1, col2 = st.columns(2)
    
    with col1:
        evaluate_btn = st.button("Évaluer la performance", use_container_width=True)
    
    with col2:
        find_threshold_btn = st.button("Trouver le meilleur seuil", use_container_width=True)
    
    # Préparation des données pour l'évaluation
    # Diviser les probes en enrolled et non-enrolled
    enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if gt]
    non_enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if not gt]
    
    enrolled_probes = probes[enrolled_indices]
    non_enrolled_probes = probes[non_enrolled_indices]
    
    # Section d'évaluation
    if evaluate_btn or find_threshold_btn:
        st.header("Évaluation")
        progress = st.progress(0)
        status_text = st.empty()
        
        def update_progress(value, message):
            progress.progress(value / 100)
            status_text.info(message)
        
        try:
            if evaluate_btn:
                update_progress(5, "Évaluation en cours...")
                
                # Évaluer la performance avec le seuil spécifié
                results = bf.evaluate_performance(
                    gallery=gallery,
                    enrolled_probes=enrolled_probes,
                    non_enrolled_probes=non_enrolled_probes,
                    threshold=threshold,
                    metric=metric,
                    progress_callback=update_progress
                )
                
                # Stocker les résultats dans la session
                st.session_state.bf_results = results
                
                update_progress(100, "Évaluation terminée!")
                
            elif find_threshold_btn:
                update_progress(5, "Recherche du meilleur seuil...")
                
                # Trouver le meilleur seuil
                best_threshold, results = bf.find_best_threshold(
                    gallery=gallery,
                    enrolled_probes=enrolled_probes,
                    non_enrolled_probes=non_enrolled_probes,
                    metric=metric,
                    progress_callback=update_progress
                )
                
                # Stocker les résultats et mettre à jour le seuil
                st.session_state.bf_results = results
                threshold = best_threshold
                
                update_progress(100, f"Meilleur seuil trouvé: {best_threshold:.4f}")
                st.success(f"Meilleur seuil trouvé: {best_threshold:.4f}")
        
        except Exception as e:
            st.error(f"Erreur lors de l'évaluation: {str(e)}")
            progress.progress(0)
    
    # Affichage des résultats si disponibles
    if st.session_state.bf_results is not None:
        results = st.session_state.bf_results
        
        st.header("Résultats")
        
        # Affichage du temps d'exécution et du seuil
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Temps d'exécution", f"{results['execution_time']:.2f} s")
        with col2:
            st.metric("Seuil utilisé", f"{results['threshold']:.4f}")
        
        # Affichage des métriques de performance
        st.subheader("Métriques de performance")
        
        metrics_df = create_metrics_table(results['performance'])
        st.dataframe(metrics_df, use_container_width=True)
        
        # Affichage de la matrice de confusion
        st.subheader("Matrice de confusion")
        
        conf_matrix_fig = create_confusion_matrix_figure(results['confusion_matrix'])
        st.plotly_chart(conf_matrix_fig, use_container_width=True)
        
        # Affichage de la courbe ROC
        if 'roc_curve' in results:
            st.subheader("Courbe ROC")
            
            roc_fig = create_roc_curve(results)
            st.plotly_chart(roc_fig, use_container_width=True)
    
    # Section pour tester l'authentification sur une image
    st.markdown("---")
    st.header("Tester l'authentification")
    
    # Sélection d'une image aléatoire ou utilisation de celle déjà sélectionnée
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.test_probe is not None:
            st.image(array_to_image(st.session_state.test_probe), width=200)
            probe_info = f"ID: {st.session_state.test_probe_id} "
            probe_info += f"({'Enrolled' if st.session_state.test_probe_enrolled else 'Non-enrolled'})"
            st.caption(probe_info)
    
    with col2:
        if st.button("Sélectionner une image aléatoire", use_container_width=True):
            # Sélectionner aléatoirement une image probe
            idx = random.randint(0, dataset.n_probes - 1)
            st.session_state.test_probe = dataset.probes[idx]
            st.session_state.test_probe_id = dataset.probe_ids[idx]
            st.session_state.test_probe_enrolled = dataset.ground_truth[idx]
            st.rerun()
    
    # Authentifier l'image sélectionnée
    if st.session_state.test_probe is not None:
        if st.button("Authentifier", use_container_width=True):
            # Prétraiter l'image
            probe_flat = st.session_state.test_probe.flatten()
            if gallery[0].shape[0] != probe_flat.shape[0]:
                # Redimensionner si nécessaire
                probe_img = array_to_image(st.session_state.test_probe)
                probe_img = probe_img.resize((int(np.sqrt(gallery[0].shape[0])), int(np.sqrt(gallery[0].shape[0]))))
                probe_flat = np.array(probe_img).flatten() / 255.0
            
            # Authentifier
            is_auth, closest_idx, min_distance = bf.authenticate(
                probe=probe_flat,
                gallery=gallery,
                threshold=threshold,
                metric=metric
            )
            
            # Afficher le résultat
            st.subheader("Résultat de l'authentification")
            
            # Conteneur pour le résultat
            if is_auth:
                st.markdown(f"""
                <div class="auth-result-success">
                ✅ Authentifié avec une distance de {min_distance:.4f}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="auth-result-failure">
                ❌ Non authentifié - Distance: {min_distance:.4f} (seuil: {threshold:.4f})
                </div>
                """, unsafe_allow_html=True)
            
            # Afficher l'image la plus proche dans la galerie
            st.subheader("Correspondance la plus proche")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Image de test**")
                st.image(array_to_image(st.session_state.test_probe), width=200)
                probe_info = f"ID: {st.session_state.test_probe_id} "
                probe_info += f"({'Enrolled' if st.session_state.test_probe_enrolled else 'Non-enrolled'})"
                st.caption(probe_info)
            
            with col2:
                st.markdown("**Meilleure correspondance**")
                best_match = gallery[closest_idx]
                # Reconvertir en 2D pour l'affichage
                size = int(np.sqrt(best_match.shape[0]))
                best_match_2d = best_match.reshape(size, size)
                st.image(array_to_image(best_match_2d), width=200)
                st.caption(f"ID: {dataset.gallery_ids[closest_idx]} (Distance: {min_distance:.4f})")
            
            # Vérité terrain
            is_correct = (is_auth and st.session_state.test_probe_enrolled) or \
                        (not is_auth and not st.session_state.test_probe_enrolled)
            
            if is_correct:
                st.success("✓ Le résultat correspond à la vérité terrain.")
            else:
                st.error("✗ Le résultat ne correspond pas à la vérité terrain.")

def eigenfaces_page():
    """Page pour tester la méthode Eigenfaces."""
    st.title("Méthode Eigenfaces")
    
    # Vérifier si un dataset est chargé
    if st.session_state.dataset is None:
        st.warning("Veuillez d'abord charger un dataset dans l'onglet 'Accueil'.")
        return
    
    dataset = st.session_state.dataset
    
    # Récupérer les données prétraitées
    gallery = st.session_state.precomputed['gallery_processed']
    probes = st.session_state.precomputed['probes_processed']
    
    # Section paramètres
    st.header("Paramètres")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Nombre de composantes
        n_components = st.number_input(
            "Nombre de composantes principales",
            min_value=1,
            max_value=min(100, dataset.n_gallery),
            value=min(20, dataset.n_gallery),
            step=1,
            help="Nombre d'eigenfaces à utiliser"
        )
    
    with col2:
        # Seuil d'authentification
        threshold = st.slider(
            "Seuil d'authentification",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
            help="Seuil de distance en dessous duquel une probe est considérée comme authentifiée"
        )
    
    # Boutons d'action
    col1, col2 = st.columns(2)
    
    with col1:
        train_btn = st.button("Entraîner et évaluer", use_container_width=True)
    
    with col2:
        find_threshold_btn = st.button("Trouver le meilleur seuil", use_container_width=True)
    
    # Préparation des données pour l'évaluation
    # Diviser les probes en enrolled et non-enrolled
    enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if gt]
    non_enrolled_indices = [i for i, gt in enumerate(dataset.ground_truth) if not gt]
    
    enrolled_probes = probes[enrolled_indices]
    non_enrolled_probes = probes[non_enrolled_indices]
    
    # Section d'entraînement et d'évaluation
    if train_btn or find_threshold_btn:
        st.header("Entraînement et évaluation")
        progress = st.progress(0)
        status_text = st.empty()
        
        def update_progress(value, message):
            progress.progress(value / 100)
            status_text.info(message)
        
        try:
            if train_btn:
                update_progress(5, "Entraînement du modèle Eigenfaces...")
                
                # Créer et entraîner le modèle
                eigenfaces_model = ef.EigenfacesModel(n_components=n_components)
                eigenfaces_model.fit(gallery)
                
                update_progress(50, "Évaluation en cours...")
                
                # Évaluer la performance
                results = ef.evaluate_performance(
                    model=eigenfaces_model,
                    gallery=gallery,
                    enrolled_probes=enrolled_probes,
                    non_enrolled_probes=non_enrolled_probes,
                    threshold=threshold,
                    progress_callback=lambda v, m: update_progress(50 + v * 0.5, m)
                )
                
                # Ajouter le modèle aux résultats
                results['model'] = eigenfaces_model
                
                # Stocker les résultats dans la session
                st.session_state.ef_results = results
                
                update_progress(100, "Entraînement et évaluation terminés!")
                
            elif find_threshold_btn:
                update_progress(5, "Recherche du meilleur seuil...")
                
                # Trouver le meilleur seuil
                best_threshold, results, eigenfaces_model = ef.find_best_threshold(
                    gallery=gallery,
                    enrolled_probes=enrolled_probes,
                    non_enrolled_probes=non_enrolled_probes,
                    n_components=n_components,
                    progress_callback=update_progress
                )
                
                # Ajouter le modèle aux résultats
                results['model'] = eigenfaces_model
                
                # Stocker les résultats et mettre à jour le seuil
                st.session_state.ef_results = results
                threshold = best_threshold
                
                update_progress(100, f"Meilleur seuil trouvé: {best_threshold:.4f}")
                st.success(f"Meilleur seuil trouvé: {best_threshold:.4f}")
        
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement/évaluation: {str(e)}")
            progress.progress(0)
    
    # Affichage des résultats si disponibles
    if st.session_state.ef_results is not None:
        results = st.session_state.ef_results
        eigenfaces_model = results.get('model')
        
        st.header("Résultats")
        
        # Affichage du temps d'exécution et du seuil
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Temps d'exécution", f"{results['execution_time']:.2f} s")
        with col2:
            st.metric("Seuil utilisé", f"{results['threshold']:.4f}")
        with col3:
            st.metric("Nombre d'eigenfaces", f"{eigenfaces_model.eigenfaces.shape[0]}")
        
        # Affichage des métriques de performance
        st.subheader("Métriques de performance")
        
        metrics_df = create_metrics_table(results['performance'])
        st.dataframe(metrics_df, use_container_width=True)
        
        # Affichage de la matrice de confusion
        st.subheader("Matrice de confusion")
        
        conf_matrix_fig = create_confusion_matrix_figure(results['confusion_matrix'])
        st.plotly_chart(conf_matrix_fig, use_container_width=True)
        
        # Affichage de la courbe ROC
        if 'roc_curve' in results:
            st.subheader("Courbe ROC")
            
            roc_fig = create_roc_curve(results)
            st.plotly_chart(roc_fig, use_container_width=True)
        
        # Visualisation des eigenfaces
        st.subheader("Visualisation des Eigenfaces")
        
        # Visage moyen
        st.markdown("**Visage moyen**")
        mean_face = eigenfaces_model.mean_face
        mean_face_size = int(np.sqrt(mean_face.shape[0]))
        mean_face_2d = mean_face.reshape(mean_face_size, mean_face_size)
        st.image(array_to_image(mean_face_2d), width=150)
        
        # Première eigenfaces
        st.markdown("**Principales eigenfaces**")
        
        # Afficher au maximum 9 eigenfaces en grille de 3x3
        n_faces_to_show = min(9, eigenfaces_model.eigenfaces.shape[0])
        face_size = int(np.sqrt(eigenfaces_model.eigenfaces.shape[1]))
        
        # Créer une grille
        cols = st.columns(3)
        for i in range(n_faces_to_show):
            with cols[i % 3]:
                eigenface = eigenfaces_model.eigenfaces[i].reshape(face_size, face_size)
                st.image(array_to_image(eigenface), width=120)
                st.caption(f"Eigenface {i+1}")
    
    # Section pour tester l'authentification sur une image
    st.markdown("---")
    st.header("Tester l'authentification")
    
    # Vérifier si un modèle a été entraîné
    if st.session_state.ef_results is None or 'model' not in st.session_state.ef_results:
        st.warning("Veuillez d'abord entraîner un modèle en cliquant sur 'Entraîner et évaluer'.")
        return
    
    eigenfaces_model = st.session_state.ef_results['model']
    
    # Sélection d'une image aléatoire ou utilisation de celle déjà sélectionnée
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.test_probe is not None:
            st.image(array_to_image(st.session_state.test_probe), width=200)
            probe_info = f"ID: {st.session_state.test_probe_id} "
            probe_info += f"({'Enrolled' if st.session_state.test_probe_enrolled else 'Non-enrolled'})"
            st.caption(probe_info)
    
    with col2:
        if st.button("Sélectionner une image aléatoire", use_container_width=True):
            # Sélectionner aléatoirement une image probe
            idx = random.randint(0, dataset.n_probes - 1)
            st.session_state.test_probe = dataset.probes[idx]
            st.session_state.test_probe_id = dataset.probe_ids[idx]
            st.session_state.test_probe_enrolled = dataset.ground_truth[idx]
            st.rerun()
    
    # Authentifier l'image sélectionnée
    if st.session_state.test_probe is not None:
        if st.button("Authentifier", use_container_width=True):
            # Prétraiter l'image
            probe_flat = st.session_state.test_probe.flatten()
            if gallery[0].shape[0] != probe_flat.shape[0]:
                # Redimensionner si nécessaire
                probe_img = array_to_image(st.session_state.test_probe)
                probe_img = probe_img.resize((int(np.sqrt(gallery[0].shape[0])), int(np.sqrt(gallery[0].shape[0]))))
                probe_flat = np.array(probe_img).flatten() / 255.0
            
            # Authentifier
            is_auth, closest_idx, min_distance = ef.authenticate(
                probe=probe_flat,
                model=eigenfaces_model,
                threshold=threshold
            )
            
            # Afficher le résultat
            st.subheader("Résultat de l'authentification")
            
            # Conteneur pour le résultat
            if is_auth:
                st.markdown(f"""
                <div class="auth-result-success">
                ✅ Authentifié avec une distance de {min_distance:.4f}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="auth-result-failure">
                ❌ Non authentifié - Distance: {min_distance:.4f} (seuil: {threshold:.4f})
                </div>
                """, unsafe_allow_html=True)
            
            # Récupérer l'image correspondante dans la gallery
            if closest_idx is not None:
                best_match_id = eigenfaces_model.gallery_ids[closest_idx]
                # Trouver l'index dans le dataset
                best_match_orig_idx = dataset.gallery_ids.index(best_match_id)
                
                # Afficher l'image la plus proche dans la galerie
                st.subheader("Correspondance la plus proche")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Image de test**")
                    st.image(array_to_image(st.session_state.test_probe), width=200)
                    probe_info = f"ID: {st.session_state.test_probe_id} "
                    probe_info += f"({'Enrolled' if st.session_state.test_probe_enrolled else 'Non-enrolled'})"
                    st.caption(probe_info)
                
                with col2:
                    st.markdown("**Meilleure correspondance**")
                    best_match = dataset.gallery[best_match_orig_idx]
                    st.image(array_to_image(best_match), width=200)
                    st.caption(f"ID: {best_match_id} (Distance: {min_distance:.4f})")
                
                # Reconstruction de l'image
                st.subheader("Reconstruction par eigenfaces")
                
                # Projeter et reconstruire l'image de test
                probe_projection = eigenfaces_model.project(probe_flat.reshape(1, -1))
                probe_reconstruction = eigenfaces_model.reconstruct(probe_projection)
                
                # Afficher l'originale et la reconstruction
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Image originale**")
                    st.image(array_to_image(st.session_state.test_probe), width=200)
                
                with col2:
                    st.markdown("**Reconstruction**")
                    recon_size = int(np.sqrt(probe_reconstruction.shape[1]))
                    recon_2d = probe_reconstruction[0].reshape(recon_size, recon_size)
                    st.image(array_to_image(recon_2d), width=200)
                
                # Vérité terrain
                is_correct = (is_auth and st.session_state.test_probe_enrolled) or \
                            (not is_auth and not st.session_state.test_probe_enrolled)
                
                if is_correct:
                    st.success("✓ Le résultat correspond à la vérité terrain.")
                else:
                    st.error("✗ Le résultat ne correspond pas à la vérité terrain.")

def comparison_page():
    """Page pour comparer les méthodes Brute Force et Eigenfaces."""
    st.title("Comparaison des méthodes")
    
    # Vérifier si les deux méthodes ont été évaluées
    bf_results = st.session_state.bf_results
    ef_results = st.session_state.ef_results
    
    if bf_results is None or ef_results is None:
        st.warning("""
        Veuillez d'abord évaluer les deux méthodes :
        1. Allez dans l'onglet 'Brute Force' et cliquez sur 'Évaluer la performance'
        2. Allez dans l'onglet 'Eigenfaces' et cliquez sur 'Entraîner et évaluer'
        """)
        return
    
    # Affichage des paramètres utilisés
    st.header("Paramètres utilisés")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Brute Force")
        st.write(f"**Seuil** : {bf_results['threshold']:.4f}")
        st.write(f"**Métrique** : {bf_results['metric']}")
        st.write(f"**Temps d'exécution** : {bf_results['execution_time']:.2f} s")
    
    with col2:
        st.subheader("Eigenfaces")
        st.write(f"**Seuil** : {ef_results['threshold']:.4f}")
        st.write(f"**Composantes** : {ef_results['model'].eigenfaces.shape[0]}")
        st.write(f"**Temps d'exécution** : {ef_results['execution_time']:.2f} s")
    
    # Tableau comparatif des métriques
    st.header("Comparaison des métriques")
    
    # Créer le DataFrame de comparaison
    comparison_df = create_comparison_df(bf_results, ef_results)
    
    # Afficher le tableau
    st.dataframe(comparison_df, use_container_width=True)
    
    # Affichage du gagnant
    better_method_counts = comparison_df["Meilleure méthode"].value_counts()
    if "Eigenfaces" in better_method_counts and "Brute Force" in better_method_counts:
        if better_method_counts["Eigenfaces"] > better_method_counts["Brute Force"]:
            winner = "Eigenfaces"
        elif better_method_counts["Eigenfaces"] < better_method_counts["Brute Force"]:
            winner = "Brute Force"
        else:
            winner = "Égalité"
    elif "Eigenfaces" in better_method_counts:
        winner = "Eigenfaces"
    elif "Brute Force" in better_method_counts:
        winner = "Brute Force"
    else:
        winner = "Indéterminé"
    
    # Style de couleur selon le gagnant
    winner_color = "#198754" if winner == "Eigenfaces" else "#dc3545" if winner == "Brute Force" else "#6c757d"
    
    # Afficher le gagnant avec un style
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0; padding: 15px; background-color: {winner_color}; color: white; border-radius: 5px;">
        <h3 style="margin: 0;">Méthode globalement plus performante: {winner}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualisations comparatives
    st.header("Visualisations comparatives")
    
    # Comparaison des matrices de confusion
    st.subheader("Matrices de confusion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Brute Force**")
        conf_matrix_bf = create_confusion_matrix_figure(bf_results['confusion_matrix'])
        st.plotly_chart(conf_matrix_bf, use_container_width=True)
    
    with col2:
        st.markdown("**Eigenfaces**")
        conf_matrix_ef = create_confusion_matrix_figure(ef_results['confusion_matrix'])
        st.plotly_chart(conf_matrix_ef, use_container_width=True)
    
    # Comparaison des courbes ROC
    if 'roc_curve' in bf_results and 'roc_curve' in ef_results:
        st.subheader("Courbes ROC")
        
        # Créer une figure Plotly pour les deux courbes ROC
        fig = go.Figure()
        
        # Ajouter la courbe ROC de Brute Force
        roc_data_bf = bf_results['roc_curve']
        fig.add_trace(go.Scatter(
            x=roc_data_bf['fpr'],
            y=roc_data_bf['tpr'],
            mode='lines',
            name=f"Brute Force (AUC = {roc_data_bf.get('auc', 0):.3f})",
            line=dict(color='#dc3545', width=2)
        ))
        
        # Ajouter la courbe ROC d'Eigenfaces
        roc_data_ef = ef_results['roc_curve']
        fig.add_trace(go.Scatter(
            x=roc_data_ef['fpr'],
            y=roc_data_ef['tpr'],
            mode='lines',
            name=f"Eigenfaces (AUC = {roc_data_ef.get('auc', 0):.3f})",
            line=dict(color='#198754', width=2)
        ))
        
        # Ajouter la ligne de référence (aléatoire)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Aléatoire',
            line=dict(color='grey', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Comparaison des courbes ROC",
            xaxis_title="Taux de faux positifs",
            yaxis_title="Taux de vrais positifs",
            legend=dict(x=0.01, y=0.99),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Mettre à jour les axes
        fig.update_xaxes(range=[0, 1], constrain="domain")
        fig.update_yaxes(range=[0, 1], constrain="domain", scaleanchor="x", scaleratio=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualisation comparative des métriques
    st.subheader("Comparaison des métriques clés")
    
    # Sélectionner les métriques importantes pour le graphique
    key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'true_positive_rate', 'false_positive_rate']
    available_metrics = set(bf_results['performance'].keys()).intersection(set(ef_results['performance'].keys()))
    key_metrics = [m for m in key_metrics if m in available_metrics]
    
    bf_values = [bf_results['performance'].get(metric, 0) for metric in key_metrics]
    ef_values = [ef_results['performance'].get(metric, 0) for metric in key_metrics]
    
    # Créer le graphique en barres
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=key_metrics,
        y=bf_values,
        name='Brute Force',
        marker_color='#dc3545'
    ))
    
    fig.add_trace(go.Bar(
        x=key_metrics,
        y=ef_values,
        name='Eigenfaces',
        marker_color='#198754'
    ))
    
    fig.update_layout(
        title="Métriques de performance",
        xaxis_title="Métrique",
        yaxis_title="Valeur",
        legend=dict(x=0.01, y=0.99),
        height=400,
        barmode='group',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(key_metrics))),
            ticktext=[m.replace('_', ' ').title() for m in key_metrics]
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparaison des temps d'exécution
    st.subheader("Comparaison des temps d'exécution")
    
    # Créer le graphique en barres pour les temps d'exécution
    fig = go.Figure()
    
    execution_times = [bf_results['execution_time'], ef_results['execution_time']]
    methods = ['Brute Force', 'Eigenfaces']
    colors = ['#dc3545', '#198754']
    
    fig.add_trace(go.Bar(
        x=methods,
        y=execution_times,
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Temps d'exécution",
        xaxis_title="Méthode",
        yaxis_title="Temps (secondes)",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Conclusion
    st.header("Conclusion")
    
    # Générer automatiquement une conclusion
    speed_winner = "Brute Force" if bf_results['execution_time'] < ef_results['execution_time'] else "Eigenfaces"
    speed_ratio = max(execution_times) / min(execution_times)
    
    accuracy_bf = bf_results['performance'].get('accuracy', 0)
    accuracy_ef = ef_results['performance'].get('accuracy', 0)
    accuracy_winner = "Brute Force" if accuracy_bf > accuracy_ef else "Eigenfaces"
    
    st.markdown(f"""
    ### Résumé de la comparaison:
    
    - **Performance globale:** La méthode **{winner}** est généralement plus performante.
    - **Précision:** La méthode **{accuracy_winner}** offre une meilleure précision ({max(accuracy_bf, accuracy_ef):.2%} contre {min(accuracy_bf, accuracy_ef):.2%}).
    - **Vitesse:** La méthode **{speed_winner}** est plus rapide (environ {speed_ratio:.1f}x).
    
    #### Recommandation:
    """)
    
    if winner == "Eigenfaces" and speed_winner == "Eigenfaces":
        st.success("La méthode Eigenfaces est à privilégier dans ce cas, car elle est à la fois plus précise et plus rapide.")
    elif winner == "Brute Force" and speed_winner == "Brute Force":
        st.success("La méthode Brute Force est à privilégier dans ce cas, car elle est à la fois plus précise et plus rapide.")
    elif winner == "Eigenfaces" and speed_winner == "Brute Force":
        st.info("La méthode Eigenfaces offre de meilleures performances, mais la méthode Brute Force est plus rapide. Le choix dépend de vos contraintes: privilégiez Eigenfaces si la précision est cruciale, ou Brute Force si la vitesse est prioritaire.")
    elif winner == "Brute Force" and speed_winner == "Eigenfaces":
        st.info("La méthode Brute Force offre de meilleures performances, mais la méthode Eigenfaces est plus rapide. Le choix dépend de vos contraintes: privilégiez Brute Force si la précision est cruciale, ou Eigenfaces si la vitesse est prioritaire.")
    else:
        st.info("Les deux méthodes ont des avantages et des inconvénients comparables. Le choix dépendra du cas d'utilisation spécifique et des contraintes du système.")

if __name__ == "__main__":
    main() 