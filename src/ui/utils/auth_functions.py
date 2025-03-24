"""
Fonctions utilitaires pour l'authentification faciale.

Ce module fournit des fonctions partagées par différentes parties de l'interface utilisateur.
"""

import numpy as np
import time
import os
from typing import Dict, Tuple, List, Optional, Any, Callable
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
import joblib

# Importations retardées
_dependencies_loaded = False
brute_force = None
eigenfaces = None 
deep_learning = None
load_dataset = None
evaluate_authentication_method = None
compare_methods = None

def load_dependencies() -> bool:
    """
    Charge les dépendances nécessaires.
    
    Returns:
        bool: True si le chargement a réussi, False sinon
    """
    global brute_force, eigenfaces, deep_learning, load_dataset, evaluate_authentication_method
    global compare_methods, _dependencies_loaded
    
    if _dependencies_loaded:
        return True
        
    try:
        from src.utils import load_dataset, evaluate_authentication_method, compare_methods
        import src.brute_force as brute_force
        import src.eigenfaces as eigenfaces
        import src.deep_learning as deep_learning
        _dependencies_loaded = True
        return True
    except Exception as e:
        print(f"Erreur lors du chargement des dépendances: {e}")
        return False

def load_dataset_with_timing(dataset_num: int, progress_callback: Optional[Callable] = None) -> Dict:
    """
    Charge un dataset avec des informations de timing.
    
    Args:
        dataset_num (int): Numéro du dataset à charger
        progress_callback: Fonction pour rapporter la progression
        
    Returns:
        Dict: Dictionnaire contenant les données et les statistiques de temps
    """
    if not load_dependencies():
        return {'error': "Impossible de charger les dépendances nécessaires"}
    
    if progress_callback:
        progress_callback(10, f"Chargement du dataset {dataset_num}...")
    
    start_time = time.time()
    
    # Charger le dataset
    dataset = load_dataset(dataset_num)
    loading_time = time.time() - start_time
    
    if progress_callback:
        progress_callback(50, f"Prétraitement des images (chargement terminé en {loading_time:.2f}s)...")
    
    # Prétraiter les images
    preprocess_start = time.time()
    gallery_processed, probes_processed = dataset.preprocess_images()
    preprocessing_time = time.time() - preprocess_start
    
    if progress_callback:
        progress_callback(90, "Finalisation...")
    
    total_time = time.time() - start_time
    
    if progress_callback:
        progress_callback(100, f"Chargement terminé en {total_time:.2f}s")
    
    return {
        'success': True,
        'dataset': dataset,
        'gallery_processed': gallery_processed,
        'probes_processed': probes_processed,
        'message': f"Dataset {dataset_num} chargé avec succès!",
        'gallery_shape': gallery_processed.shape,
        'probes_shape': probes_processed.shape,
        'statistics': {
            'loading_time': loading_time,
            'preprocessing_time': preprocessing_time,
            'total_time': total_time
        }
    }

def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convertit une figure matplotlib en image base64.
    
    Args:
        fig (plt.Figure): Figure matplotlib à convertir
        
    Returns:
        str: Chaîne base64 représentant l'image
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def ensure_directory(path: str) -> Path:
    """
    S'assure qu'un répertoire existe, le crée si nécessaire.
    
    Args:
        path (str): Chemin du répertoire
        
    Returns:
        Path: Objet Path représentant le répertoire
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def save_results(results: Dict, method: str, file_prefix: str = "") -> str:
    """
    Sauvegarde les résultats d'une évaluation dans un fichier.
    
    Args:
        results (Dict): Résultats à sauvegarder
        method (str): Méthode d'authentification utilisée
        file_prefix (str): Préfixe pour le nom du fichier
        
    Returns:
        str: Chemin du fichier sauvegardé
    """
    results_dir = ensure_directory(f"results/{method}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{file_prefix}_{timestamp}.joblib" if file_prefix else f"{timestamp}.joblib"
    filepath = results_dir / filename
    
    # Sauvegarder les résultats
    joblib.dump(results, filepath)
    
    return str(filepath)

def load_saved_results(filepath: str) -> Dict:
    """
    Charge des résultats sauvegardés à partir d'un fichier.
    
    Args:
        filepath (str): Chemin du fichier à charger
        
    Returns:
        Dict: Résultats chargés
    """
    if not os.path.exists(filepath):
        return {'error': f"Le fichier {filepath} n'existe pas"}
    
    try:
        results = joblib.load(filepath)
        return {'success': True, 'results': results}
    except Exception as e:
        return {'error': f"Erreur lors du chargement des résultats: {str(e)}"}

def format_duration(seconds: float) -> str:
    """
    Formate une durée en secondes en une chaîne lisible.
    
    Args:
        seconds (float): Durée en secondes
        
    Returns:
        str: Durée formatée
    """
    if seconds < 0.001:  # Moins d'une milliseconde
        return f"{seconds*1e6:.2f} µs"
    elif seconds < 1.0:  # Moins d'une seconde
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:   # Moins d'une minute
        return f"{seconds:.2f} s"
    else:                # Minutes et secondes
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} min {secs:.2f} s"

def generate_performance_summary(results: Dict, method: str) -> str:
    """
    Génère un résumé des performances d'une méthode d'authentification.
    
    Args:
        results (Dict): Résultats de l'évaluation
        method (str): Méthode d'authentification utilisée
        
    Returns:
        str: Résumé formaté des performances
    """
    if 'performance' not in results:
        return "Aucune information de performance disponible"
    
    perf = results['performance']
    
    # Préparer les statistiques de performance
    stats = {
        'Précision': perf.get('precision', 0) * 100,
        'Rappel': perf.get('recall', 0) * 100,
        'Exactitude': perf.get('accuracy', 0) * 100,
        'Exactitude équilibrée': perf.get('balanced_accuracy', 0) * 100,
        'Spécificité': perf.get('specificity', 0) * 100,
        'F1-Score': perf.get('f1_score', 0) * 100
    }
    
    # Préparer les informations spécifiques à la méthode
    method_info = ""
    if method == "brute_force":
        method_info = f"Rayon optimal: {results.get('radius', 0):.4f}"
    elif method == "eigenfaces":
        model_info = results.get('model_info', {})
        n_components = model_info.get('n_components', 0)
        var_explained = model_info.get('variance_explained', 0) * 100
        method_info = f"Rayon optimal: {results.get('radius', 0):.4f}\n"
        method_info += f"Composantes principales: {n_components} ({var_explained:.2f}% de variance expliquée)"
    
    # Préparer les temps de calcul
    time_info = ""
    if 'computation_time' in results:
        time_info = f"Temps de calcul: {format_duration(results['computation_time'])}\n"
    
    # Assembler le résumé
    summary = f"Résultats pour la méthode {method}:\n\n"
    if method_info:
        summary += f"{method_info}\n\n"
    if time_info:
        summary += f"{time_info}\n"
    
    summary += "Statistiques de performance:\n"
    for name, value in stats.items():
        summary += f"- {name}: {value:.2f}%\n"
    
    # Ajouter la matrice de confusion sous forme textuelle si disponible
    if 'conf_matrix' in perf:
        cm = perf['conf_matrix']
        summary += "\nMatrice de confusion:\n"
        summary += f"  Vrais Négatifs: {cm[0, 0]}, Faux Positifs: {cm[0, 1]}\n"
        summary += f"  Faux Négatifs: {cm[1, 0]}, Vrais Positifs: {cm[1, 1]}\n"
    
    # Ajouter le chemin du fichier de sauvegarde si disponible
    if 'saved_file' in results:
        summary += f"\nRésultats sauvegardés dans: {results['saved_file']}"
    
    return summary

def array_to_base64(arr: np.ndarray) -> str:
    """
    Convertit un tableau numpy en chaîne base64.
    
    Args:
        arr (np.ndarray): Tableau à convertir
        
    Returns:
        str: Chaîne base64 représentant le tableau
    """
    if arr is None:
        return ""
    
    if arr.ndim == 1:
        # Essayer de déterminer une forme carrée
        size = int(np.sqrt(arr.shape[0]))
        if size * size == arr.shape[0]:
            arr = arr.reshape(size, size)
        else:
            return ""
    
    # Normaliser entre 0 et 255
    if arr.dtype != np.uint8:
        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    
    from PIL import Image
    img = Image.fromarray(arr)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}" 