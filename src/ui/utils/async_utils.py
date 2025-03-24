"""
Utilitaires pour la gestion des opérations asynchrones dans l'interface utilisateur.

Ce module fournit des fonctions pour exécuter des opérations en arrière-plan
et mettre à jour l'interface utilisateur de manière asynchrone.
"""

import threading
import queue
import time
from typing import Dict, Any, Callable, Optional
import traceback

# Files d'attente pour la communication entre threads
progress_queue = queue.Queue()
result_queue = queue.Queue()

# Variable globale pour stocker l'état
global_state = {
    'dataset': None,
    'gallery_processed': None,
    'probes_processed': None,
    'eigenfaces_model': None,
    'cnn_model': None,
    'current_probe': None,
    'current_probe_identity': None,
    'current_ground_truth': None,
    'is_processing': False,
    'last_evaluation_results': {},
    'last_evaluated_method': None,
    'saved_figures': {},
    'timing_info': {}
}

def clear_queues():
    """Vide les files d'attente pour éviter des résultats résiduels."""
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

def run_async(func: Callable) -> Callable:
    """
    Décorateur pour exécuter une fonction de manière asynchrone dans un thread séparé.
    
    Args:
        func (Callable): Fonction à exécuter de manière asynchrone
        
    Returns:
        Callable: Fonction enveloppée qui exécute func de manière asynchrone
    """
    def wrapper(*args, **kwargs):
        global global_state
        global_state['is_processing'] = True
        
        # Vider les files d'attente
        clear_queues()
        
        def update_progress(percentage, message):
            """Fonction de callback pour la mise à jour de la progression"""
            try:
                progress_queue.put((percentage, message), block=False)
            except queue.Full:
                pass
        
        def task_done():
            """Fonction appelée lorsque la tâche est terminée"""
            global global_state
            global_state['is_processing'] = False
        
        def run_func():
            """Fonction qui exécute la tâche en arrière-plan"""
            try:
                start_time = time.time()
                result = func(*args, **kwargs, progress_callback=update_progress)
                execution_time = time.time() - start_time
                
                # Ajouter le temps d'exécution au résultat
                if isinstance(result, dict):
                    result['execution_time'] = execution_time
                    
                    # Enregistrer le temps d'exécution pour la tâche et la méthode
                    task_name = func.__name__
                    method = kwargs.get('method', args[0] if args else None)
                    
                    if method:
                        global_state['timing_info'][f"{task_name}_{method}"] = execution_time
                    else:
                        global_state['timing_info'][task_name] = execution_time
                
                try:
                    result_queue.put(("success", result), block=False)
                except queue.Full:
                    pass
                    
            except Exception as e:
                print(f"Erreur dans la fonction asynchrone: {e}")
                traceback.print_exc()
                try:
                    result_queue.put(("error", str(e)), block=False)
                except queue.Full:
                    pass
            finally:
                task_done()
        
        # Démarrer la tâche dans un thread séparé
        threading.Thread(target=run_func, daemon=True).start()
    
    return wrapper

def get_progress_update():
    """
    Récupère la dernière mise à jour de progression.
    
    Returns:
        tuple: (progress_value, progress_message) ou (None, None) si pas de mise à jour
    """
    try:
        if not progress_queue.empty():
            return progress_queue.get_nowait()
    except queue.Empty:
        pass
    
    return None, None

def get_result_update():
    """
    Récupère le dernier résultat.
    
    Returns:
        tuple: (status, result) ou (None, None) si pas de résultat
    """
    try:
        if not result_queue.empty():
            return result_queue.get_nowait()
    except queue.Empty:
        pass
    
    return None, None

def is_processing():
    """
    Vérifie si un traitement est en cours.
    
    Returns:
        bool: True si un traitement est en cours, False sinon
    """
    return global_state['is_processing']

def update_global_state(key: str, value: Any):
    """
    Met à jour une valeur dans l'état global.
    
    Args:
        key (str): Clé de la valeur à mettre à jour
        value (Any): Nouvelle valeur
    """
    global_state[key] = value

def get_global_state(key: str, default: Any = None) -> Any:
    """
    Récupère une valeur de l'état global.
    
    Args:
        key (str): Clé de la valeur à récupérer
        default (Any): Valeur par défaut si la clé n'existe pas
        
    Returns:
        Any: Valeur correspondant à la clé, ou default si la clé n'existe pas
    """
    return global_state.get(key, default)

def record_timing(task: str, method: Optional[str] = None, duration: float = 0):
    """
    Enregistre le temps d'exécution d'une tâche.
    
    Args:
        task (str): Nom de la tâche
        method (Optional[str]): Méthode utilisée
        duration (float): Durée d'exécution en secondes
    """
    key = f"{task}_{method}" if method else task
    global_state['timing_info'][key] = duration

def get_timing(task: str, method: Optional[str] = None) -> float:
    """
    Récupère le temps d'exécution d'une tâche.
    
    Args:
        task (str): Nom de la tâche
        method (Optional[str]): Méthode utilisée
        
    Returns:
        float: Durée d'exécution en secondes, ou 0 si non disponible
    """
    key = f"{task}_{method}" if method else task
    return global_state['timing_info'].get(key, 0)

def register_figure(key: str, figure_data: str):
    """
    Enregistre une figure dans l'état global.
    
    Args:
        key (str): Clé de la figure
        figure_data (str): Données de la figure (base64)
    """
    global_state['saved_figures'][key] = figure_data

def get_figure(key: str) -> str:
    """
    Récupère une figure enregistrée.
    
    Args:
        key (str): Clé de la figure
        
    Returns:
        str: Données de la figure, ou chaîne vide si non disponible
    """
    return global_state['saved_figures'].get(key, "")

def has_figure(key: str) -> bool:
    """
    Vérifie si une figure est enregistrée.
    
    Args:
        key (str): Clé de la figure
        
    Returns:
        bool: True si la figure est enregistrée, False sinon
    """
    return key in global_state['saved_figures'] 