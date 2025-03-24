"""
Module d'évaluation des performances du système d'authentification faciale.

Ce module fournit des fonctions pour évaluer les performances des différentes 
méthodes d'authentification et les comparer entre elles.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Callable, Any
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc

class PerformanceMetrics:
    """
    Classe pour calculer et stocker les métriques de performance d'un système d'authentification.
    
    Attributes:
        true_positives (int): Nombre de vrais positifs
        false_positives (int): Nombre de faux positifs
        true_negatives (int): Nombre de vrais négatifs
        false_negatives (int): Nombre de faux négatifs
        accuracy (float): Exactitude (accuracy)
        precision (float): Précision
        recall (float): Rappel (sensibilité)
        specificity (float): Spécificité
        execution_time (float): Temps d'exécution en secondes
    """
    
    def __init__(self, y_true: List[bool], y_pred: List[bool], execution_time: float):
        """
        Initialise un objet PerformanceMetrics.
        
        Args:
            y_true (List[bool]): Vérités terrain (True = utilisateur enregistré)
            y_pred (List[bool]): Prédictions (True = accès autorisé)
            execution_time (float): Temps d'exécution en secondes
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        self.true_positives = tp
        self.false_positives = fp
        self.true_negatives = tn
        self.false_negatives = fn
        
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred)
        self.recall = recall_score(y_true, y_pred)  # Sensibilité
        self.specificity = tn / (tn + fp)
        self.execution_time = execution_time
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convertit les métriques en dictionnaire.
        
        Returns:
            Dict[str, float]: Métriques sous forme de dictionnaire
        """
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "execution_time": self.execution_time
        }
    
    def __str__(self) -> str:
        """
        Représentation en chaîne de caractères des métriques.
        
        Returns:
            str: Représentation formatée des métriques
        """
        return (
            f"Performance Metrics:\n"
            f"  Accuracy: {self.accuracy:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall (Sensitivity): {self.recall:.4f}\n"
            f"  Specificity: {self.specificity:.4f}\n"
            f"  Execution Time: {self.execution_time:.4f} seconds\n"
            f"  Confusion Matrix:\n"
            f"    TP: {self.true_positives}, FP: {self.false_positives}\n"
            f"    FN: {self.false_negatives}, TN: {self.true_negatives}"
        )


def evaluate_authentication_method(auth_func: Callable, gallery: np.ndarray, 
                                   probes: np.ndarray, ground_truth: List[bool], 
                                   **kwargs) -> PerformanceMetrics:
    """
    Évalue une méthode d'authentification sur un ensemble de probes.
    
    Args:
        auth_func (Callable): Fonction d'authentification à évaluer
        gallery (np.ndarray): Ensemble d'images de référence
        probes (np.ndarray): Ensemble d'images de test
        ground_truth (List[bool]): Vérité terrain pour les probes
        **kwargs: Arguments supplémentaires à passer à auth_func
        
    Returns:
        PerformanceMetrics: Métriques de performance
    """
    predictions = []
    start_time = time.time()
    
    for probe in probes:
        # Appel de la fonction d'authentification avec la probe et la gallery
        prediction = auth_func(probe, gallery, **kwargs)
        predictions.append(prediction)
    
    execution_time = time.time() - start_time
    
    return PerformanceMetrics(ground_truth, predictions, execution_time)


def plot_roc_curve(y_true: List[bool], y_scores: List[float], method_name: str) -> float:
    """
    Trace la courbe ROC pour une méthode d'authentification.
    
    Args:
        y_true (List[bool]): Vérité terrain
        y_scores (List[float]): Scores de confiance pour chaque prédiction
        method_name (str): Nom de la méthode
        
    Returns:
        float: Aire sous la courbe ROC (AUC)
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {method_name}')
    plt.legend(loc="lower right")
    
    return roc_auc


def compare_methods(methods_results: Dict[str, PerformanceMetrics]) -> None:
    """
    Compare différentes méthodes d'authentification visuellement.
    
    Args:
        methods_results (Dict[str, PerformanceMetrics]): Dictionnaire des résultats par méthode
    """
    method_names = list(methods_results.keys())
    accuracy_values = [m.accuracy for m in methods_results.values()]
    precision_values = [m.precision for m in methods_results.values()]
    recall_values = [m.recall for m in methods_results.values()]
    specificity_values = [m.specificity for m in methods_results.values()]
    times = [m.execution_time for m in methods_results.values()]
    
    # Création du graphique de barres
    x = np.arange(len(method_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width*1.5, accuracy_values, width, label='Accuracy')
    rects2 = ax.bar(x - width/2, precision_values, width, label='Precision')
    rects3 = ax.bar(x + width/2, recall_values, width, label='Recall')
    rects4 = ax.bar(x + width*1.5, specificity_values, width, label='Specificity')
    
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Authentication Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.legend()
    
    plt.tight_layout()
    
    # Graphique du temps d'exécution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(method_names, times)
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Comparison of Execution Times')
    
    plt.tight_layout()
    plt.show() 