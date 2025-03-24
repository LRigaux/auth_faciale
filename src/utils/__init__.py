"""
Package d'utilitaires pour le système d'authentification faciale.
"""

from .data_loader import FaceDataset, load_dataset
from .evaluation import PerformanceMetrics, evaluate_authentication_method, plot_roc_curve, compare_methods

__all__ = [
    'FaceDataset', 
    'load_dataset',
    'PerformanceMetrics',
    'evaluate_authentication_method',
    'plot_roc_curve',
    'compare_methods'
] 