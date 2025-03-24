"""
Core functionality for facial authentication system.

This package contains the implementation of the core functionalities:
- Dataset management
- Brute force authentication
- Eigenfaces authentication
- Performance evaluation
"""

from .dataset import FaceDataset, load_dataset, create_synthetic_dataset
from .brute_force import compute_distances, authenticate as bf_authenticate, find_best_threshold as bf_find_best_threshold
from .eigenfaces import EigenfacesModel, authenticate as ef_authenticate, find_best_threshold as ef_find_best_threshold
from .evaluation import evaluate_performance, compare_methods

__all__ = [
    'FaceDataset', 
    'load_dataset',
    'create_synthetic_dataset',
    'compute_distances',
    'bf_authenticate',
    'bf_find_best_threshold',
    'EigenfacesModel',
    'ef_authenticate',
    'ef_find_best_threshold',
    'evaluate_performance',
    'compare_methods'
] 