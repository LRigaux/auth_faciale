"""
Package d'authentification par Eigenfaces.
"""

from .authentication import EigenfacesModel, authenticate, find_best_radius

__all__ = ['EigenfacesModel', 'authenticate', 'find_best_radius'] 