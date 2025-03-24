"""
Package d'authentification par deep learning.
"""

from .authentication import CNNAuthenticator, prepare_siamese_pairs, authenticate, create_siamese_model

__all__ = ['CNNAuthenticator', 'prepare_siamese_pairs', 'authenticate', 'create_siamese_model'] 