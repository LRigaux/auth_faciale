"""
Module d'authentification par force brute.

Ce module implémente la méthode d'authentification par force brute,
qui compare directement l'image requête avec toutes les images de la gallery.
"""

import numpy as np
from typing import List, Tuple, Union


def compute_distances(data: np.ndarray, query: np.ndarray, norm: str = "L2") -> np.ndarray:
    """
    Calcule les distances entre un vecteur requête et tous les vecteurs d'un dataset.
    
    Args:
        data (np.ndarray): Dataset de vecteurs (chaque ligne est un vecteur)
        query (np.ndarray): Vecteur requête
        norm (str): Norme à utiliser ("L1", "L2" ou "inf")
        
    Returns:
        np.ndarray: Tableau des distances
        
    Raises:
        ValueError: Si la norme n'est pas reconnue
    """
    if norm == "L1":
        # Distance de Manhattan
        return np.sum(np.abs(data - query), axis=1)
    elif norm == "L2":
        # Distance euclidienne au carré
        return np.sum((data - query)**2, axis=1)
    elif norm == "inf":
        # Distance de Chebyshev
        return np.max(np.abs(data - query), axis=1)
    else:
        raise ValueError("Norme non reconnue. Utilisez 'L1', 'L2' ou 'inf'.")


def radius_search(data: np.ndarray, query: np.ndarray, radius: float, norm: str = "L2") -> Tuple[np.ndarray, np.ndarray]:
    """
    Recherche les voisins d'un vecteur requête dans un rayon donné.
    
    Args:
        data (np.ndarray): Dataset de vecteurs (chaque ligne est un vecteur)
        query (np.ndarray): Vecteur requête
        radius (float): Rayon de recherche
        norm (str): Norme à utiliser ("L1", "L2" ou "inf")
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Indices des voisins dans le rayon et leurs distances au vecteur requête
    """
    distances = compute_distances(data, query, norm)
    indices = np.where(distances <= radius)[0]
    return indices, distances[indices]


def authenticate(probe: np.ndarray, gallery: np.ndarray, radius: float, norm: str = "L2") -> bool:
    """
    Authentifie une image requête en cherchant des visages similaires dans la gallery.
    
    Args:
        probe (np.ndarray): Image requête à authentifier
        gallery (np.ndarray): Gallery d'images de référence
        radius (float): Rayon de recherche
        norm (str): Norme à utiliser ("L1", "L2" ou "inf")
        
    Returns:
        bool: True si l'authentification est réussie, False sinon
    """
    probe_vector = probe.flatten()
    gallery_vectors = gallery.reshape(gallery.shape[0], -1)
    
    indices, distances = radius_search(gallery_vectors, probe_vector, radius, norm)
    
    # L'authentification réussit s'il y a au moins un voisin dans le rayon
    return len(indices) > 0


def find_best_radius(gallery: np.ndarray, enrolled_probes: np.ndarray, 
                   non_enrolled_probes: np.ndarray, norm: str = "L2") -> float:
    """
    Trouve le meilleur rayon pour l'authentification.
    
    Args:
        gallery (np.ndarray): Gallery d'images de référence
        enrolled_probes (np.ndarray): Images de test d'utilisateurs enregistrés
        non_enrolled_probes (np.ndarray): Images de test d'utilisateurs non enregistrés
        norm (str): Norme à utiliser ("L1", "L2" ou "inf")
        
    Returns:
        float: Rayon optimal
    """
    # Aplatir les images
    gallery_vectors = gallery.reshape(gallery.shape[0], -1)
    enrolled_vectors = enrolled_probes.reshape(enrolled_probes.shape[0], -1)
    non_enrolled_vectors = non_enrolled_probes.reshape(non_enrolled_probes.shape[0], -1)
    
    # Calculer les distances minimales pour les utilisateurs enregistrés
    min_distances_enrolled = []
    for probe in enrolled_vectors:
        distances = compute_distances(gallery_vectors, probe, norm)
        min_distances_enrolled.append(np.min(distances))
    
    # Calculer les distances minimales pour les utilisateurs non enregistrés
    min_distances_non_enrolled = []
    for probe in non_enrolled_vectors:
        distances = compute_distances(gallery_vectors, probe, norm)
        min_distances_non_enrolled.append(np.min(distances))
    
    # Convertir en numpy arrays
    min_distances_enrolled = np.array(min_distances_enrolled)
    min_distances_non_enrolled = np.array(min_distances_non_enrolled)
    
    # Trouver le rayon optimal
    best_radius = 0
    best_accuracy = 0
    
    # Tester différentes valeurs de rayon
    for percentile in range(10, 100, 5):
        radius_candidate = np.percentile(min_distances_enrolled, percentile)
        
        # Calculer les performances avec ce rayon
        true_positives = np.sum(min_distances_enrolled <= radius_candidate)
        false_positives = np.sum(min_distances_non_enrolled <= radius_candidate)
        true_negatives = len(min_distances_non_enrolled) - false_positives
        false_negatives = len(min_distances_enrolled) - true_positives
        
        accuracy = (true_positives + true_negatives) / (len(min_distances_enrolled) + len(min_distances_non_enrolled))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_radius = radius_candidate
    
    return best_radius 