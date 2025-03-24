"""
Module d'authentification par Eigenfaces.

Ce module implémente la méthode d'authentification par Eigenfaces,
qui utilise l'Analyse en Composantes Principales (ACP) pour réduire 
la dimensionnalité des images avant la comparaison.
"""

import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from ..brute_force.authentication import radius_search


class EigenfacesModel:
    """
    Modèle d'authentification basé sur la méthode Eigenfaces.
    
    Attributes:
        n_components (int): Nombre de composantes principales à conserver
        mean_face (np.ndarray): Visage moyen calculé sur la gallery
        eigenfaces (np.ndarray): Vecteurs propres (eigenfaces)
        gallery_weights (np.ndarray): Poids des visages de la gallery dans l'espace des eigenfaces
        explained_variance_ratio (np.ndarray): Ratio de variance expliquée par chaque composante
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Initialise un modèle Eigenfaces.
        
        Args:
            n_components (Optional[int]): Nombre de composantes principales à conserver.
                Si None, ce nombre sera déterminé automatiquement.
        """
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None
        self.gallery_weights = None
        self.explained_variance_ratio = None
        
    def fit(self, gallery: np.ndarray, variance_threshold: float = 0.95) -> None:
        """
        Entraîne le modèle Eigenfaces sur la gallery.
        
        Args:
            gallery (np.ndarray): Images de la gallery, chaque ligne est une image aplatie
            variance_threshold (float): Seuil de variance expliquée pour choisir le nombre de composantes
        """
        # Calculer le visage moyen
        self.mean_face = np.mean(gallery, axis=0)
        
        # Centrer les données
        centered_gallery = gallery - self.mean_face
        
        # Calculer la matrice de covariance
        cov_matrix = np.dot(centered_gallery.T, centered_gallery) / (gallery.shape[0] - 1)
        
        # Calculer les vecteurs propres et valeurs propres
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Trier les vecteurs propres par valeur propre décroissante
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculer le ratio de variance expliquée
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total_variance
        
        # Déterminer le nombre de composantes à conserver si non spécifié
        if self.n_components is None:
            cumulative_variance = np.cumsum(self.explained_variance_ratio)
            self.n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        # Limiter le nombre de composantes
        self.n_components = min(self.n_components, gallery.shape[0])
        
        # Sélectionner les principaux vecteurs propres (eigenfaces)
        self.eigenfaces = eigenvectors[:, :self.n_components]
        
        # Projeter la gallery dans l'espace des eigenfaces
        self.gallery_weights = np.dot(centered_gallery, self.eigenfaces)
        
        print(f"Modèle Eigenfaces entraîné avec {self.n_components} composantes principales")
        print(f"Variance expliquée: {np.sum(self.explained_variance_ratio[:self.n_components]):.4f}")
    
    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Projette des images dans l'espace des eigenfaces.
        
        Args:
            images (np.ndarray): Images à projeter, chaque ligne est une image aplatie
            
        Returns:
            np.ndarray: Coordonnées des images dans l'espace des eigenfaces
        """
        # Centrer les images
        centered_images = images - self.mean_face
        
        # Projeter dans l'espace des eigenfaces
        return np.dot(centered_images, self.eigenfaces)
    
    def visualize_eigenfaces(self, image_shape: Tuple[int, int], n_eigenfaces: int = 8) -> None:
        """
        Visualise les eigenfaces principales.
        
        Args:
            image_shape (Tuple[int, int]): Forme originale des images (hauteur, largeur)
            n_eigenfaces (int): Nombre d'eigenfaces à visualiser
        """
        n_eigenfaces = min(n_eigenfaces, self.n_components)
        
        fig, axes = plt.subplots(2, n_eigenfaces // 2 + 1, figsize=(15, 6))
        axes = axes.ravel()
        
        # Afficher le visage moyen
        mean_face_image = self.mean_face.reshape(image_shape)
        axes[0].imshow(mean_face_image, cmap='gray')
        axes[0].set_title('Visage moyen')
        axes[0].axis('off')
        
        # Afficher les principales eigenfaces
        for i in range(n_eigenfaces):
            eigenface = self.eigenfaces[:, i].reshape(image_shape)
            axes[i+1].imshow(eigenface, cmap='gray')
            axes[i+1].set_title(f'Eigenface {i+1}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_variance(self) -> None:
        """
        Visualise la variance expliquée par les composantes principales.
        """
        plt.figure(figsize=(10, 5))
        
        # Variance expliquée par composante
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.explained_variance_ratio[:self.n_components]) + 1), 
                self.explained_variance_ratio[:self.n_components])
        plt.xlabel('Composante principale')
        plt.ylabel('Ratio de variance expliquée')
        plt.title('Variance expliquée par composante')
        
        # Variance cumulée
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.explained_variance_ratio) + 1), 
                 np.cumsum(self.explained_variance_ratio))
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.axvline(x=self.n_components, color='g', linestyle='--')
        plt.xlabel('Nombre de composantes')
        plt.ylabel('Variance expliquée cumulée')
        plt.title('Variance expliquée cumulée')
        
        plt.tight_layout()
        plt.show()


def authenticate(probe: np.ndarray, gallery: np.ndarray, radius: float, model: EigenfacesModel) -> bool:
    """
    Authentifie une image requête en utilisant la méthode Eigenfaces.
    
    Args:
        probe (np.ndarray): Image requête à authentifier
        gallery (np.ndarray): Gallery d'images de référence (non utilisée directement)
        radius (float): Rayon de recherche
        model (EigenfacesModel): Modèle Eigenfaces pré-entraîné
        
    Returns:
        bool: True si l'authentification est réussie, False sinon
    """
    # Vérifier que le modèle est entraîné
    if model.eigenfaces is None:
        raise ValueError("Le modèle Eigenfaces n'est pas entraîné")
        
    # Aplatir l'image requête si nécessaire
    if len(probe.shape) > 1 and probe.ndim > 1:
        probe = probe.flatten()
    
    # Projeter l'image requête dans l'espace des eigenfaces
    probe_weights = model.transform(probe.reshape(1, -1))
    
    # Rechercher les voisins dans l'espace des eigenfaces
    indices, distances = radius_search(model.gallery_weights, probe_weights[0], radius)
    
    # L'authentification réussit s'il y a au moins un voisin dans le rayon
    return len(indices) > 0


def find_best_radius(gallery: np.ndarray, enrolled_probes: np.ndarray, 
                   non_enrolled_probes: np.ndarray, n_components: Optional[int] = None) -> Tuple[float, EigenfacesModel]:
    """
    Trouve le meilleur rayon pour l'authentification par Eigenfaces.
    
    Args:
        gallery (np.ndarray): Gallery d'images de référence
        enrolled_probes (np.ndarray): Images de test d'utilisateurs enregistrés
        non_enrolled_probes (np.ndarray): Images de test d'utilisateurs non enregistrés
        n_components (Optional[int]): Nombre de composantes principales à conserver
        
    Returns:
        Tuple[float, EigenfacesModel]: Rayon optimal et modèle Eigenfaces entraîné
    """
    # Aplatir les images
    gallery_vectors = gallery.reshape(gallery.shape[0], -1)
    enrolled_vectors = enrolled_probes.reshape(enrolled_probes.shape[0], -1)
    non_enrolled_vectors = non_enrolled_probes.reshape(non_enrolled_probes.shape[0], -1)
    
    # Créer et entraîner le modèle Eigenfaces
    model = EigenfacesModel(n_components)
    model.fit(gallery_vectors)
    
    # Projeter les probes dans l'espace des eigenfaces
    enrolled_weights = model.transform(enrolled_vectors)
    non_enrolled_weights = model.transform(non_enrolled_vectors)
    
    # Calculer les distances minimales pour les utilisateurs enregistrés
    min_distances_enrolled = []
    for weights in enrolled_weights:
        distances = np.sqrt(np.sum((model.gallery_weights - weights)**2, axis=1))
        min_distances_enrolled.append(np.min(distances))
    
    # Calculer les distances minimales pour les utilisateurs non enregistrés
    min_distances_non_enrolled = []
    for weights in non_enrolled_weights:
        distances = np.sqrt(np.sum((model.gallery_weights - weights)**2, axis=1))
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
    
    return best_radius, model 