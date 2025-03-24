"""
Module d'authentification par la méthode des Eigenfaces.

Ce module implémente l'authentification faciale basée sur les Eigenfaces (PCA)
en utilisant scikit-learn pour l'analyse en composantes principales.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from matplotlib.figure import Figure
from typing import Tuple, Dict, List, Optional, Union, Any
import time

class EigenfacesModel:
    """
    Modèle d'authentification basé sur les Eigenfaces.
    
    Implémente une méthode d'authentification faciale basée sur la réduction de dimensionnalité
    par Analyse en Composantes Principales (PCA).
    """
    
    def __init__(self, n_components: Optional[int] = None, variance_threshold: float = 0.95):
        """
        Initialise le modèle Eigenfaces.
        
        Args:
            n_components: Nombre de composantes principales à conserver.
                Si None, le nombre sera déterminé automatiquement pour
                expliquer variance_threshold de la variance.
            variance_threshold: Seuil de variance expliquée (entre 0 et 1)
                utilisé si n_components est None.
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.gallery_projections = None
        self.gallery_mean = None
        self.is_fitted = False
        
    def fit(self, gallery: np.ndarray) -> 'EigenfacesModel':
        """
        Entraîne le modèle PCA sur les images de la galerie.
        
        Args:
            gallery: Tableau 2D de taille (n_images, n_pixels) contenant les images prétraitées.
                
        Returns:
            self: Le modèle entraîné.
        """
        # Calculer la moyenne des images
        self.gallery_mean = np.mean(gallery, axis=0)
        
        # Centrer les données
        gallery_centered = gallery - self.gallery_mean
        
        # Déterminer le nombre de composantes automatiquement si n_components est None
        if self.n_components is None:
            # Commencer avec un PCA qui conserve presque toute la variance
            temp_pca = PCA(n_components=min(gallery.shape[0], gallery.shape[1]))
            temp_pca.fit(gallery_centered)
            
            # Trouver le nombre de composantes nécessaires pour atteindre le seuil de variance
            explained_variance_ratio_cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
            self.n_components = np.argmax(explained_variance_ratio_cumsum >= self.variance_threshold) + 1
            print(f"Nombre de composantes automatiquement déterminé: {self.n_components}")
        
        # Créer et entraîner le PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(gallery_centered)
        
        # Projeter la galerie dans l'espace des eigenfaces
        self.gallery_projections = self.pca.transform(gallery_centered)
        
        self.is_fitted = True
        return self
    
    def project(self, images: np.ndarray) -> np.ndarray:
        """
        Projette des images dans l'espace des eigenfaces.
        
        Args:
            images: Tableau 2D de taille (n_images, n_pixels) contenant les images prétraitées.
                
        Returns:
            np.ndarray: Tableau 2D de taille (n_images, n_components) contenant les projections.
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de pouvoir projeter des images")
            
        # Centrer les données avec la moyenne de la galerie
        images_centered = images - self.gallery_mean
        
        # Projeter dans l'espace des eigenfaces
        return self.pca.transform(images_centered)
    
    def reconstruct(self, projections: np.ndarray) -> np.ndarray:
        """
        Reconstruit des images à partir de leurs projections.
        
        Args:
            projections: Tableau 2D de taille (n_images, n_components) contenant les projections.
                
        Returns:
            np.ndarray: Tableau 2D de taille (n_images, n_pixels) contenant les images reconstruites.
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de pouvoir reconstruire des images")
            
        # Reconstruire à partir des projections
        reconstructed = self.pca.inverse_transform(projections)
        
        # Ajouter la moyenne
        return reconstructed + self.gallery_mean
    
    def compute_distances(self, probe_projections: np.ndarray) -> np.ndarray:
        """
        Calcule les distances euclidiennes entre les projections d'une probe et celles de la galerie.
        
        Args:
            probe_projections: Tableau 2D de taille (n_probes, n_components) contenant
                les projections des probes.
                
        Returns:
            np.ndarray: Tableau 2D de taille (n_probes, n_gallery) contenant les distances.
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de pouvoir calculer des distances")
        
        distances = []
        
        for probe_projection in probe_projections:
            # Calculer les distances euclidiennes entre cette projection et toutes celles de la galerie
            dist = np.sqrt(np.sum((self.gallery_projections - probe_projection) ** 2, axis=1))
            distances.append(dist)
            
        return np.array(distances)
    
    def authenticate(self, probe: np.ndarray, radius: float) -> Tuple[bool, float]:
        """
        Authentifie une image probe.
        
        Args:
            probe: Tableau 1D de taille (n_pixels) contenant l'image probe prétraitée.
            radius: Rayon du seuil d'authentification.
                
        Returns:
            Tuple[bool, float]: Un tuple contenant la décision d'authentification (True si authentifié)
                et la distance minimale trouvée.
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de pouvoir authentifier")
            
        # Ajouter une dimension pour avoir un tableau 2D avec une seule image
        probe_reshaped = probe.reshape(1, -1)
        
        # Projeter l'image probe
        probe_projection = self.project(probe_reshaped)
        
        # Calculer les distances avec toutes les images de la galerie
        distances = self.compute_distances(probe_projection)
        
        # Trouver la distance minimale
        min_distance = np.min(distances)
        
        # Authentifier si la distance minimale est inférieure au rayon
        is_authenticated = min_distance <= radius
        
        return is_authenticated, min_distance
    
    def visualize_eigenfaces(self, image_shape: Tuple[int, int], n_eigenfaces: int = 8) -> Figure:
        """
        Visualise les eigenfaces.
        
        Args:
            image_shape: Tuple (height, width) contenant les dimensions de l'image originale.
            n_eigenfaces: Nombre d'eigenfaces à visualiser.
                
        Returns:
            Figure: Figure matplotlib contenant la visualisation des eigenfaces.
        """
        if not self.is_fitted or self.pca is None:
            raise ValueError("Le modèle doit être entraîné avant de pouvoir visualiser les eigenfaces")
            
        n_eigenfaces = min(n_eigenfaces, self.n_components)
        
        # Créer une figure avec n_eigenfaces sous-figures
        n_cols = 4
        n_rows = (n_eigenfaces + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, 3 * n_rows))
        
        # Ajouter la moyenne des visages comme première sous-figure
        ax = fig.add_subplot(n_rows, n_cols, 1)
        ax.imshow(self.gallery_mean.reshape(image_shape), cmap='gray')
        ax.set_title('Visage moyen')
        ax.axis('off')
        
        # Ajouter les eigenfaces
        for i in range(n_eigenfaces - 1):
            ax = fig.add_subplot(n_rows, n_cols, i + 2)
            eigenface = self.pca.components_[i].reshape(image_shape)
            ax.imshow(eigenface, cmap='gray')
            ax.set_title(f'Eigenface {i+1}')
            ax.axis('off')
            
        plt.tight_layout()
        return fig
    
    def visualize_variance(self) -> Figure:
        """
        Visualise la variance expliquée par chaque composante principale.
                
        Returns:
            Figure: Figure matplotlib contenant la visualisation de la variance.
        """
        if not self.is_fitted or self.pca is None:
            raise ValueError("Le modèle doit être entraîné avant de pouvoir visualiser la variance")
            
        # Créer une figure avec deux sous-figures
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Variance expliquée par chaque composante
        variance_ratio = self.pca.explained_variance_ratio_
        ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio)
        ax1.set_xlabel('Composante principale')
        ax1.set_ylabel('Variance expliquée')
        ax1.set_title('Variance expliquée par composante principale')
        
        # Variance expliquée cumulée
        variance_ratio_cumsum = np.cumsum(variance_ratio)
        ax2.plot(range(1, len(variance_ratio_cumsum) + 1), variance_ratio_cumsum, marker='o')
        ax2.set_xlabel('Nombre de composantes principales')
        ax2.set_ylabel('Variance expliquée cumulée')
        ax2.set_title('Variance expliquée cumulée')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% de variance')
        ax2.axhline(y=0.99, color='g', linestyle='--', label='99% de variance')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def get_reconstruction_quality(self, images: np.ndarray) -> Figure:
        """
        Visualise la qualité de reconstruction des images.
        
        Args:
            images: Tableau 2D de taille (n_images, n_pixels) contenant les images à reconstruire.
                
        Returns:
            Figure: Figure matplotlib contenant la visualisation des reconstructions.
        """
        if not self.is_fitted or self.pca is None:
            raise ValueError("Le modèle doit être entraîné avant de pouvoir reconstruire des images")
            
        # Limiter le nombre d'images pour la visualisation
        n_images = min(5, images.shape[0])
        images = images[:n_images]
        
        # Projeter les images dans l'espace des eigenfaces
        projections = self.project(images)
        
        # Reconstruire les images
        reconstructed = self.reconstruct(projections)
        
        # Calculer l'erreur de reconstruction
        error = np.mean((images - reconstructed) ** 2, axis=1)
        
        # Déterminer la forme de l'image
        img_size = int(np.sqrt(images.shape[1]))
        img_shape = (img_size, img_size)
        
        # Créer une figure avec trois rangées (original, reconstruit, différence)
        fig, axes = plt.subplots(3, n_images, figsize=(15, 9))
        
        for i in range(n_images):
            # Image originale
            axes[0, i].imshow(images[i].reshape(img_shape), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Image reconstruite
            axes[1, i].imshow(reconstructed[i].reshape(img_shape), cmap='gray')
            axes[1, i].set_title(f'Reconstruit\nErreur: {error[i]:.4f}')
            axes[1, i].axis('off')
            
            # Différence
            diff = images[i] - reconstructed[i]
            axes[2, i].imshow(diff.reshape(img_shape), cmap='RdBu_r')
            axes[2, i].set_title('Différence')
            axes[2, i].axis('off')
            
        plt.tight_layout()
        return fig


def evaluate_performance(model: EigenfacesModel, enrolled_probes: np.ndarray, 
                        non_enrolled_probes: np.ndarray, radius: float, 
                        progress_callback=None) -> Dict[str, Any]:
    """
    Évalue les performances du modèle Eigenfaces avec un rayon donné.
    
    Args:
        model: Modèle Eigenfaces entraîné.
        enrolled_probes: Tableau 2D contenant les probes des utilisateurs enregistrés.
        non_enrolled_probes: Tableau 2D contenant les probes des utilisateurs non enregistrés.
        radius: Rayon du seuil d'authentification.
        progress_callback: Fonction de rappel pour indiquer la progression (facultatif).
            
    Returns:
        Dict[str, Any]: Dictionnaire contenant les métriques de performance.
    """
    # Vérifier si le modèle est entraîné
    if not model.is_fitted:
        raise ValueError("Le modèle doit être entraîné avant de pouvoir évaluer les performances")
    
    start_time = time.time()
    
    # Projeter les probes
    if progress_callback:
        progress_callback(15, "Projection des probes...")
    
    enrolled_projections = model.project(enrolled_probes)
    non_enrolled_projections = model.project(non_enrolled_probes)
    
    # Calculer les distances pour les deux ensembles
    if progress_callback:
        progress_callback(30, "Calcul des distances...")
    
    enrolled_distances = model.compute_distances(enrolled_projections)
    non_enrolled_distances = model.compute_distances(non_enrolled_projections)
    
    # Prendre la distance minimale pour chaque probe
    enrolled_min_distances = np.min(enrolled_distances, axis=1)
    non_enrolled_min_distances = np.min(non_enrolled_distances, axis=1)
    
    # Prévoir les authentifications
    if progress_callback:
        progress_callback(45, "Prédiction des authentifications...")
    
    enrolled_authentications = enrolled_min_distances <= radius
    non_enrolled_authentications = non_enrolled_min_distances <= radius
    
    # Calculer les vrais positifs, faux positifs, vrais négatifs, faux négatifs
    tp = np.sum(enrolled_authentications)  # Vrais positifs
    fn = len(enrolled_authentications) - tp  # Faux négatifs
    tn = len(non_enrolled_authentications) - np.sum(non_enrolled_authentications)  # Vrais négatifs
    fp = np.sum(non_enrolled_authentications)  # Faux positifs
    
    # Calculer les métriques
    if progress_callback:
        progress_callback(60, "Calcul des métriques de performance...")
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Créer les labels et prédictions pour la matrice de confusion
    true_labels = np.concatenate([np.ones(len(enrolled_authentications)), 
                                np.zeros(len(non_enrolled_authentications))])
    predicted_labels = np.concatenate([enrolled_authentications.astype(int), 
                                     non_enrolled_authentications.astype(int)])
    
    confmat = confusion_matrix(true_labels, predicted_labels)
    
    execution_time = time.time() - start_time
    
    # Résultat
    results = {
        'radius': radius,
        'performance': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'tp': int(tp),
            'fn': int(fn),
            'tn': int(tn),
            'fp': int(fp)
        },
        'enrolled_distances': enrolled_min_distances.tolist(),
        'non_enrolled_distances': non_enrolled_min_distances.tolist(),
        'confusion_matrix': confmat.tolist(),
        'execution_time': execution_time
    }
    
    return results


def find_best_radius(gallery: np.ndarray, enrolled_probes: np.ndarray, 
                    non_enrolled_probes: np.ndarray, n_components: Optional[int] = None,
                    progress_callback=None) -> Tuple[float, EigenfacesModel, Dict[str, Any]]:
    """
    Trouve le meilleur rayon pour l'authentification et évalue les performances.
    
    Args:
        gallery: Tableau 2D contenant les images de la galerie.
        enrolled_probes: Tableau 2D contenant les probes des utilisateurs enregistrés.
        non_enrolled_probes: Tableau 2D contenant les probes des utilisateurs non enregistrés.
        n_components: Nombre de composantes principales à utiliser (facultatif).
        progress_callback: Fonction de rappel pour indiquer la progression (facultatif).
            
    Returns:
        Tuple[float, EigenfacesModel, Dict[str, Any]]: Tuple contenant le meilleur rayon,
            le modèle entraîné et les résultats des performances.
    """
    start_time = time.time()
    
    # Entraîner le modèle
    if progress_callback:
        progress_callback(5, "Entraînement du modèle Eigenfaces...")
    
    model = EigenfacesModel(n_components=n_components)
    model.fit(gallery)
    
    # Projeter les probes
    if progress_callback:
        progress_callback(20, "Projection des probes...")
    
    enrolled_projections = model.project(enrolled_probes)
    non_enrolled_projections = model.project(non_enrolled_probes)
    
    # Calculer les distances
    if progress_callback:
        progress_callback(30, "Calcul des distances pour toutes les probes...")
    
    enrolled_distances = model.compute_distances(enrolled_projections)
    non_enrolled_distances = model.compute_distances(non_enrolled_projections)
    
    # Calculer les distances minimales
    enrolled_min_distances = np.min(enrolled_distances, axis=1)
    non_enrolled_min_distances = np.min(non_enrolled_distances, axis=1)
    
    # Déterminer les rayons à tester
    all_distances = np.concatenate([enrolled_min_distances, non_enrolled_min_distances])
    min_radius = np.min(all_distances)
    max_radius = np.max(all_distances)
    
    # Créer une série de rayons à tester (10 points uniformément répartis)
    radiuses = np.linspace(min_radius, max_radius, 10)
    
    # Tester chaque rayon et trouver celui qui maximise le F1-score
    best_f1 = -1
    best_radius = None
    best_results = None
    
    if progress_callback:
        progress_callback(40, "Évaluation des performances pour différents rayons...")
    
    for i, radius in enumerate(radiuses):
        # Calculer les métriques pour ce rayon
        results = evaluate_performance(model, enrolled_probes, non_enrolled_probes, radius)
        f1 = results['performance']['f1_score']
        
        if f1 > best_f1:
            best_f1 = f1
            best_radius = radius
            best_results = results
            
        # Mise à jour de la progression
        if progress_callback:
            progress_percent = 40 + (i + 1) / len(radiuses) * 30
            progress_callback(progress_percent, f"Évaluation du rayon {i+1}/{len(radiuses)}")
    
    # Ajouter des métadonnées supplémentaires aux résultats
    best_results['execution_time'] = time.time() - start_time
    best_results['all_radiuses'] = radiuses.tolist()
    best_results['all_f1_scores'] = [
        evaluate_performance(model, enrolled_probes, non_enrolled_probes, r)['performance']['f1_score'] 
        for r in radiuses
    ]
    
    if progress_callback:
        progress_callback(75, f"Meilleur rayon trouvé: {best_radius:.4f}")
    
    return best_radius, model, best_results


def visualize_radius_performances(results: Dict[str, Any]) -> Figure:
    """
    Visualise les performances pour différents rayons.
    
    Args:
        results: Dictionnaire contenant les résultats de l'évaluation.
            
    Returns:
        Figure: Figure matplotlib contenant la visualisation des performances.
    """
    # Vérifier que les données nécessaires sont présentes
    if 'all_radiuses' not in results or 'all_f1_scores' not in results:
        raise ValueError("Les résultats ne contiennent pas les données nécessaires pour cette visualisation")
    
    radiuses = results['all_radiuses']
    f1_scores = results['all_f1_scores']
    best_radius = results['radius']
    best_f1 = results['performance']['f1_score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tracer la courbe du F1-score en fonction du rayon
    ax.plot(radiuses, f1_scores, marker='o', linestyle='-', color='blue')
    
    # Marquer le meilleur rayon
    ax.plot(best_radius, best_f1, marker='*', markersize=15, color='red', 
            label=f'Meilleur rayon: {best_radius:.4f}\nF1-score: {best_f1:.4f}')
    
    ax.set_xlabel('Rayon')
    ax.set_ylabel('F1-score')
    ax.set_title('F1-score en fonction du rayon')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    return fig


def visualize_distances_distribution(results: Dict[str, Any]) -> Figure:
    """
    Visualise la distribution des distances pour les utilisateurs enregistrés et non enregistrés.
    
    Args:
        results: Dictionnaire contenant les résultats de l'évaluation.
            
    Returns:
        Figure: Figure matplotlib contenant la visualisation des distributions.
    """
    # Vérifier que les données nécessaires sont présentes
    if 'enrolled_distances' not in results or 'non_enrolled_distances' not in results:
        raise ValueError("Les résultats ne contiennent pas les données nécessaires pour cette visualisation")
    
    enrolled_distances = np.array(results['enrolled_distances'])
    non_enrolled_distances = np.array(results['non_enrolled_distances'])
    best_radius = results['radius']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tracer les histogrammes des distances
    bins = 30
    ax.hist(enrolled_distances, bins=bins, alpha=0.5, label='Utilisateurs enregistrés')
    ax.hist(non_enrolled_distances, bins=bins, alpha=0.5, label='Utilisateurs non enregistrés')
    
    # Tracer une ligne verticale pour le rayon optimal
    ax.axvline(best_radius, color='red', linestyle='--', 
               label=f'Rayon optimal: {best_radius:.4f}')
    
    ax.set_xlabel('Distance minimale')
    ax.set_ylabel('Nombre de probes')
    ax.set_title('Distribution des distances minimales')
    ax.legend()
    
    plt.tight_layout()
    return fig


def visualize_confusion_matrix(results: Dict[str, Any]) -> Figure:
    """
    Visualise la matrice de confusion.
    
    Args:
        results: Dictionnaire contenant les résultats de l'évaluation.
            
    Returns:
        Figure: Figure matplotlib contenant la visualisation de la matrice de confusion.
    """
    # Vérifier que les données nécessaires sont présentes
    if 'confusion_matrix' not in results:
        raise ValueError("Les résultats ne contiennent pas la matrice de confusion")
    
    confmat = np.array(results['confusion_matrix'])
    
    # Calculer les pourcentages pour chaque catégorie
    confmat_norm = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    confmat_norm = np.round(confmat_norm * 100, 2)
    
    # Créer les étiquettes avec nombres et pourcentages
    labels = []
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            labels.append(f"{confmat[i, j]}\n({confmat_norm[i, j]}%)")
    labels = np.array(labels).reshape(confmat.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tracer la matrice de confusion
    sns.heatmap(confmat, annot=labels, fmt='', cmap='Blues', ax=ax)
    
    # Ajouter les labels
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Vrai')
    ax.set_title('Matrice de confusion')
    ax.set_xticklabels(['Non authentifié', 'Authentifié'])
    ax.set_yticklabels(['Non enregistré', 'Enregistré'])
    
    plt.tight_layout()
    return fig

def authenticate(probe: np.ndarray, gallery: np.ndarray, radius: float = None, model: EigenfacesModel = None) -> bool:
    """
    Authentifie une image probe en utilisant la méthode Eigenfaces.
    
    Args:
        probe: Image probe à authentifier (aplatie ou 2D)
        gallery: Galerie d'images de référence (aplaties ou 2D)
        radius: Rayon de décision pour l'authentification
        model: Modèle Eigenfaces préentraîné (facultatif)
        
    Returns:
        bool: True si l'authentification est réussie, False sinon
    """
    # S'assurer que probe est un vecteur 1D
    if probe.ndim > 1:
        probe = probe.flatten()

    # Si aucun modèle n'est fourni, en créer un nouveau et l'entraîner
    if model is None:
        # Vérifier si gallery est 2D (déjà aplatie) ou 3D
        if gallery.ndim == 3:
            gallery_flattened = gallery.reshape(gallery.shape[0], -1)
        else:
            gallery_flattened = gallery
            
        # Créer et entraîner le modèle
        model = EigenfacesModel()
        model.fit(gallery_flattened)
        
        # Déterminer un rayon par défaut si non spécifié
        if radius is None:
            # Calculer les distances entre chaque paire d'images de la galerie
            all_projections = model.gallery_projections
            all_distances = []
            
            for i in range(len(all_projections)):
                for j in range(i+1, len(all_projections)):
                    dist = np.sqrt(np.sum((all_projections[i] - all_projections[j]) ** 2))
                    all_distances.append(dist)
            
            # Utiliser la moyenne des distances comme rayon par défaut
            if all_distances:
                radius = np.mean(all_distances) * 0.8
            else:
                radius = 0.5  # Valeur arbitraire si galerie trop petite

    # Authentifier avec le modèle
    is_authenticated, _ = model.authenticate(probe, radius)
    return is_authenticated 