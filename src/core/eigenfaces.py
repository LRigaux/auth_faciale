"""
Eigenfaces authentication module for facial authentication.

This module implements facial authentication based on Eigenfaces (PCA)
using scikit-learn for principal component analysis.
"""

import numpy as np
import time
from typing import Dict, Tuple, List, Optional, Any, Callable, Union
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

class EigenfacesModel:
    """
    Authentication model based on Eigenfaces.
    
    Implements a facial authentication method based on dimensionality reduction
    through Principal Component Analysis (PCA).
    """
    
    def __init__(self, n_components: Optional[int] = None, variance_threshold: float = 0.95):
        """
        Initialize the Eigenfaces model.
        
        Args:
            n_components: Number of principal components to keep.
                If None, the number will be automatically determined to
                explain variance_threshold of the variance.
            variance_threshold: Explained variance threshold (between 0 and 1)
                used if n_components is None.
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.gallery_projections = None
        self.gallery_mean = None
        self.is_fitted = False
        
    def fit(self, gallery: np.ndarray) -> 'EigenfacesModel':
        """
        Train the PCA model on gallery images.
        
        Args:
            gallery: Gallery images (can be of shape n_images, height, width or already flattened)
            
        Returns:
            self: The trained model.
        """
        # Make sure the gallery is correctly flattened
        if gallery.ndim == 3:
            # If images are 3D (n_images, height, width)
            n_samples = gallery.shape[0]
            gallery_flat = gallery.reshape(n_samples, -1)
        else:
            # If images are already flattened
            gallery_flat = gallery.copy()
        
        # Calculate the mean of the images
        self.gallery_mean = np.mean(gallery_flat, axis=0)
        
        # Center the data
        gallery_centered = gallery_flat - self.gallery_mean
        
        # Automatically determine the number of components if n_components is None
        if self.n_components is None:
            # Start with a PCA that keeps almost all the variance
            temp_pca = PCA(n_components=min(gallery_flat.shape[0], gallery_flat.shape[1]))
            temp_pca.fit(gallery_centered)
            
            # Find the number of components needed to reach the variance threshold
            explained_variance_ratio_cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
            self.n_components = np.argmax(explained_variance_ratio_cumsum >= self.variance_threshold) + 1
            print(f"Number of components automatically determined: {self.n_components}")
        
        # Create and train the PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(gallery_centered)
        
        # Project the gallery into the eigenfaces space
        self.gallery_projections = self.pca.transform(gallery_centered)
        
        self.is_fitted = True
        return self
    
    def project(self, images: np.ndarray) -> np.ndarray:
        """
        Project images into the eigenfaces space.
        
        Args:
            images: 2D array of shape (n_images, n_pixels) containing preprocessed images.
                
        Returns:
            np.ndarray: 2D array of shape (n_images, n_components) containing projections.
        """
        if not self.is_fitted:
            raise ValueError("The model must be trained before projecting images")
            
        # Center the data with the gallery mean
        images_centered = images - self.gallery_mean
        
        # Project into the eigenfaces space
        return self.pca.transform(images_centered)
    
    def reconstruct(self, projections: np.ndarray) -> np.ndarray:
        """
        Reconstruct images from their projections.
        
        Args:
            projections: 2D array of shape (n_images, n_components) containing projections.
                
        Returns:
            np.ndarray: 2D array of shape (n_images, n_pixels) containing reconstructed images.
        """
        if not self.is_fitted:
            raise ValueError("The model must be trained before reconstructing images")
            
        # Reconstruct from projections
        reconstructed = self.pca.inverse_transform(projections)
        
        # Add the mean
        return reconstructed + self.gallery_mean
    
    def compute_distances(self, probe_projections: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between probe projections and gallery projections.
        
        Args:
            probe_projections: 2D array of shape (n_probes, n_components) containing
                projections of probes.
                
        Returns:
            np.ndarray: 2D array of shape (n_probes, n_gallery) containing distances.
        """
        if not self.is_fitted:
            raise ValueError("The model must be trained before computing distances")
        
        distances = []
        
        for probe_projection in probe_projections:
            # Compute Euclidean distances between this projection and all gallery projections
            dist = np.sqrt(np.sum((self.gallery_projections - probe_projection) ** 2, axis=1))
            distances.append(dist)
            
        return np.array(distances)
    
    def find_closest_match(self, probe: np.ndarray) -> Tuple[int, float]:
        """
        Find the closest match in the gallery for a probe.
        
        Args:
            probe: Probe image to match
            
        Returns:
            Tuple[int, float]: Index of the closest gallery image and the distance
        """
        if not self.is_fitted:
            raise ValueError("The model must be trained before finding matches")
            
        # Ensure probe is a 2D array with a single image
        probe_reshaped = probe.reshape(1, -1) if probe.ndim == 1 else probe
        
        # Project the probe
        probe_projection = self.project(probe_reshaped)
        
        # Compute distances with all gallery images
        distances = self.compute_distances(probe_projection)[0]
        
        # Find the closest match
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        return min_idx, min_distance
    
    def authenticate(self, probe: np.ndarray, threshold: float) -> Tuple[bool, int, float]:
        """
        Authenticate a probe image.
        
        Args:
            probe: Probe image to authenticate.
            threshold: Authentication threshold radius.
                
        Returns:
            Tuple[bool, int, float]: A tuple containing the authentication decision (True if authenticated),
                the index of the closest match, and the minimum distance found.
        """
        if not self.is_fitted:
            raise ValueError("The model must be trained before authenticating")
        
        # Find closest match
        closest_idx, min_distance = self.find_closest_match(probe)
        
        # Authenticate if the minimum distance is below the threshold
        is_authenticated = min_distance <= threshold
        
        return is_authenticated, closest_idx, min_distance


def authenticate(probe: np.ndarray, gallery: np.ndarray, threshold: float = None, 
                model: EigenfacesModel = None) -> Tuple[bool, int, float]:
    """
    Authenticate a probe image using the Eigenfaces method.
    
    Args:
        probe: Probe image to authenticate (flattened or 2D)
        gallery: Gallery of reference images (flattened or 2D)
        threshold: Decision threshold for authentication
        model: Pre-trained Eigenfaces model (optional)
        
    Returns:
        Tuple[bool, int, float]: Authentication result (True if authenticated),
                                 index of closest match, and the minimum distance
    """
    # If no model is provided, create a new one and train it
    if model is None:
        # Create and train the model
        model = EigenfacesModel()
        model.fit(gallery)
        
        # Determine a default threshold if not specified
        if threshold is None:
            # Compute distances between each pair of gallery images
            all_projections = model.gallery_projections
            all_distances = []
            
            for i in range(len(all_projections)):
                for j in range(i+1, len(all_projections)):
                    dist = np.sqrt(np.sum((all_projections[i] - all_projections[j]) ** 2))
                    all_distances.append(dist)
            
            # Use the mean of distances as default threshold
            if all_distances:
                threshold = np.mean(all_distances) * 0.8
            else:
                threshold = 0.5  # Arbitrary value if gallery too small

    # Authenticate with the model
    is_authenticated, closest_idx, min_distance = model.authenticate(probe, threshold)
    return is_authenticated, closest_idx, min_distance


def evaluate_performance(model: EigenfacesModel, gallery: np.ndarray, enrolled_probes: np.ndarray, 
                        non_enrolled_probes: np.ndarray, threshold: float, 
                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Evaluate Eigenfaces model performance with a given threshold.
    
    Args:
        model: Trained Eigenfaces model.
        gallery: Gallery of reference images
        enrolled_probes: Probes from enrolled users
        non_enrolled_probes: Probes from non-enrolled users
        threshold: Authentication threshold
        progress_callback: Callback function for progress reporting
        
    Returns:
        Dict[str, Any]: Dictionary containing performance metrics
    """
    # Check if the model is trained
    if not model.is_fitted:
        raise ValueError("The model must be trained before evaluating performance")
    
    start_time = time.time()
    
    # Project the probes
    if progress_callback:
        progress_callback(15, "Projecting probes...")
    
    enrolled_projections = model.project(enrolled_probes)
    non_enrolled_projections = model.project(non_enrolled_probes)
    
    # Compute distances for both sets
    if progress_callback:
        progress_callback(30, "Computing distances...")
    
    enrolled_distances = model.compute_distances(enrolled_projections)
    non_enrolled_distances = model.compute_distances(non_enrolled_projections)
    
    # Take the minimum distance for each probe
    enrolled_min_distances = np.min(enrolled_distances, axis=1)
    non_enrolled_min_distances = np.min(non_enrolled_distances, axis=1)
    
    # Predict authentications
    if progress_callback:
        progress_callback(45, "Predicting authentications...")
    
    enrolled_authentications = enrolled_min_distances <= threshold
    non_enrolled_authentications = non_enrolled_min_distances <= threshold
    
    # Calculate true positives, false positives, true negatives, false negatives
    tp = np.sum(enrolled_authentications)  # True positives
    fn = len(enrolled_authentications) - tp  # False negatives
    tn = len(non_enrolled_authentications) - np.sum(non_enrolled_authentications)  # True negatives
    fp = np.sum(non_enrolled_authentications)  # False positives
    
    # Calculate metrics
    if progress_callback:
        progress_callback(60, "Computing performance metrics...")
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create labels and predictions for confusion matrix
    true_labels = np.concatenate([np.ones(len(enrolled_authentications)), 
                                np.zeros(len(non_enrolled_authentications))])
    predicted_labels = np.concatenate([enrolled_authentications.astype(int), 
                                     non_enrolled_authentications.astype(int)])
    
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    execution_time = time.time() - start_time
    
    if progress_callback:
        progress_callback(100, f"Evaluation completed in {execution_time:.2f} seconds")
    
    # Result
    results = {
        'threshold': threshold,
        'performance': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'tp': int(tp),
            'fn': int(fn),
            'tn': int(tn),
            'fp': int(fp)
        },
        'enrolled_distances': enrolled_min_distances.tolist(),
        'non_enrolled_distances': non_enrolled_min_distances.tolist(),
        'confusion_matrix': conf_matrix.tolist(),
        'execution_time': execution_time
    }
    
    return results


def find_best_threshold(gallery: np.ndarray, enrolled_probes: np.ndarray, 
                       non_enrolled_probes: np.ndarray, n_components: Optional[int] = None,
                       progress_callback: Optional[Callable] = None) -> Tuple[float, EigenfacesModel, Dict[str, Any]]:
    """
    Find the best threshold for authentication and evaluate performance.
    
    Args:
        gallery: Gallery of reference images
        enrolled_probes: Probes from enrolled users
        non_enrolled_probes: Probes from non-enrolled users
        n_components: Number of principal components to use (optional)
        progress_callback: Callback function for progress reporting
        
    Returns:
        Tuple[float, EigenfacesModel, Dict[str, Any]]: Best threshold,
            trained model and performance results
    """
    start_time = time.time()
    
    # Train the model
    if progress_callback:
        progress_callback(5, "Training Eigenfaces model...")
    
    model = EigenfacesModel(n_components=n_components)
    model.fit(gallery)
    
    # Project the probes
    if progress_callback:
        progress_callback(20, "Projecting probes...")
    
    enrolled_projections = model.project(enrolled_probes)
    non_enrolled_projections = model.project(non_enrolled_probes)
    
    # Compute distances
    if progress_callback:
        progress_callback(30, "Computing distances for all probes...")
    
    enrolled_distances = model.compute_distances(enrolled_projections)
    non_enrolled_distances = model.compute_distances(non_enrolled_projections)
    
    # Compute minimum distances
    enrolled_min_distances = np.min(enrolled_distances, axis=1)
    non_enrolled_min_distances = np.min(non_enrolled_distances, axis=1)
    
    # Determine thresholds to test
    all_distances = np.concatenate([enrolled_min_distances, non_enrolled_min_distances])
    min_threshold = np.min(all_distances)
    max_threshold = np.max(all_distances)
    
    # Create a series of thresholds to test (10 points uniformly distributed)
    thresholds = np.linspace(min_threshold, max_threshold, 10)
    
    # Test each threshold and find the one that maximizes the F1-score
    best_f1 = -1
    best_threshold = None
    best_results = None
    
    if progress_callback:
        progress_callback(40, "Evaluating performance for different thresholds...")
    
    for i, threshold in enumerate(thresholds):
        # Calculate metrics for this threshold
        enrolled_authentications = enrolled_min_distances <= threshold
        non_enrolled_authentications = non_enrolled_min_distances <= threshold
        
        tp = np.sum(enrolled_authentications)
        fn = len(enrolled_authentications) - tp
        tn = len(non_enrolled_authentications) - np.sum(non_enrolled_authentications)
        fp = np.sum(non_enrolled_authentications)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
        if progress_callback:
            progress = 40 + 40 * (i + 1) / len(thresholds)
            progress_callback(progress, f"Evaluating threshold {i+1}/{len(thresholds)}")
    
    # Compute full performance metrics for the best threshold
    if progress_callback:
        progress_callback(80, f"Computing detailed metrics for best threshold: {best_threshold:.4f}")
    
    best_results = evaluate_performance(
        model, gallery, enrolled_probes, non_enrolled_probes, best_threshold, progress_callback
    )
    
    # Add additional metadata to results
    best_results['execution_time'] = time.time() - start_time
    best_results['all_thresholds'] = thresholds.tolist()
    best_results['n_components'] = model.n_components
    
    if progress_callback:
        progress_callback(100, f"Best threshold found: {best_threshold:.4f}")
    
    return best_threshold, model, best_results 