"""
Brute force authentication module for facial authentication.

This module implements the brute force authentication method,
which directly compares the query image with all gallery images.
"""

import numpy as np
import time
from typing import Dict, Tuple, List, Optional, Any, Callable, Union

def compute_distances(data: np.ndarray, query: np.ndarray, metric: str = "L2") -> np.ndarray:
    """
    Compute distances between a query vector and all vectors in a dataset.
    
    Args:
        data: Dataset of vectors (each row is a vector)
        query: Query vector
        metric: Distance metric to use ("L1", "L2" or "cosine")
        
    Returns:
        np.ndarray: Array of distances
        
    Raises:
        ValueError: If the metric is not recognized
    """
    if metric == "L1":
        # Manhattan distance
        return np.sum(np.abs(data - query), axis=1)
    elif metric == "L2":
        # Euclidean distance (squared)
        return np.sqrt(np.sum((data - query)**2, axis=1))
    elif metric == "cosine":
        # Cosine distance
        dot_product = np.dot(data, query)
        norm_data = np.sqrt(np.sum(data**2, axis=1))
        norm_query = np.sqrt(np.sum(query**2))
        
        # Avoid division by zero
        valid_indices = (norm_data != 0) & (norm_query != 0)
        similarities = np.zeros(len(data))
        similarities[valid_indices] = dot_product[valid_indices] / (norm_data[valid_indices] * norm_query)
        
        # Convert similarity to distance (1 - similarity, clamped to [0, 2])
        return np.clip(1 - similarities, 0, 2)
    else:
        raise ValueError("Unrecognized metric. Use 'L1', 'L2' or 'cosine'.")

def find_closest_match(gallery: np.ndarray, probe: np.ndarray, metric: str = "L2") -> Tuple[int, float]:
    """
    Find the closest match of a probe in the gallery.
    
    Args:
        gallery: Gallery images in vector form (each row is an image)
        probe: Probe image to authenticate
        metric: Distance metric to use ("L1", "L2" or "cosine")
        
    Returns:
        Tuple[int, float]: Tuple containing the index of the closest match and the minimum distance
    """
    # Ensure probe is a 1D vector
    probe_vector = probe.flatten() if probe.ndim > 1 else probe
    
    # Compute distances
    distances = compute_distances(gallery, probe_vector, metric)
    
    # Find minimum distance and corresponding index
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    
    return min_idx, min_distance

def authenticate(probe: np.ndarray, gallery: np.ndarray, threshold: float, metric: str = "L2") -> Tuple[bool, int, float]:
    """
    Authenticate a probe image by finding similar faces in the gallery.
    
    Args:
        probe: Query image to authenticate
        gallery: Gallery of reference images
        threshold: Distance threshold for authentication
        metric: Distance metric to use ("L1", "L2" or "cosine")
        
    Returns:
        Tuple[bool, int, float]: Authentication result (True if authenticated),
                                  index of closest match, and the minimum distance
    """
    # Find closest match
    closest_idx, min_distance = find_closest_match(gallery, probe, metric)
    
    # Authentication succeeds if the minimum distance is below the threshold
    is_authenticated = min_distance <= threshold
    
    return is_authenticated, closest_idx, min_distance

def evaluate_performance(gallery: np.ndarray, enrolled_probes: np.ndarray, 
                        non_enrolled_probes: np.ndarray, threshold: float,
                        metric: str = "L2", progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Evaluate brute force authentication performance with a given threshold.
    
    Args:
        gallery: Gallery of reference images
        enrolled_probes: Probes from enrolled users
        non_enrolled_probes: Probes from non-enrolled users
        threshold: Authentication threshold
        metric: Distance metric to use
        progress_callback: Callback function for progress reporting
        
    Returns:
        Dict[str, Any]: Dictionary containing performance metrics
    """
    start_time = time.time()
    
    if progress_callback:
        progress_callback(10, "Computing distances for enrolled probes...")
    
    # Compute minimum distances for enrolled probes
    enrolled_min_distances = []
    for i, probe in enumerate(enrolled_probes):
        _, min_distance = find_closest_match(gallery, probe, metric)
        enrolled_min_distances.append(min_distance)
        if progress_callback and i % max(1, len(enrolled_probes) // 10) == 0:
            progress = 10 + 40 * (i / len(enrolled_probes))
            progress_callback(progress, f"Processing enrolled probe {i+1}/{len(enrolled_probes)}")
    
    if progress_callback:
        progress_callback(50, "Computing distances for non-enrolled probes...")
    
    # Compute minimum distances for non-enrolled probes
    non_enrolled_min_distances = []
    for i, probe in enumerate(non_enrolled_probes):
        _, min_distance = find_closest_match(gallery, probe, metric)
        non_enrolled_min_distances.append(min_distance)
        if progress_callback and i % max(1, len(non_enrolled_probes) // 10) == 0:
            progress = 50 + 40 * (i / len(non_enrolled_probes))
            progress_callback(progress, f"Processing non-enrolled probe {i+1}/{len(non_enrolled_probes)}")
    
    # Convert to numpy arrays
    enrolled_min_distances = np.array(enrolled_min_distances)
    non_enrolled_min_distances = np.array(non_enrolled_min_distances)
    
    # Compute authentication decisions
    enrolled_authentications = enrolled_min_distances <= threshold
    non_enrolled_authentications = non_enrolled_min_distances <= threshold
    
    # Compute true positives, false positives, true negatives, false negatives
    tp = np.sum(enrolled_authentications)  # True positives
    fn = len(enrolled_authentications) - tp  # False negatives
    tn = len(non_enrolled_authentications) - np.sum(non_enrolled_authentications)  # True negatives
    fp = np.sum(non_enrolled_authentications)  # False positives
    
    # Compute metrics
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
    
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    execution_time = time.time() - start_time
    
    if progress_callback:
        progress_callback(100, f"Evaluation completed in {execution_time:.2f} seconds")
    
    # Result
    results = {
        'threshold': threshold,
        'metric': metric,
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
                       non_enrolled_probes: np.ndarray, metric: str = "L2",
                       progress_callback: Optional[Callable] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Find the best threshold for authentication and evaluate performance.
    
    Args:
        gallery: Gallery of reference images
        enrolled_probes: Probes from enrolled users
        non_enrolled_probes: Probes from non-enrolled users
        metric: Distance metric to use
        progress_callback: Callback function for progress reporting
        
    Returns:
        Tuple[float, Dict[str, Any]]: Best threshold and performance results
    """
    start_time = time.time()
    
    if progress_callback:
        progress_callback(5, "Computing distances for threshold estimation...")
    
    # Compute minimum distances for enrolled and non-enrolled probes
    enrolled_min_distances = []
    for i, probe in enumerate(enrolled_probes):
        _, min_distance = find_closest_match(gallery, probe, metric)
        enrolled_min_distances.append(min_distance)
        if progress_callback and i % max(1, len(enrolled_probes) // 5) == 0:
            progress = 5 + 30 * (i / len(enrolled_probes))
            progress_callback(progress, f"Processing enrolled probe {i+1}/{len(enrolled_probes)}")
    
    non_enrolled_min_distances = []
    for i, probe in enumerate(non_enrolled_probes):
        _, min_distance = find_closest_match(gallery, probe, metric)
        non_enrolled_min_distances.append(min_distance)
        if progress_callback and i % max(1, len(non_enrolled_probes) // 5) == 0:
            progress = 35 + 30 * (i / len(non_enrolled_probes))
            progress_callback(progress, f"Processing non-enrolled probe {i+1}/{len(non_enrolled_probes)}")
    
    # Convert to numpy arrays
    enrolled_min_distances = np.array(enrolled_min_distances)
    non_enrolled_min_distances = np.array(non_enrolled_min_distances)
    
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
        progress_callback(65, "Evaluating performance for different thresholds...")
    
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
            progress = 65 + 25 * (i + 1) / len(thresholds)
            progress_callback(progress, f"Evaluating threshold {i+1}/{len(thresholds)}")
    
    # Compute full performance metrics for the best threshold
    if progress_callback:
        progress_callback(90, f"Computing detailed metrics for best threshold: {best_threshold:.4f}")
        
    best_results = evaluate_performance(
        gallery, enrolled_probes, non_enrolled_probes, best_threshold, metric
    )
    
    # Add additional metadata to results
    best_results['all_thresholds'] = thresholds.tolist()
    best_results['execution_time'] = time.time() - start_time
    
    if progress_callback:
        progress_callback(100, f"Best threshold found: {best_threshold:.4f}")
    
    return best_threshold, best_results 