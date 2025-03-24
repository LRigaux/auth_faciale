"""
Evaluation module for facial authentication methods.

This module provides functions to evaluate and compare the performance
of different facial authentication methods.
"""

import numpy as np
import time
import json
from typing import Dict, List, Any, Optional, Callable, Union, Tuple

def evaluate_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate performance metrics from authentication results.
    
    Args:
        results: Dictionary containing authentication results (confusion matrix)
        
    Returns:
        Dict[str, Any]: Dictionary containing performance metrics
    """
    # Extract confusion matrix
    conf_matrix = np.array(results['confusion_matrix'])
    
    if conf_matrix.shape == (2, 2):
        tn, fp = conf_matrix[0, 0], conf_matrix[0, 1]
        fn, tp = conf_matrix[1, 0], conf_matrix[1, 1]
    else:
        # Try to use performance data directly if available
        if 'performance' in results:
            return results['performance']
        else:
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'specificity': 0,
                'f1_score': 0
            }
    
    # Calculate metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'tp': int(tp),
        'fn': int(fn),
        'tn': int(tn),
        'fp': int(fp)
    }

def compare_methods(results_brute_force: Dict[str, Any], 
                   results_eigenfaces: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare the performance of brute force and eigenfaces methods.
    
    Args:
        results_brute_force: Results from brute force method
        results_eigenfaces: Results from eigenfaces method
        
    Returns:
        Dict[str, Any]: Comparison of methods
    """
    # Extract performance metrics
    bf_perf = results_brute_force.get('performance', {})
    ef_perf = results_eigenfaces.get('performance', {})
    
    # Compare execution times
    bf_time = results_brute_force.get('execution_time', 0)
    ef_time = results_eigenfaces.get('execution_time', 0)
    
    # Create comparison dictionary
    comparison = {
        'accuracy': {
            'brute_force': bf_perf.get('accuracy', 0),
            'eigenfaces': ef_perf.get('accuracy', 0),
            'difference': ef_perf.get('accuracy', 0) - bf_perf.get('accuracy', 0)
        },
        'precision': {
            'brute_force': bf_perf.get('precision', 0),
            'eigenfaces': ef_perf.get('precision', 0),
            'difference': ef_perf.get('precision', 0) - bf_perf.get('precision', 0)
        },
        'recall': {
            'brute_force': bf_perf.get('recall', 0),
            'eigenfaces': ef_perf.get('recall', 0),
            'difference': ef_perf.get('recall', 0) - bf_perf.get('recall', 0)
        },
        'f1_score': {
            'brute_force': bf_perf.get('f1_score', 0),
            'eigenfaces': ef_perf.get('f1_score', 0),
            'difference': ef_perf.get('f1_score', 0) - bf_perf.get('f1_score', 0)
        },
        'execution_time': {
            'brute_force': bf_time,
            'eigenfaces': ef_time,
            'ratio': ef_time / bf_time if bf_time > 0 else float('inf')
        }
    }
    
    # Determine overall winner based on F1-score
    if bf_perf.get('f1_score', 0) > ef_perf.get('f1_score', 0):
        comparison['winner'] = 'brute_force'
    elif ef_perf.get('f1_score', 0) > bf_perf.get('f1_score', 0):
        comparison['winner'] = 'eigenfaces'
    else:
        comparison['winner'] = 'tie'
    
    return comparison

def format_execution_time(seconds: float) -> str:
    """
    Format execution time in a human-readable format.
    
    Args:
        seconds: Execution time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1_000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes} min {seconds:.2f} s"

def create_performance_summary(results: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    """
    Create a summary of performance results for display.
    
    Args:
        results: Results from authentication method
        method_name: Name of the authentication method
        
    Returns:
        Dict[str, Any]: Summary of performance results
    """
    if not results or 'performance' not in results:
        return {
            'method': method_name,
            'accuracy': 'N/A',
            'precision': 'N/A',
            'recall': 'N/A',
            'f1_score': 'N/A',
            'execution_time': 'N/A'
        }
    
    perf = results['performance']
    
    return {
        'method': method_name,
        'accuracy': f"{perf.get('accuracy', 0):.2%}",
        'precision': f"{perf.get('precision', 0):.2%}",
        'recall': f"{perf.get('recall', 0):.2%}",
        'f1_score': f"{perf.get('f1_score', 0):.2%}",
        'execution_time': format_execution_time(results.get('execution_time', 0))
    } 