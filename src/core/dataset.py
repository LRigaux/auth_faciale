"""
Module for dataset management in facial authentication.

This module provides classes and functions for loading and manipulating facial datasets.
"""

import os
import numpy as np
import random
import time
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from PIL import Image

class FaceDataset:
    """
    Unified class for facial image datasets management.
    
    Attributes:
        gallery (np.ndarray): Reference images (one per person)
        probes (np.ndarray): Test images
        gallery_ids (List): Identifiers of people in the gallery
        probe_ids (List): Identifiers of people in the test images
        ground_truth (List[bool]): Ground truth for test images (True if the person is in the gallery)
        dataset_path (str, optional): Path to the original dataset
    """
    
    def __init__(self, gallery: np.ndarray = None, probes: np.ndarray = None, 
                 gallery_ids: List = None, probe_ids: List = None, 
                 ground_truth: List[bool] = None, dataset_path: str = None):
        """
        Initialize a facial image dataset.
        
        Args:
            gallery: Reference images (can be empty at initialization)
            probes: Test images (can be empty at initialization)
            gallery_ids: Identifiers of people in the gallery
            probe_ids: Identifiers of people in the test images
            ground_truth: Ground truth for test images
            dataset_path: Path to the original dataset (optional)
        """
        self.gallery = gallery if gallery is not None else np.array([])
        self.probes = probes if probes is not None else np.array([])
        self.gallery_ids = gallery_ids if gallery_ids is not None else []
        self.probe_ids = probe_ids if probe_ids is not None else []
        self.ground_truth = ground_truth if ground_truth is not None else []
        self.dataset_path = dataset_path
        
    def validate(self):
        """Verify dimensions after initialization"""
        if self.gallery.size > 0 and self.gallery.ndim != 3:
            raise ValueError(f"Gallery must be a 3D array (n_samples, height, width), not {self.gallery.shape}")
        if self.probes.size > 0 and self.probes.ndim != 3:
            raise ValueError(f"Probes must be a 3D array (n_samples, height, width), not {self.probes.shape}")
        if len(self.gallery_ids) > 0 and len(self.gallery_ids) != len(self.gallery):
            raise ValueError(f"Number of gallery IDs ({len(self.gallery_ids)}) does not match number of images ({len(self.gallery)})")
        if len(self.ground_truth) > 0 and len(self.ground_truth) != len(self.probes):
            raise ValueError(f"Number of ground truths ({len(self.ground_truth)}) does not match number of probes ({len(self.probes)})")
        if len(self.probe_ids) > 0 and len(self.probe_ids) != len(self.probes):
            raise ValueError(f"Number of probe IDs ({len(self.probe_ids)}) does not match number of probes ({len(self.probes)})")
    
    @property
    def image_shape(self) -> Tuple[int, int]:
        """Return image dimensions (height, width)"""
        if self.gallery.size > 0:
            return self.gallery.shape[1:3]
        elif self.probes.size > 0:
            return self.probes.shape[1:3]
        return (0, 0)
    
    @property
    def n_gallery(self) -> int:
        """Return number of images in the gallery"""
        return len(self.gallery)
    
    @property
    def n_probes(self) -> int:
        """Return number of test images"""
        return len(self.probes)
    
    @property
    def n_enrolled_probes(self) -> int:
        """Return number of test images of enrolled users"""
        return sum(self.ground_truth)
    
    @property
    def n_non_enrolled_probes(self) -> int:
        """Return number of test images of non-enrolled users"""
        return sum(not gt for gt in self.ground_truth)
    
    def get_enrolled_probes(self) -> np.ndarray:
        """Return test images of enrolled users"""
        if len(self.probes) == 0:
            return np.array([])
        return self.probes[np.array(self.ground_truth)]
    
    def get_non_enrolled_probes(self) -> np.ndarray:
        """Return test images of non-enrolled users"""
        if len(self.probes) == 0:
            return np.array([])
        return self.probes[np.logical_not(np.array(self.ground_truth))]
    
    def get_enrolled_indices(self) -> List[int]:
        """Return indices of enrolled users probes"""
        return [i for i, gt in enumerate(self.ground_truth) if gt]
    
    def get_non_enrolled_indices(self) -> List[int]:
        """Return indices of non-enrolled users probes"""
        return [i for i, gt in enumerate(self.ground_truth) if not gt]
    
    def get_gallery_person(self, person_id: Any) -> Optional[np.ndarray]:
        """
        Get gallery image for a given person.
        
        Args:
            person_id: Person ID
            
        Returns:
            ndarray: Gallery image or None if not found
        """
        try:
            # Look for ID in gallery_ids
            if isinstance(self.gallery_ids, list):
                indices = [i for i, gid in enumerate(self.gallery_ids) if gid == person_id]
                if indices:
                    return self.gallery[indices[0]]
            
            # Try direct index if person_id is an integer
            if isinstance(person_id, int) and 0 <= person_id < len(self.gallery):
                return self.gallery[person_id]
                
            # Return empty if not found
            return None
        except Exception as e:
            print(f"Error retrieving a person from the gallery: {e}")
            return None
    
    def get_gallery_images_for_identity(self, identity_id: Any) -> List[np.ndarray]:
        """
        Get all gallery images for a given identity.
        
        Args:
            identity_id: Identity ID to search for
            
        Returns:
            List[np.ndarray]: List of gallery images for this identity
        """
        images = []
        
        # Find all occurrences of identity_id in gallery_ids
        if isinstance(self.gallery_ids, list):
            indices = [i for i, gid in enumerate(self.gallery_ids) if gid == identity_id]
            for idx in indices:
                images.append(self.gallery[idx])
                
        return images
    
    def get_probe_images_for_identity(self, identity_id: Any) -> List[np.ndarray]:
        """
        Get all probe images for a given identity.
        
        Args:
            identity_id: Identity ID to search for
            
        Returns:
            List[np.ndarray]: List of probe images for this identity
        """
        images = []
        
        # Find all occurrences of identity_id in probe_ids
        if isinstance(self.probe_ids, list):
            indices = [i for i, pid in enumerate(self.probe_ids) if pid == identity_id]
            for idx in indices:
                images.append(self.probes[idx])
                
        return images
    
    def get_random_probe(self, enrolled: bool = None) -> Tuple[np.ndarray, Any, bool]:
        """
        Return a random test image.
        
        Args:
            enrolled: If True, return an image of an enrolled user,
                     if False, return an image of a non-enrolled user,
                     if None, return any image
        
        Returns:
            Tuple: (image, person ID, enrollment status)
        """
        if len(self.probes) == 0:
            raise ValueError("No probe available")
            
        if enrolled is None:
            idx = random.randint(0, len(self.probes) - 1)
        elif enrolled:
            enrolled_indices = [i for i, gt in enumerate(self.ground_truth) if gt]
            if not enrolled_indices:
                raise ValueError("No enrolled user in the dataset")
            idx = random.choice(enrolled_indices)
        else:
            non_enrolled_indices = [i for i, gt in enumerate(self.ground_truth) if not gt]
            if not non_enrolled_indices:
                raise ValueError("No non-enrolled user in the dataset")
            idx = random.choice(non_enrolled_indices)
        
        return self.probes[idx], self.probe_ids[idx], self.ground_truth[idx]
    
    def preprocess_images(self, method: str = 'normalize', flatten: bool = True,
                           progress_callback: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess dataset images.
        
        Args:
            method: Preprocessing method ('normalize', 'standardize', 'histogram', 'clahe')
            flatten: If True, flatten images to get vectors
            progress_callback: Callback function for progress
            
        Returns:
            Tuple: (gallery_processed, probes_processed)
        """
        start_time = time.time()
        
        if progress_callback:
            progress_callback(5, f"Preprocessing images with method '{method}'...")
        
        # Preprocessing function according to method
        def preprocess_batch(images, method, flatten):
            if len(images) == 0:
                return np.array([])
                
            if method == 'normalize':
                # Simple normalization [0, 1]
                processed = images.astype(np.float32) / 255.0
            elif method == 'histogram':
                # Histogram equalization
                processed = np.array([cv2.equalizeHist(img.astype(np.uint8)) for img in images])
                processed = processed.astype(np.float32) / 255.0
            elif method == 'clahe':
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = np.array([clahe.apply(img.astype(np.uint8)) for img in images])
                processed = processed.astype(np.float32) / 255.0
            else:
                # Default: normalization
                processed = images.astype(np.float32) / 255.0
                
            # Flatten if requested
            if flatten and processed.size > 0:
                processed = processed.reshape(processed.shape[0], -1)
                
            return processed
        
        # Preprocess gallery
        if progress_callback:
            progress_callback(20, "Preprocessing gallery...")
        gallery_processed = preprocess_batch(self.gallery, method, flatten)
        
        # Preprocess probes
        if progress_callback:
            progress_callback(50, "Preprocessing probes...")
        probes_processed = preprocess_batch(self.probes, method, flatten)
        
        # Standardization (applies after other preprocessings)
        if method == 'standardize' and gallery_processed.size > 0 and probes_processed.size > 0:
            if progress_callback:
                progress_callback(70, "Applying standardization...")
                
            # Combine all images for fitting
            all_images = np.vstack([gallery_processed, probes_processed])
            scaler = StandardScaler()
            scaler.fit(all_images)
            
            # Apply transformation
            gallery_processed = scaler.transform(gallery_processed)
            probes_processed = scaler.transform(probes_processed)
        
        if progress_callback:
            total_time = time.time() - start_time
            progress_callback(100, f"Preprocessing completed in {total_time:.2f} seconds")
        
        return gallery_processed, probes_processed


def load_dataset(dataset_num: int, progress_callback: Optional[Callable] = None) -> FaceDataset:
    """
    Load a specific dataset.
    
    Args:
        dataset_num: Dataset number to load (1 or 2)
        progress_callback: Callback function for progress
    
    Returns:
        FaceDataset: Loaded dataset
    """
    start_time = time.time()
    
    if progress_callback:
        progress_callback(5, "Initializing data loading...")
    
    # Définir le chemin du dataset
    dataset_path = f"data/dataset{dataset_num}"
    
    if not os.path.exists(dataset_path):
        # Fallback aux datasets synthétiques si les vrais datasets ne sont pas disponibles
        if dataset_num == 1:
            # Synthetic dataset - small size for quick tests
            return create_synthetic_dataset(progress_callback=progress_callback)
        elif dataset_num == 2:
            # Larger synthetic dataset
            return create_synthetic_dataset(n_subjects=30, n_probes_per_subject=4, img_size=64, 
                                           progress_callback=progress_callback)
        else:
            raise ValueError(f"Dataset {dataset_num} not recognized")
    
    if progress_callback:
        progress_callback(10, f"Loading dataset {dataset_num} from {dataset_path}...")
    
    # Liste tous les fichiers d'images
    image_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        raise ValueError(f"No images found in {dataset_path}")
    
    if progress_callback:
        progress_callback(20, f"Found {len(image_files)} images. Processing...")
    
    # Extraire les identités à partir des noms de fichiers
    # Format du nom de fichier: X.Y.jpg où X est l'identité
    identities = {}
    for img_path in image_files:
        filename = os.path.basename(img_path)
        parts = filename.split('.')
        if len(parts) >= 2:
            identity = parts[0]
            if identity not in identities:
                identities[identity] = []
            identities[identity].append(img_path)
    
    # Trier les identités par nom pour avoir un ordre cohérent
    sorted_identities = sorted(identities.keys())
    
    if progress_callback:
        progress_callback(30, f"Found {len(sorted_identities)} unique identities. Creating gallery and probes...")
    
    # Créer la gallery (une image par identité)
    gallery = []
    gallery_ids = []
    probes = []
    probe_ids = []
    ground_truth = []
    
    # Taille de l'image à utiliser pour le redimensionnement
    target_size = (150, 150)  # Taille originale selon le README
    
    # Pour chaque identité, première image = gallery, reste = probes
    for i, identity in enumerate(sorted_identities):
        if progress_callback and i % 20 == 0:
            percent = 30 + (i / len(sorted_identities)) * 60
            progress_callback(int(percent), f"Processing identity {i+1}/{len(sorted_identities)}...")
        
        identity_images = identities[identity]
        
        if not identity_images:
            continue
        
        # Premier fichier pour gallery
        gallery_img_path = identity_images[0]
        try:
            img = Image.open(gallery_img_path).convert('L')  # Conversion en niveaux de gris
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0  # Normalisation
            gallery.append(img_array)
            gallery_ids.append(identity)
        except Exception as e:
            print(f"Error loading gallery image {gallery_img_path}: {e}")
            continue
        
        # Reste pour probes (toutes les images restantes au lieu de seulement 4)
        for probe_img_path in identity_images[1:]:
            try:
                img = Image.open(probe_img_path).convert('L')
                img = img.resize(target_size)
                img_array = np.array(img) / 255.0
                probes.append(img_array)
                probe_ids.append(identity)
                ground_truth.append(True)  # Les probes de la même identité sont enrolled
            except Exception as e:
                print(f"Error loading probe image {probe_img_path}: {e}")
    
    # Définir un nombre raisonnable d'identités non-enrolled (environ 25% des identités totales)
    # pour ne pas surcharger le dataset mais avoir suffisamment de cas négatifs
    non_enrolled_identities = int(len(sorted_identities) * 0.25)
    
    # Ajouter les probes non-enrolled
    if progress_callback:
        progress_callback(80, f"Adding non-enrolled probes...")
    
    non_enrolled_count = 0
    identity_idx = 0
    
    while non_enrolled_count < non_enrolled_identities and identity_idx < len(sorted_identities):
        identity = sorted_identities[identity_idx]
        
        # Si l'identité n'est pas dans la gallery (ou vers la fin de la liste pour augmenter la variété)
        if identity not in gallery_ids or identity_idx >= len(gallery_ids) * 3 // 4:
            identity_images = identities[identity]
            
            if identity_images:
                # Ajouter toutes les images de cette identité comme non-enrolled
                for probe_img_path in identity_images:
                    try:
                        img = Image.open(probe_img_path).convert('L')
                        img = img.resize(target_size)
                        img_array = np.array(img) / 255.0
                        probes.append(img_array)
                        probe_ids.append(identity)
                        ground_truth.append(False)  # Non-enrolled
                    except Exception as e:
                        print(f"Error loading non-enrolled probe {probe_img_path}: {e}")
                
                non_enrolled_count += 1
        
        identity_idx += 1
    
    # Convertir en arrays numpy
    gallery = np.array(gallery)
    probes = np.array(probes)
    
    if progress_callback:
        total_time = time.time() - start_time
        progress_callback(90, f"Creating dataset object with {len(gallery)} gallery images and {len(probes)} probes...")
    
    # Créer et retourner l'objet dataset
    dataset = FaceDataset(gallery, probes, gallery_ids, probe_ids, ground_truth, dataset_path)
    
    if progress_callback:
        total_time = time.time() - start_time
        progress_callback(100, f"Dataset loaded in {total_time:.2f} seconds")
    
    return dataset

def create_synthetic_dataset(n_subjects: int = 20, n_probes_per_subject: int = 3,
                            img_size: int = 48, n_imposters: int = 5,
                            progress_callback: Optional[Callable] = None) -> FaceDataset:
    """
    Create a synthetic dataset for testing face authentication methods.
    
    Args:
        n_subjects: Number of enrolled subjects
        n_probes_per_subject: Number of probe images per subject
        img_size: Size of the synthetic images (img_size x img_size)
        n_imposters: Number of imposters (non-enrolled subjects)
        progress_callback: Callback function for progress
        
    Returns:
        FaceDataset: Synthetic dataset with gallery and probe images
    """
    if progress_callback:
        progress_callback(5, "Generating synthetic dataset...")
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Generate gallery images (one per subject)
    if progress_callback:
        progress_callback(20, f"Generating gallery images for {n_subjects} subjects...")
    
    gallery = []
    gallery_ids = []
    
    for subject_id in range(n_subjects):
        # Create a base pattern for the subject
        base_pattern = np.random.rand(img_size, img_size) * 0.3
        
        # Add some structure (simulating facial features)
        # Eyes
        eye_x1, eye_y1 = img_size//3, img_size//3
        eye_x2, eye_y2 = 2*img_size//3, img_size//3
        eye_size = max(2, img_size//10)
        
        # Add eyes (darker regions)
        base_pattern[eye_y1-eye_size//2:eye_y1+eye_size//2, 
                    eye_x1-eye_size//2:eye_x1+eye_size//2] += 0.5
        base_pattern[eye_y2-eye_size//2:eye_y2+eye_size//2, 
                    eye_x2-eye_size//2:eye_x2+eye_size//2] += 0.5
        
        # Add mouth (darker region)
        mouth_x, mouth_y = img_size//2, 2*img_size//3
        mouth_width, mouth_height = img_size//2, img_size//6
        base_pattern[mouth_y-mouth_height//2:mouth_y+mouth_height//2, 
                    mouth_x-mouth_width//2:mouth_x+mouth_width//2] += 0.4
        
        # Add some noise and normalize
        base_pattern += np.random.rand(img_size, img_size) * 0.1
        base_pattern = np.clip(base_pattern, 0, 1)
        
        # Convert to uint8
        base_pattern = (base_pattern * 255).astype(np.uint8)
        
        gallery.append(base_pattern)
        gallery_ids.append(subject_id)
    
    # Generate probe images with variations of gallery images
    if progress_callback:
        progress_callback(50, "Generating probe images...")
    
    probes = []
    probe_ids = []
    ground_truth = []
    
    # Enrolled probes (variations of gallery subjects)
    for subject_id in range(n_subjects):
        base_image = gallery[subject_id].copy()
        
        for _ in range(n_probes_per_subject):
            # Create a variation of the base image
            variation = base_image.copy()
            
            # Add random noise
            noise = np.random.normal(0, 15, (img_size, img_size))
            variation = np.clip(variation + noise, 0, 255).astype(np.uint8)
            
            # Apply random brightness changes
            brightness_factor = np.random.uniform(0.8, 1.2)
            variation = np.clip(variation * brightness_factor, 0, 255).astype(np.uint8)
            
            # Apply slight geometric distortion
            rows, cols = variation.shape
            M = np.float32([[1, np.random.uniform(-0.1, 0.1), 0],
                           [np.random.uniform(-0.1, 0.1), 1, 0]])
            variation = cv2.warpAffine(variation, M, (cols, rows))
            
            probes.append(variation)
            probe_ids.append(subject_id)
            ground_truth.append(True)  # Enrolled user
    
    # Imposters (non-enrolled subjects)
    if progress_callback:
        progress_callback(75, f"Generating {n_imposters} non-enrolled subjects...")
    
    for imposter_id in range(n_subjects, n_subjects + n_imposters):
        # Create a distinctly different pattern
        imposter_pattern = np.random.rand(img_size, img_size) * 0.5
        
        # Add some structure but in different positions
        # Eyes
        eye_x1, eye_y1 = img_size//4, img_size//4
        eye_x2, eye_y2 = 3*img_size//4, img_size//4
        eye_size = max(2, img_size//12)
        
        imposter_pattern[eye_y1-eye_size//2:eye_y1+eye_size//2, 
                        eye_x1-eye_size//2:eye_x1+eye_size//2] += 0.4
        imposter_pattern[eye_y2-eye_size//2:eye_y2+eye_size//2, 
                        eye_x2-eye_size//2:eye_x2+eye_size//2] += 0.4
        
        # Add mouth
        mouth_x, mouth_y = img_size//2, 3*img_size//4
        mouth_width, mouth_height = img_size//3, img_size//8
        imposter_pattern[mouth_y-mouth_height//2:mouth_y+mouth_height//2, 
                        mouth_x-mouth_width//2:mouth_x+mouth_width//2] += 0.3
        
        # Add some noise and normalize
        imposter_pattern += np.random.rand(img_size, img_size) * 0.15
        imposter_pattern = np.clip(imposter_pattern, 0, 1)
        
        # Convert to uint8
        imposter_pattern = (imposter_pattern * 255).astype(np.uint8)
        
        # Add variations
        for _ in range(n_probes_per_subject // 2):
            variation = imposter_pattern.copy()
            
            # Add random noise
            noise = np.random.normal(0, 10, (img_size, img_size))
            variation = np.clip(variation + noise, 0, 255).astype(np.uint8)
            
            # Apply random brightness changes
            brightness_factor = np.random.uniform(0.9, 1.1)
            variation = np.clip(variation * brightness_factor, 0, 255).astype(np.uint8)
            
            probes.append(variation)
            probe_ids.append(imposter_id)
            ground_truth.append(False)  # Non-enrolled user
    
    # Convert to numpy arrays
    gallery_np = np.array(gallery)
    probes_np = np.array(probes)
    
    # Create and return the dataset
    dataset = FaceDataset(
        gallery=gallery_np,
        probes=probes_np,
        gallery_ids=gallery_ids,
        probe_ids=probe_ids,
        ground_truth=ground_truth,
        dataset_path="synthetic"
    )
    
    if progress_callback:
        progress_callback(100, f"Synthetic dataset created with {n_subjects} enrolled and {n_imposters} non-enrolled subjects.")
    
    return dataset
