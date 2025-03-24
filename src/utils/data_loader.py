"""
Module de chargement et prétraitement des données d'images faciales.

Ce module fournit des fonctions pour charger les images faciales à partir des 
datasets fournis et les organiser en ensembles gallery et probes pour l'évaluation 
du système d'authentification.
"""

import os
import numpy as np
import random
from typing import Dict, List, Tuple
import matplotlib.image as mpimg
from PIL import Image


class FaceDataset:
    """
    Classe pour gérer les datasets d'images faciales.
    
    Attributes:
        dataset_path (str): Chemin vers le dataset
        gallery (List[np.ndarray]): Images de référence
        probes (List[np.ndarray]): Images de test
        gallery_identities (List[str]): Identités correspondant aux images gallery
        probe_identities (List[str]): Identités correspondant aux images probe
        ground_truth (List[bool]): Vérité terrain pour les probes
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialise un objet FaceDataset.
        
        Args:
            dataset_path (str): Chemin vers le dataset
        """
        self.dataset_path = dataset_path
        self.gallery = []
        self.probes = []
        self.gallery_identities = []
        self.probe_identities = []
        self.ground_truth = []
        
    def load_data(self, num_enrolled_per_identity: int = 3, num_probe_enrolled: int = 100, 
                  num_probe_not_enrolled: int = 100) -> None:
        """
        Charge les données et les divise en gallery et probes.
        
        Args:
            num_enrolled_per_identity (int): Nombre d'images par identité à inclure dans gallery
            num_probe_enrolled (int): Nombre de probes d'identités enregistrées
            num_probe_not_enrolled (int): Nombre de probes d'identités non enregistrées
        """
        print(f"Chargement des données depuis {self.dataset_path}...")
        
        # Collecte de toutes les images et identités
        image_dir = os.path.join(self.dataset_path, "images")
        all_images = []
        all_identities = []
        
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg'):
                # Parse du nom de fichier pour extraire l'identité et l'ID de l'image
                name_parts = os.path.splitext(filename)[0].split('.')
                identity = name_parts[0]
                
                # Chargement de l'image
                img_path = os.path.join(image_dir, filename)
                image = np.array(mpimg.imread(img_path))
                
                all_images.append(image)
                all_identities.append(identity)
        
        # Organisation des images par identité
        identity_to_images = {}
        for idx, identity in enumerate(all_identities):
            if identity not in identity_to_images:
                identity_to_images[identity] = []
            identity_to_images[identity].append((idx, all_images[idx]))
        
        # Division en identités enrollées et non-enrollées
        enrolled_identities = []
        not_enrolled_identities = []
        
        # Sélection aléatoire d'identités à enrôler
        all_unique_identities = list(identity_to_images.keys())
        random.shuffle(all_unique_identities)
        
        # Vérifier que nous avons assez d'identités avec suffisamment d'images
        valid_enrolled_identities = []
        for identity in all_unique_identities:
            if len(identity_to_images[identity]) >= num_enrolled_per_identity + 1:
                valid_enrolled_identities.append(identity)
                if len(valid_enrolled_identities) >= num_probe_enrolled:
                    break
        
        # Sélection des identités à enrôler et non-enrôlées
        enrolled_identities = valid_enrolled_identities[:num_probe_enrolled]
        not_enrolled_identities = [id for id in all_unique_identities 
                                 if id not in enrolled_identities][:num_probe_not_enrolled]
        
        # Constitution de la gallery (images de référence)
        for identity in enrolled_identities:
            images = identity_to_images[identity]
            selected_for_gallery = images[:num_enrolled_per_identity]
            
            for idx, img in selected_for_gallery:
                self.gallery.append(img)
                self.gallery_identities.append(identity)
        
        # Constitution des probes (images de test) - utilisateurs enregistrés
        for identity in enrolled_identities:
            images = identity_to_images[identity]
            if len(images) > num_enrolled_per_identity:
                # Prendre une image différente de celles dans la gallery
                probe_img = images[num_enrolled_per_identity][1]
                self.probes.append(probe_img)
                self.probe_identities.append(identity)
                self.ground_truth.append(True)  # Devrait être authentifié
        
        # Constitution des probes - utilisateurs non enregistrés
        for identity in not_enrolled_identities:
            images = identity_to_images[identity]
            if images:
                probe_img = images[0][1]
                self.probes.append(probe_img)
                self.probe_identities.append(identity)
                self.ground_truth.append(False)  # Ne devrait pas être authentifié
                
                if len(self.probes) >= num_probe_enrolled + num_probe_not_enrolled:
                    break
        
        print(f"Données chargées: {len(self.gallery)} images gallery, {len(self.probes)} images probe")
        print(f"dont {sum(self.ground_truth)} utilisateurs enregistrés et {len(self.ground_truth) - sum(self.ground_truth)} non enregistrés")

    def preprocess_images(self, flatten: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prétraite les images et les convertit au format requis.
        
        Args:
            flatten (bool): Si True, aplatit les images en vecteurs 1D
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Images gallery et probe prétraitées
        """
        gallery_processed = np.array(self.gallery, dtype=np.float32) / 255.0
        probes_processed = np.array(self.probes, dtype=np.float32) / 255.0
        
        if flatten:
            gallery_processed = gallery_processed.reshape(gallery_processed.shape[0], -1)
            probes_processed = probes_processed.reshape(probes_processed.shape[0], -1)
            
        return gallery_processed, probes_processed


def load_dataset(dataset_num: int, num_enrolled_per_identity: int = 3, 
                num_probe_enrolled: int = 100, num_probe_not_enrolled: int = 100) -> FaceDataset:
    """
    Fonction pratique pour charger un dataset spécifique.
    
    Args:
        dataset_num (int): Numéro du dataset à charger (1 ou 2)
        num_enrolled_per_identity (int): Nombre d'images par identité à inclure dans gallery
        num_probe_enrolled (int): Nombre de probes d'identités enregistrées
        num_probe_not_enrolled (int): Nombre de probes d'identités non enregistrées
        
    Returns:
        FaceDataset: Dataset chargé
    
    Raises:
        ValueError: Si dataset_num n'est pas 1 ou 2
    """
    if dataset_num not in [1, 2]:
        raise ValueError("Le numéro de dataset doit être 1 ou 2")
    
    dataset_path = os.path.join("data", f"dataset{dataset_num}")
    dataset = FaceDataset(dataset_path)
    dataset.load_data(num_enrolled_per_identity, num_probe_enrolled, num_probe_not_enrolled)
    
    return dataset 