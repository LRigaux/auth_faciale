"""
Fonctions utilitaires pour l'authentification faciale.

Ce module fournit diverses fonctions de chargement et de prétraitement des données,
ainsi que des fonctions d'évaluation des méthodes d'authentification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Callable
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import cv2
from sklearn.preprocessing import StandardScaler

@dataclass
class Dataset:
    """
    Classe pour stocker les données d'authentification.
    
    Attributes:
        gallery: Images de référence (une par personne)
        probes: Images de test
        gallery_ids: Identifiants des personnes dans la galerie
        ground_truth: Vérité terrain pour les images de test (True si la personne est dans la galerie)
        probe_ids: Identifiants des personnes dans les images de test
    """
    gallery: np.ndarray
    probes: np.ndarray
    gallery_ids: List[int]
    ground_truth: List[bool]
    probe_ids: List[int]
    
    def __post_init__(self):
        """Vérification des dimensions après initialisation"""
        if self.gallery.ndim != 3:
            raise ValueError(f"La galerie doit être un tableau 3D (n_samples, height, width), pas {self.gallery.shape}")
        if self.probes.ndim != 3:
            raise ValueError(f"Les probes doivent être un tableau 3D (n_samples, height, width), pas {self.probes.shape}")
        if len(self.gallery_ids) != len(self.gallery):
            raise ValueError(f"Le nombre d'IDs de galerie ({len(self.gallery_ids)}) ne correspond pas au nombre d'images ({len(self.gallery)})")
        if len(self.ground_truth) != len(self.probes):
            raise ValueError(f"Le nombre de ground truths ({len(self.ground_truth)}) ne correspond pas au nombre de probes ({len(self.probes)})")
        if len(self.probe_ids) != len(self.probes):
            raise ValueError(f"Le nombre d'IDs de probe ({len(self.probe_ids)}) ne correspond pas au nombre de probes ({len(self.probes)})")
    
    @property
    def image_shape(self) -> Tuple[int, int]:
        """Retourne les dimensions des images (hauteur, largeur)"""
        return self.gallery.shape[1:3]
    
    @property
    def n_gallery(self) -> int:
        """Retourne le nombre d'images dans la galerie"""
        return len(self.gallery)
    
    @property
    def n_probes(self) -> int:
        """Retourne le nombre d'images de test"""
        return len(self.probes)
    
    @property
    def n_enrolled_probes(self) -> int:
        """Retourne le nombre d'images de test d'utilisateurs enregistrés"""
        return sum(self.ground_truth)
    
    @property
    def n_non_enrolled_probes(self) -> int:
        """Retourne le nombre d'images de test d'utilisateurs non enregistrés"""
        return sum(not gt for gt in self.ground_truth)
    
    def get_enrolled_probes(self) -> np.ndarray:
        """Retourne les images de test des utilisateurs enregistrés"""
        return self.probes[np.array(self.ground_truth)]
    
    def get_non_enrolled_probes(self) -> np.ndarray:
        """Retourne les images de test des utilisateurs non enregistrés"""
        return self.probes[np.logical_not(np.array(self.ground_truth))]
    
    def get_gallery_person(self, person_id: int) -> Optional[np.ndarray]:
        """Retourne l'image de référence d'une personne donnée"""
        indices = [i for i, gid in enumerate(self.gallery_ids) if gid == person_id]
        if not indices:
            return None
        return self.gallery[indices[0]]
    
    def get_probe_person(self, person_id: int) -> List[np.ndarray]:
        """Retourne toutes les images de test d'une personne donnée"""
        indices = [i for i, pid in enumerate(self.probe_ids) if pid == person_id]
        if not indices:
            return []
        return [self.probes[i] for i in indices]
    
    def get_random_probe(self, enrolled: bool = None) -> Tuple[np.ndarray, int, bool]:
        """
        Retourne une image de test aléatoire.
        
        Args:
            enrolled: Si True, retourne une image d'un utilisateur enregistré,
                     si False, retourne une image d'un utilisateur non enregistré,
                     si None, retourne n'importe quelle image
                     
        Returns:
            Tuple: (image, ID de la personne, statut d'enregistrement)
        """
        if enrolled is None:
            idx = np.random.randint(0, len(self.probes))
        elif enrolled:
            indices = [i for i, gt in enumerate(self.ground_truth) if gt]
            if not indices:
                raise ValueError("Aucun utilisateur enregistré dans le jeu de données")
            idx = np.random.choice(indices)
        else:
            indices = [i for i, gt in enumerate(self.ground_truth) if not gt]
            if not indices:
                raise ValueError("Aucun utilisateur non enregistré dans le jeu de données")
            idx = np.random.choice(indices)
            
        return self.probes[idx], self.probe_ids[idx], self.ground_truth[idx]


def load_dataset(dataset_num: int, progress_callback: Optional[Callable] = None) -> Dataset:
    """
    Charge un jeu de données spécifique.
    
    Args:
        dataset_num: Numéro du jeu de données à charger (1, 2 ou 3)
        progress_callback: Fonction de rappel pour la progression, 
                        prend un pourcentage et un message
    
    Returns:
        Dataset: Jeu de données chargé
    """
    # Garder la trace du temps d'exécution
    start_time = time.time()
    
    if progress_callback:
        progress_callback(5, "Initialisation du chargement des données...")
    
    if dataset_num == 1:
        # Jeu de données synthétique - petite taille pour tests rapides
        n_subjects = 10
        n_gallery = n_subjects
        n_probes = n_subjects * 3
        img_size = 32
        
        if progress_callback:
            progress_callback(20, "Génération de données synthétiques...")
        
        # Générer des données aléatoires
        np.random.seed(42)  # Pour la reproductibilité
        gallery = np.random.rand(n_gallery, img_size, img_size)
        probes = np.zeros((n_probes, img_size, img_size))
        gallery_ids = list(range(1, n_gallery + 1))
        probe_ids = []
        ground_truth = []
        
        if progress_callback:
            progress_callback(40, "Préparation des probes...")
        
        # Créer 2 probes authentiques (avec bruit) et 1 imposture pour chaque sujet
        for i in range(n_subjects):
            # 2 probes authentiques (ajout de bruit)
            for j in range(2):
                probe_idx = i * 3 + j
                probes[probe_idx] = gallery[i] + 0.1 * np.random.randn(img_size, img_size)
                probe_ids.append(gallery_ids[i])
                ground_truth.append(True)
            
            # 1 probe imposture
            probe_idx = i * 3 + 2
            probes[probe_idx] = np.random.rand(img_size, img_size)
            fake_id = n_gallery + i + 1
            probe_ids.append(fake_id)
            ground_truth.append(False)
        
        if progress_callback:
            progress_callback(90, "Finalisation du jeu de données synthétique...")
        
    elif dataset_num == 2:
        # Jeu de données moyen - visages Yale
        yale_dir = Path("datasets/yale_faces")
        
        if not yale_dir.exists():
            raise FileNotFoundError(f"Le répertoire {yale_dir} n'existe pas. Veuillez télécharger le jeu de données Yale.")
        
        if progress_callback:
            progress_callback(10, "Analyse du répertoire du jeu de données Yale...")
        
        # Lister tous les fichiers d'images
        image_files = list(yale_dir.glob("subject*.gif"))
        
        if not image_files:
            raise FileNotFoundError(f"Aucun fichier d'image trouvé dans {yale_dir}")
        
        # Identifier les sujets uniques
        subject_ids = set()
        for filepath in image_files:
            filename = filepath.name
            subject_id = int(filename.split(".")[0].replace("subject", ""))
            subject_ids.add(subject_id)
        
        subject_ids = sorted(subject_ids)
        n_subjects = len(subject_ids)
        
        if progress_callback:
            progress_callback(20, f"Chargement des images pour {n_subjects} sujets...")
        
        # Préparer les structures de données
        gallery = []
        gallery_ids = []
        probes = []
        probe_ids = []
        ground_truth = []
        
        # Charger les images avec OpenCV pour de meilleures performances
        for i, subject_id in enumerate(subject_ids):
            if progress_callback and i % 5 == 0:
                percent = 20 + (i / n_subjects) * 70
                progress_callback(percent, f"Traitement du sujet {subject_id}...")
            
            # Trouver toutes les images de ce sujet
            subject_files = list(yale_dir.glob(f"subject{subject_id}.*.gif"))
            
            if not subject_files:
                continue
                
            # Charger la première image comme référence
            gallery_file = subject_files[0]
            img = cv2.imread(str(gallery_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Redimensionner pour la cohérence
            img = cv2.resize(img, (64, 64))
            gallery.append(img)
            gallery_ids.append(subject_id)
            
            # Les autres images sont des probes
            for probe_file in subject_files[1:3]:  # Limiter à 2 probes par sujet pour équilibrer
                img = cv2.imread(str(probe_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (64, 64))
                probes.append(img)
                probe_ids.append(subject_id)
                ground_truth.append(True)
            
            # Ajouter quelques impostures
            for impostor_id in np.random.choice([sid for sid in subject_ids if sid != subject_id], size=2, replace=False):
                impostor_files = list(yale_dir.glob(f"subject{impostor_id}.*.gif"))
                if impostor_files:
                    probe_file = np.random.choice(impostor_files)
                    img = cv2.imread(str(probe_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (64, 64))
                    probes.append(img)
                    probe_ids.append(impostor_id)
                    ground_truth.append(False)
        
        # Convertir en arrays numpy
        gallery = np.array(gallery)
        probes = np.array(probes)
        
        if progress_callback:
            progress_callback(90, "Finalisation du jeu de données Yale...")
        
    elif dataset_num == 3:
        # Jeu de données plus grand - visages AT&T
        att_dir = Path("datasets/att_faces")
        
        if not att_dir.exists():
            raise FileNotFoundError(f"Le répertoire {att_dir} n'existe pas. Veuillez télécharger le jeu de données AT&T.")
        
        if progress_callback:
            progress_callback(10, "Analyse du répertoire du jeu de données AT&T...")
        
        # Lister tous les sous-répertoires (un par sujet)
        subject_dirs = [d for d in att_dir.iterdir() if d.is_dir()]
        
        if not subject_dirs:
            raise FileNotFoundError(f"Aucun sous-répertoire de sujet trouvé dans {att_dir}")
        
        # Préparer les structures de données
        gallery = []
        gallery_ids = []
        probes = []
        probe_ids = []
        ground_truth = []
        
        # Sélectionner aléatoirement 30 sujets pour la galerie (si disponible)
        n_subjects = min(30, len(subject_dirs))
        gallery_subjects = np.random.choice(subject_dirs, size=n_subjects, replace=False)
        
        # Chargement parallèle pour de meilleures performances
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for subject_dir in gallery_subjects:
                subject_id = int(subject_dir.name.replace("s", ""))
                futures.append(executor.submit(load_att_subject, subject_dir, subject_id))
            
            # Traiter les résultats au fur et à mesure qu'ils sont disponibles
            for i, future in enumerate(as_completed(futures)):
                if progress_callback:
                    percent = 20 + (i / n_subjects) * 70
                    progress_callback(percent, f"Traitement du sujet {i+1}/{n_subjects}...")
                
                subject_data = future.result()
                if subject_data:
                    subject_gallery, subject_probes, subject_id = subject_data
                    gallery.append(subject_gallery)
                    gallery_ids.append(subject_id)
                    
                    # Ajouter les probes authentiques
                    for probe in subject_probes:
                        probes.append(probe)
                        probe_ids.append(subject_id)
                        ground_truth.append(True)
            
            # Ajouter des impostures (sujets non présents dans la galerie)
            impostor_subjects = [d for d in subject_dirs if d not in gallery_subjects]
            n_impostors = min(10, len(impostor_subjects))
            
            if impostor_subjects:
                selected_impostors = np.random.choice(impostor_subjects, size=n_impostors, replace=False)
                impostor_futures = []
                
                for subject_dir in selected_impostors:
                    subject_id = int(subject_dir.name.replace("s", ""))
                    impostor_futures.append(executor.submit(load_att_impostor, subject_dir, subject_id))
                
                # Traiter les résultats au fur et à mesure qu'ils sont disponibles
                for future in as_completed(impostor_futures):
                    impostor_data = future.result()
                    if impostor_data:
                        impostor_probes, impostor_id = impostor_data
                        for probe in impostor_probes:
                            probes.append(probe)
                            probe_ids.append(impostor_id)
                            ground_truth.append(False)
        
        # Convertir en arrays numpy
        gallery = np.array(gallery)
        probes = np.array(probes)
        
        if progress_callback:
            progress_callback(90, "Finalisation du jeu de données AT&T...")
    
    else:
        raise ValueError(f"Jeu de données {dataset_num} non reconnu")
    
    # Mesurer le temps total
    loading_time = time.time() - start_time
    
    if progress_callback:
        progress_callback(100, f"Chargement terminé en {loading_time:.2f} secondes")
    
    # Retourner le jeu de données
    return Dataset(gallery, probes, gallery_ids, ground_truth, probe_ids)


def load_att_subject(subject_dir: Path, subject_id: int) -> Optional[Tuple[np.ndarray, List[np.ndarray], int]]:
    """
    Charge les images d'un sujet du jeu de données AT&T.
    
    Args:
        subject_dir: Répertoire contenant les images du sujet
        subject_id: Identifiant du sujet
    
    Returns:
        Tuple: (image de galerie, liste d'images de probe, ID du sujet) ou None en cas d'erreur
    """
    # Lister tous les fichiers d'images pour ce sujet
    image_files = sorted(list(subject_dir.glob("*.pgm")))
    
    if not image_files:
        return None
    
    # Charger la première image comme référence
    try:
        gallery_img = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
        if gallery_img is None:
            return None
        gallery_img = cv2.resize(gallery_img, (64, 64))
        
        # Charger quelques autres images comme probes
        probes = []
        for img_file in image_files[1:4]:  # Limiter à 3 probes par sujet
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                probes.append(img)
        
        if not probes:
            return None
            
        return gallery_img, probes, subject_id
    except Exception as e:
        print(f"Erreur lors du chargement des images du sujet {subject_id}: {e}")
        return None


def load_att_impostor(subject_dir: Path, subject_id: int) -> Optional[Tuple[List[np.ndarray], int]]:
    """
    Charge les images d'un imposteur du jeu de données AT&T.
    
    Args:
        subject_dir: Répertoire contenant les images de l'imposteur
        subject_id: Identifiant de l'imposteur
    
    Returns:
        Tuple: (liste d'images de probe, ID de l'imposteur) ou None en cas d'erreur
    """
    # Lister tous les fichiers d'images pour cet imposteur
    image_files = sorted(list(subject_dir.glob("*.pgm")))
    
    if not image_files:
        return None
    
    try:
        # Sélectionner quelques images aléatoires
        selected_files = np.random.choice(image_files, size=min(3, len(image_files)), replace=False)
        
        # Charger les images
        probes = []
        for img_file in selected_files:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                probes.append(img)
        
        if not probes:
            return None
            
        return probes, subject_id
    except Exception as e:
        print(f"Erreur lors du chargement des images de l'imposteur {subject_id}: {e}")
        return None


def preprocess_images(images: np.ndarray, method: str = 'normalize',
                     progress_callback: Optional[Callable] = None) -> np.ndarray:
    """
    Prétraite un ensemble d'images pour l'authentification.
    
    Args:
        images: Tableau d'images (n_samples, height, width)
        method: Méthode de prétraitement ('normalize', 'standardize', 'histogram', 'clahe')
        progress_callback: Fonction de rappel pour la progression
    
    Returns:
        np.ndarray: Images prétraitées aplaties (n_samples, height*width)
    """
    # Garder la trace du temps d'exécution
    start_time = time.time()
    
    if progress_callback:
        progress_callback(5, f"Prétraitement des images avec la méthode '{method}'...")
    
    # Convertir en float si ce n'est pas déjà le cas
    if images.dtype != np.float32 and images.dtype != np.float64:
        images = images.astype(np.float32)
    
    # Prétraitement parallèle
    n_samples = len(images)
    processed_images = np.zeros((n_samples, images.shape[1] * images.shape[2]), dtype=np.float32)
    
    if method == 'normalize':
        # Normalisation simple [0, 1]
        for i, img in enumerate(images):
            processed_images[i] = (img / 255.0).flatten()
            if progress_callback and i % 10 == 0:
                progress_callback(10 + (i / n_samples) * 80, f"Normalisation de l'image {i+1}/{n_samples}")
    
    elif method == 'standardize':
        # Standardisation (moyenne=0, variance=1)
        scaler = StandardScaler()
        
        if progress_callback:
            progress_callback(20, "Calcul des statistiques pour la standardisation...")
        
        # Préparer les données pour le scaler
        flattened = images.reshape(n_samples, -1)
        processed_images = scaler.fit_transform(flattened)
        
        if progress_callback:
            progress_callback(80, "Standardisation terminée")
    
    elif method == 'histogram':
        # Égalisation d'histogramme
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for i, img in enumerate(images):
                futures[executor.submit(process_histogram, img)] = i
            
            for i, future in enumerate(as_completed(futures)):
                idx = futures[future]
                processed_images[idx] = future.result()
                
                if progress_callback and i % 10 == 0:
                    progress_callback(10 + (i / n_samples) * 80, f"Égalisation d'histogramme {i+1}/{n_samples}")
    
    elif method == 'clahe':
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for i, img in enumerate(images):
                futures[executor.submit(process_clahe, img)] = i
            
            for i, future in enumerate(as_completed(futures)):
                idx = futures[future]
                processed_images[idx] = future.result()
                
                if progress_callback and i % 10 == 0:
                    progress_callback(10 + (i / n_samples) * 80, f"CLAHE {i+1}/{n_samples}")
    
    else:
        raise ValueError(f"Méthode de prétraitement '{method}' non reconnue")
    
    # Mesurer le temps total
    processing_time = time.time() - start_time
    
    if progress_callback:
        progress_callback(100, f"Prétraitement terminé en {processing_time:.2f} secondes")
    
    return processed_images


def process_histogram(img: np.ndarray) -> np.ndarray:
    """
    Applique l'égalisation d'histogramme à une image.
    
    Args:
        img: Image à traiter
        
    Returns:
        np.ndarray: Image prétraitée et aplatie
    """
    # Convertir en uint8 si nécessaire
    if img.dtype != np.uint8:
        img = (255 * img / img.max()).astype(np.uint8)
    
    # Appliquer l'égalisation d'histogramme
    img_eq = cv2.equalizeHist(img)
    
    # Normaliser et aplatir
    return (img_eq / 255.0).flatten()


def process_clahe(img: np.ndarray) -> np.ndarray:
    """
    Applique CLAHE à une image.
    
    Args:
        img: Image à traiter
        
    Returns:
        np.ndarray: Image prétraitée et aplatie
    """
    # Convertir en uint8 si nécessaire
    if img.dtype != np.uint8:
        img = (255 * img / img.max()).astype(np.uint8)
    
    # Créer et appliquer CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    # Normaliser et aplatir
    return (img_clahe / 255.0).flatten()


def evaluate_authentication_method(dataset: Dataset, 
                                  authenticate_func: Callable, 
                                  params: Dict[str, Any],
                                  preprocessed_gallery: Optional[np.ndarray] = None,
                                  preprocessed_probes: Optional[np.ndarray] = None,
                                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Évalue une méthode d'authentification sur un jeu de données.
    
    Args:
        dataset: Jeu de données à utiliser
        authenticate_func: Fonction d'authentification à évaluer
        params: Paramètres pour la fonction d'authentification
        preprocessed_gallery: Galerie prétraitée (facultatif)
        preprocessed_probes: Probes prétraitées (facultatif)
        progress_callback: Fonction de rappel pour la progression
    
    Returns:
        Dict: Dictionnaire contenant les résultats de l'évaluation
    """
    # Garder la trace du temps d'exécution
    start_time = time.time()
    
    if progress_callback:
        progress_callback(5, "Préparation de l'évaluation...")
    
    # Utiliser les données prétraitées si fournies
    gallery = preprocessed_gallery if preprocessed_gallery is not None else dataset.gallery
    probes = preprocessed_probes if preprocessed_probes is not None else dataset.probes
    
    # Récupérer les ID et la vérité terrain
    gallery_ids = dataset.gallery_ids
    probe_ids = dataset.probe_ids
    ground_truth = dataset.ground_truth
    
    # Structures pour stocker les résultats
    predictions = []
    authentication_times = []
    
    if progress_callback:
        progress_callback(10, "Évaluation des probes...")
    
    # Évaluer chaque probe
    for i, probe in enumerate(probes):
        if progress_callback and i % 10 == 0:
            progress_callback(10 + (i / len(probes)) * 80, f"Authentification de la probe {i+1}/{len(probes)}")
        
        # Mesurer le temps d'authentification
        auth_start = time.time()
        result = authenticate_func(probe, gallery, **params)
        auth_time = time.time() - auth_start
        
        # Stocker le résultat et le temps
        predictions.append(result)
        authentication_times.append(auth_time)
    
    # Calculer les métriques
    if progress_callback:
        progress_callback(90, "Calcul des métriques...")
    
    # Convertir les listes en arrays numpy pour faciliter les calculs
    predictions_array = np.array(predictions)
    ground_truth_array = np.array(ground_truth)
    
    # Calculer les métriques
    tp = np.sum(np.logical_and(predictions_array, ground_truth_array))
    fp = np.sum(np.logical_and(predictions_array, np.logical_not(ground_truth_array)))
    tn = np.sum(np.logical_and(np.logical_not(predictions_array), np.logical_not(ground_truth_array)))
    fn = np.sum(np.logical_and(np.logical_not(predictions_array), ground_truth_array))
    
    # Métriques dérivées
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Temps moyen d'authentification
    mean_auth_time = np.mean(authentication_times)
    min_auth_time = np.min(authentication_times)
    max_auth_time = np.max(authentication_times)
    
    # Temps total d'évaluation
    total_time = time.time() - start_time
    
    if progress_callback:
        progress_callback(100, f"Évaluation terminée en {total_time:.2f} secondes")
    
    # Retourner les résultats sous forme de dictionnaire
    return {
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        },
        'timing': {
            'mean_auth_time': mean_auth_time,
            'min_auth_time': min_auth_time,
            'max_auth_time': max_auth_time,
            'total_eval_time': total_time
        },
        'details': {
            'predictions': predictions_array.tolist(),
            'ground_truth': ground_truth_array.tolist(),
            'authentication_times': authentication_times
        }
    }


def compare_methods(dataset: Dataset, methods: List[Dict], 
                   preprocessed_gallery: Optional[np.ndarray] = None,
                   preprocessed_probes: Optional[np.ndarray] = None,
                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Compare plusieurs méthodes d'authentification sur un jeu de données.
    
    Args:
        dataset: Jeu de données à utiliser
        methods: Liste de dictionnaires décrivant les méthodes à comparer
                Chaque dictionnaire doit contenir:
                - 'name': Nom de la méthode
                - 'func': Fonction d'authentification
                - 'params': Paramètres pour la fonction
        preprocessed_gallery: Galerie prétraitée (facultatif)
        preprocessed_probes: Probes prétraitées (facultatif)
        progress_callback: Fonction de rappel pour la progression
    
    Returns:
        Dict: Dictionnaire contenant les résultats de la comparaison
    """
    # Garder la trace du temps d'exécution
    start_time = time.time()
    
    if progress_callback:
        progress_callback(5, "Préparation de la comparaison...")
    
    # Résultats pour chaque méthode
    results = {}
    
    # Évaluer chaque méthode
    for i, method in enumerate(methods):
        method_name = method['name']
        method_func = method['func']
        method_params = method['params']
        
        if progress_callback:
            progress_callback(5 + (i / len(methods)) * 90, f"Évaluation de la méthode '{method_name}'...")
        
        # Définir une fonction de rappel de progression pour cette méthode
        def method_progress_callback(percent, message):
            if progress_callback:
                # Mapper la progression de la méthode à une partie de notre progression globale
                mapped_percent = 5 + (i / len(methods)) * 90 + (percent / 100) * (90 / len(methods))
                progress_callback(mapped_percent, f"{method_name}: {message}")
        
        # Évaluer la méthode
        method_result = evaluate_authentication_method(
            dataset, method_func, method_params,
            preprocessed_gallery, preprocessed_probes,
            progress_callback=method_progress_callback
        )
        
        # Stocker les résultats
        results[method_name] = method_result
    
    # Temps total de comparaison
    total_time = time.time() - start_time
    
    if progress_callback:
        progress_callback(100, f"Comparaison terminée en {total_time:.2f} secondes")
    
    # Retourner les résultats sous forme de dictionnaire
    return {
        'results': results,
        'total_time': total_time
    }


def visualize_metrics(results: Dict[str, Any], metric_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Visualise des métriques de performance pour plusieurs méthodes.
    
    Args:
        results: Dictionnaire contenant les résultats de compare_methods
        metric_names: Liste des noms de métriques à visualiser (par défaut: toutes)
    
    Returns:
        plt.Figure: Figure matplotlib avec les graphiques
    """
    # Extraire les résultats
    method_results = results['results']
    method_names = list(method_results.keys())
    
    # Métriques à visualiser
    if metric_names is None:
        metric_names = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    
    # Préparer les données
    metrics_data = {}
    for metric in metric_names:
        metrics_data[metric] = [method_results[method]['metrics'][metric] for method in method_names]
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Position des barres
    x = np.arange(len(method_names))
    width = 0.15  # largeur des barres
    
    # Tracé des barres pour chaque métrique
    for i, metric in enumerate(metric_names):
        offset = (i - len(metric_names)/2 + 0.5) * width
        ax.bar(x + offset, metrics_data[metric], width, label=metric)
    
    # Embellir le graphique
    ax.set_xlabel('Méthode')
    ax.set_ylabel('Score')
    ax.set_title('Comparaison des méthodes d\'authentification')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_timing(results: Dict[str, Any]) -> plt.Figure:
    """
    Visualise les temps d'authentification pour plusieurs méthodes.
    
    Args:
        results: Dictionnaire contenant les résultats de compare_methods
    
    Returns:
        plt.Figure: Figure matplotlib avec les graphiques
    """
    # Extraire les résultats
    method_results = results['results']
    method_names = list(method_results.keys())
    
    # Préparer les données de temps
    mean_times = [method_results[method]['timing']['mean_auth_time'] * 1000 for method in method_names]  # en ms
    min_times = [method_results[method]['timing']['min_auth_time'] * 1000 for method in method_names]
    max_times = [method_results[method]['timing']['max_auth_time'] * 1000 for method in method_names]
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Position des barres
    x = np.arange(len(method_names))
    width = 0.25  # largeur des barres
    
    # Tracé des barres
    ax.bar(x - width, min_times, width, label='Temps minimal')
    ax.bar(x, mean_times, width, label='Temps moyen')
    ax.bar(x + width, max_times, width, label='Temps maximal')
    
    # Embellir le graphique
    ax.set_xlabel('Méthode')
    ax.set_ylabel('Temps (ms)')
    ax.set_title('Temps d\'authentification par méthode')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig 