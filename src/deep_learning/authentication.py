"""
Module d'authentification par deep learning.

Ce module implémente la méthode d'authentification par réseaux de neurones convolutifs (CNN),
qui apprend automatiquement des caractéristiques discriminantes pour l'authentification faciale.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda, concatenate
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from typing import Tuple, List, Dict, Optional, Any, Union
import matplotlib.pyplot as plt


def create_siamese_model(input_shape: Tuple[int, int, int]) -> Model:
    """
    Crée un modèle siamois pour l'authentification faciale.
    
    Args:
        input_shape (Tuple[int, int, int]): Forme des images d'entrée (hauteur, largeur, canaux)
        
    Returns:
        keras.Model: Modèle siamois
    """
    # Sous-réseau pour extraire les caractéristiques
    def create_base_network(input_shape):
        input_layer = Input(shape=input_shape)
        
        # Première couche de convolution
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(input_layer)
        x = MaxPooling2D((2, 2), name='pool1')(x)
        
        # Deuxième couche de convolution
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = MaxPooling2D((2, 2), name='pool2')(x)
        
        # Troisième couche de convolution
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
        x = MaxPooling2D((2, 2), name='pool3')(x)
        
        # Quatrième couche de convolution
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(x)
        x = MaxPooling2D((2, 2), name='pool4')(x)
        
        # Aplatir
        x = Flatten()(x)
        
        # Couche dense finale
        x = Dense(128, activation='relu', name='fc1')(x)
        
        return Model(inputs=input_layer, outputs=x)
    
    # Créer le sous-réseau
    base_network = create_base_network(input_shape)
    
    # Entrées pour les paires d'images
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    # Extraire les caractéristiques des deux images
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Calculer la distance euclidienne entre les caractéristiques
    distance = Lambda(lambda x: tf.keras.backend.sqrt(
        tf.keras.backend.sum(tf.keras.backend.square(x[0] - x[1]), axis=1, keepdims=True)
    ))([processed_a, processed_b])
    
    # Modèle siamois
    siamese_model = Model(inputs=[input_a, input_b], outputs=distance)
    
    return siamese_model, base_network


class CNNAuthenticator:
    """
    Classe pour l'authentification faciale par CNN.
    
    Attributes:
        input_shape (Tuple[int, int, int]): Forme des images d'entrée
        siamese_model (keras.Model): Modèle siamois
        feature_extractor (keras.Model): Sous-réseau d'extraction de caractéristiques
        gallery_features (np.ndarray): Caractéristiques extraites de la gallery
        threshold (float): Seuil de distance pour l'authentification
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (150, 150, 1)):
        """
        Initialise un authenticateur CNN.
        
        Args:
            input_shape (Tuple[int, int, int]): Forme des images d'entrée (hauteur, largeur, canaux)
        """
        self.input_shape = input_shape
        self.siamese_model, self.feature_extractor = create_siamese_model(input_shape)
        self.gallery_features = None
        self.threshold = 0.5
        
    def train(self, train_pairs: np.ndarray, train_labels: np.ndarray, 
              validation_pairs: np.ndarray, validation_labels: np.ndarray,
              batch_size: int = 32, epochs: int = 10) -> Dict[str, List[float]]:
        """
        Entraîne le modèle siamois.
        
        Args:
            train_pairs (np.ndarray): Paires d'images d'entraînement [gauche, droite]
            train_labels (np.ndarray): Étiquettes d'entraînement (1: même identité, 0: identités différentes)
            validation_pairs (np.ndarray): Paires d'images de validation
            validation_labels (np.ndarray): Étiquettes de validation
            batch_size (int): Taille du batch
            epochs (int): Nombre d'époques
            
        Returns:
            Dict[str, List[float]]: Historique d'entraînement
        """
        # Compiler le modèle
        self.siamese_model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        # Callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Entraîner le modèle
        history = self.siamese_model.fit(
            [train_pairs[:, 0], train_pairs[:, 1]], train_labels,
            validation_data=([validation_pairs[:, 0], validation_pairs[:, 1]], validation_labels),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[reduce_lr, early_stopping]
        )
        
        return history.history
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extrait les caractéristiques des images.
        
        Args:
            images (np.ndarray): Images à traiter
            
        Returns:
            np.ndarray: Caractéristiques extraites
        """
        # Prétraiter les images si nécessaire
        if images.ndim == 3:
            # Ajouter une dimension de canal si nécessaire
            images = images.reshape((-1, *self.input_shape))
        
        # Extraire les caractéristiques
        features = self.feature_extractor.predict(images)
        
        return features
    
    def compute_gallery_features(self, gallery: np.ndarray) -> None:
        """
        Calcule et stocke les caractéristiques de la gallery.
        
        Args:
            gallery (np.ndarray): Images de la gallery
        """
        self.gallery_features = self.extract_features(gallery)
        print(f"Caractéristiques de la gallery calculées: {self.gallery_features.shape}")
    
    def find_best_threshold(self, enrolled_probes: np.ndarray, 
                          non_enrolled_probes: np.ndarray) -> float:
        """
        Trouve le meilleur seuil de distance pour l'authentification.
        
        Args:
            enrolled_probes (np.ndarray): Images de test d'utilisateurs enregistrés
            non_enrolled_probes (np.ndarray): Images de test d'utilisateurs non enregistrés
            
        Returns:
            float: Seuil optimal
        """
        # Extraire les caractéristiques des probes
        enrolled_features = self.extract_features(enrolled_probes)
        non_enrolled_features = self.extract_features(non_enrolled_probes)
        
        # Calculer les distances minimales pour les utilisateurs enregistrés
        min_distances_enrolled = []
        for feat in enrolled_features:
            distances = np.sqrt(np.sum((self.gallery_features - feat)**2, axis=1))
            min_distances_enrolled.append(np.min(distances))
        
        # Calculer les distances minimales pour les utilisateurs non enregistrés
        min_distances_non_enrolled = []
        for feat in non_enrolled_features:
            distances = np.sqrt(np.sum((self.gallery_features - feat)**2, axis=1))
            min_distances_non_enrolled.append(np.min(distances))
        
        # Convertir en numpy arrays
        min_distances_enrolled = np.array(min_distances_enrolled)
        min_distances_non_enrolled = np.array(min_distances_non_enrolled)
        
        # Trouver le seuil optimal
        best_threshold = 0
        best_accuracy = 0
        
        # Tester différentes valeurs de seuil
        for percentile in range(10, 100, 5):
            threshold_candidate = np.percentile(min_distances_enrolled, percentile)
            
            # Calculer les performances avec ce seuil
            true_positives = np.sum(min_distances_enrolled <= threshold_candidate)
            false_positives = np.sum(min_distances_non_enrolled <= threshold_candidate)
            true_negatives = len(min_distances_non_enrolled) - false_positives
            false_negatives = len(min_distances_enrolled) - true_positives
            
            accuracy = (true_positives + true_negatives) / (len(min_distances_enrolled) + len(min_distances_non_enrolled))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold_candidate
        
        self.threshold = best_threshold
        print(f"Meilleur seuil: {self.threshold:.4f} avec une précision de {best_accuracy:.4f}")
        
        return self.threshold
    
    def save_model(self, filepath: str) -> None:
        """
        Sauvegarde le modèle.
        
        Args:
            filepath (str): Chemin où sauvegarder le modèle
        """
        self.feature_extractor.save(filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Charge un modèle pré-entraîné.
        
        Args:
            filepath (str): Chemin du modèle à charger
        """
        self.feature_extractor = keras.models.load_model(filepath)
    
    def visualize_activations(self, image: np.ndarray) -> None:
        """
        Visualise les activations des couches CNN pour une image.
        
        Args:
            image (np.ndarray): Image à analyser
        """
        # Prétraiter l'image
        if image.ndim == 2:
            image = image.reshape(1, *self.input_shape)
        elif image.ndim == 3 and image.shape[2] != self.input_shape[2]:
            image = image.reshape(1, *self.input_shape)
        else:
            image = image.reshape(1, *image.shape)
        
        # Obtenir les sorties des couches de convolution
        conv_layers = [layer for layer in self.feature_extractor.layers if 'conv' in layer.name]
        
        # Créer un modèle pour obtenir les activations
        layer_outputs = [layer.output for layer in conv_layers]
        activation_model = Model(inputs=self.feature_extractor.input, outputs=layer_outputs)
        
        # Obtenir les activations
        activations = activation_model.predict(image)
        
        # Afficher les activations
        fig, axes = plt.subplots(len(activations), 8, figsize=(20, 4 * len(activations)))
        
        for i, activation in enumerate(activations):
            activation = activation[0]
            n_features = min(8, activation.shape[2])
            
            for j in range(n_features):
                if len(activations) > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]
                
                ax.imshow(activation[:, :, j], cmap='viridis')
                ax.set_title(f"{conv_layers[i].name} - Feature {j+1}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()


def prepare_siamese_pairs(gallery: np.ndarray, gallery_identities: List[str], 
                         probes: np.ndarray, probe_identities: List[str], 
                         n_pairs: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prépare des paires d'images pour l'entraînement du modèle siamois.
    
    Args:
        gallery (np.ndarray): Images de la gallery
        gallery_identities (List[str]): Identités correspondant aux images gallery
        probes (np.ndarray): Images de probes
        probe_identities (List[str]): Identités correspondant aux probes
        n_pairs (int): Nombre de paires à générer
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Paires d'images et étiquettes (1: même identité, 0: identités différentes)
    """
    n_pairs_per_class = n_pairs // 2  # Moitié de paires positives, moitié négatives
    
    # Créer des dictionnaires d'identités
    gallery_by_identity = {}
    for i, identity in enumerate(gallery_identities):
        if identity not in gallery_by_identity:
            gallery_by_identity[identity] = []
        gallery_by_identity[identity].append(i)
    
    probe_by_identity = {}
    for i, identity in enumerate(probe_identities):
        if identity not in probe_by_identity:
            probe_by_identity[identity] = []
        probe_by_identity[identity].append(i)
    
    # Générer des paires positives (même identité)
    pairs = []
    labels = []
    
    # Identités communes entre gallery et probes
    common_identities = set(gallery_by_identity.keys()) & set(probe_by_identity.keys())
    
    # Paires positives
    positive_count = 0
    for identity in common_identities:
        if positive_count >= n_pairs_per_class:
            break
            
        gallery_indices = gallery_by_identity[identity]
        probe_indices = probe_by_identity[identity]
        
        for gallery_idx in gallery_indices:
            for probe_idx in probe_indices:
                pairs.append([gallery[gallery_idx], probes[probe_idx]])
                labels.append(1)
                positive_count += 1
                
                if positive_count >= n_pairs_per_class:
                    break
            if positive_count >= n_pairs_per_class:
                break
    
    # Paires négatives (identités différentes)
    negative_count = 0
    all_identities = list(set(gallery_identities + probe_identities))
    
    for _ in range(n_pairs_per_class):
        # Choisir deux identités différentes
        while True:
            identity1 = np.random.choice(all_identities)
            identity2 = np.random.choice(all_identities)
            
            if identity1 != identity2:
                break
        
        # Choisir une image de chaque identité
        if identity1 in gallery_by_identity:
            gallery_idx = np.random.choice(gallery_by_identity[identity1])
            img1 = gallery[gallery_idx]
        else:
            probe_idx = np.random.choice(probe_by_identity[identity1])
            img1 = probes[probe_idx]
            
        if identity2 in probe_by_identity:
            probe_idx = np.random.choice(probe_by_identity[identity2])
            img2 = probes[probe_idx]
        else:
            gallery_idx = np.random.choice(gallery_by_identity[identity2])
            img2 = gallery[gallery_idx]
        
        pairs.append([img1, img2])
        labels.append(0)
        negative_count += 1
        
    return np.array(pairs), np.array(labels)


def authenticate(probe: np.ndarray, gallery: np.ndarray, threshold: float, 
               authenticator: CNNAuthenticator) -> bool:
    """
    Authentifie une image requête en utilisant le modèle CNN.
    
    Args:
        probe (np.ndarray): Image requête à authentifier
        gallery (np.ndarray): Gallery d'images de référence (non utilisée directement)
        threshold (float): Seuil de distance pour l'authentification
        authenticator (CNNAuthenticator): Modèle CNN pré-entraîné
        
    Returns:
        bool: True si l'authentification est réussie, False sinon
    """
    # Vérifier que le modèle est entraîné
    if authenticator.gallery_features is None:
        authenticator.compute_gallery_features(gallery)
    
    # Prétraiter l'image requête
    if probe.ndim == 2:
        probe = probe.reshape(1, *authenticator.input_shape)
    elif probe.ndim == 3 and probe.shape[2] != authenticator.input_shape[2]:
        probe = probe.reshape(1, *authenticator.input_shape)
    else:
        probe = probe.reshape(1, *probe.shape)
    
    # Extraire les caractéristiques de l'image requête
    probe_features = authenticator.extract_features(probe)
    
    # Calculer les distances avec la gallery
    distances = np.sqrt(np.sum((authenticator.gallery_features - probe_features)**2, axis=1))
    min_distance = np.min(distances)
    
    # L'authentification réussit si la distance minimale est inférieure au seuil
    return min_distance <= threshold 