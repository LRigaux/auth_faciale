"""
Interface graphique pour le système d'authentification faciale.

Ce module fournit une interface graphique pour tester et comparer
les différentes méthodes d'authentification.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Ajouter le répertoire parent au chemin de recherche pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import load_dataset, evaluate_authentication_method, compare_methods
import src.brute_force as brute_force
import src.eigenfaces as eigenfaces
import src.deep_learning as deep_learning


class FaceAuthenticationApp:
    """
    Application d'authentification faciale avec interface graphique.
    """
    
    def __init__(self, root):
        """
        Initialise l'interface graphique.
        
        Args:
            root: Fenêtre principale Tkinter
        """
        self.root = root
        self.root.title("Système d'Authentification Faciale")
        self.root.geometry("1000x700")
        
        # Variables
        self.dataset = None
        self.dataset_num = tk.IntVar(value=1)
        self.method = tk.StringVar(value="brute_force")
        self.gallery_processed = None
        self.probes_processed = None
        self.eigenfaces_model = None
        self.cnn_model = None
        self.current_probe = None
        
        # Création de l'interface
        self._create_widgets()
        
    def _create_widgets(self):
        """Crée les widgets de l'interface."""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Zone de configuration
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Sélection du dataset
        ttk.Label(config_frame, text="Dataset:").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(config_frame, text="Dataset 1", variable=self.dataset_num, value=1).grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(config_frame, text="Dataset 2", variable=self.dataset_num, value=2).grid(row=0, column=2, sticky=tk.W)
        
        # Sélection de la méthode
        ttk.Label(config_frame, text="Méthode:").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(config_frame, text="Force Brute", variable=self.method, value="brute_force").grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(config_frame, text="Eigenfaces", variable=self.method, value="eigenfaces").grid(row=1, column=2, sticky=tk.W)
        ttk.Radiobutton(config_frame, text="CNN", variable=self.method, value="cnn").grid(row=1, column=3, sticky=tk.W)
        
        # Bouton de chargement des données
        ttk.Button(config_frame, text="Charger les données", command=self._load_data).grid(row=2, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        # Bouton d'évaluation
        ttk.Button(config_frame, text="Évaluer les performances", command=self._evaluate_performance).grid(row=2, column=2, columnspan=2, pady=10, sticky=tk.W)
        
        # Zone de test
        test_frame = ttk.LabelFrame(main_frame, text="Test d'authentification", padding="10")
        test_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Cadre gauche (image à tester)
        left_frame = ttk.Frame(test_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(left_frame, text="Image à authentifier:").pack(anchor=tk.W)
        
        self.canvas_probe = tk.Canvas(left_frame, width=150, height=150, bg='white')
        self.canvas_probe.pack(pady=10)
        
        ttk.Button(left_frame, text="Sélectionner une image", command=self._select_image).pack(pady=5)
        ttk.Button(left_frame, text="Authentifier", command=self._authenticate).pack(pady=5)
        
        # Cadre droit (résultats)
        right_frame = ttk.Frame(test_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(right_frame, text="Résultats:").pack(anchor=tk.W)
        
        self.result_text = tk.Text(right_frame, width=40, height=10, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Zone de visualisation
        viz_frame = ttk.LabelFrame(main_frame, text="Visualisation", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Button(viz_frame, text="Visualiser les métriques", command=self._visualize_metrics).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_frame, text="Visualiser les Eigenfaces", command=self._visualize_eigenfaces).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_frame, text="Visualiser les activations CNN", command=self._visualize_cnn_activations).pack(side=tk.LEFT, padx=5)
    
    def _load_data(self):
        """Charge les données du dataset sélectionné."""
        try:
            dataset_num = self.dataset_num.get()
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Chargement du dataset {dataset_num}...\n")
            self.root.update()
            
            # Charger le dataset
            self.dataset = load_dataset(dataset_num)
            
            # Prétraiter les images
            self.gallery_processed, self.probes_processed = self.dataset.preprocess_images()
            
            # Réinitialiser les modèles
            self.eigenfaces_model = None
            self.cnn_model = None
            
            self.result_text.insert(tk.END, f"Dataset {dataset_num} chargé avec succès!\n")
            self.result_text.insert(tk.END, f"Gallery: {self.gallery_processed.shape[0]} images\n")
            self.result_text.insert(tk.END, f"Probes: {self.probes_processed.shape[0]} images\n")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement des données: {str(e)}")
            raise
    
    def _evaluate_performance(self):
        """Évalue les performances de la méthode sélectionnée."""
        if self.dataset is None:
            messagebox.showwarning("Attention", "Veuillez d'abord charger les données.")
            return
        
        method = self.method.get()
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Évaluation de la méthode {method}...\n")
        self.root.update()
        
        try:
            if method == "brute_force":
                # Séparer les probes en "enregistrés" et "non enregistrés"
                enrolled_indices = [i for i, gt in enumerate(self.dataset.ground_truth) if gt]
                non_enrolled_indices = [i for i, gt in enumerate(self.dataset.ground_truth) if not gt]
                
                enrolled_probes = self.probes_processed[enrolled_indices]
                non_enrolled_probes = self.probes_processed[non_enrolled_indices]
                
                # Trouver le meilleur rayon
                radius = brute_force.find_best_radius(
                    self.gallery_processed, 
                    enrolled_probes, 
                    non_enrolled_probes
                )
                
                # Évaluer les performances
                metrics = evaluate_authentication_method(
                    lambda p, g, **kwargs: brute_force.authenticate(p, g, radius),
                    self.gallery_processed,
                    self.probes_processed,
                    self.dataset.ground_truth
                )
                
                self.result_text.insert(tk.END, f"Rayon optimal: {radius:.4f}\n\n")
                self.result_text.insert(tk.END, str(metrics))
                
            elif method == "eigenfaces":
                # Séparer les probes en "enregistrés" et "non enregistrés"
                enrolled_indices = [i for i, gt in enumerate(self.dataset.ground_truth) if gt]
                non_enrolled_indices = [i for i, gt in enumerate(self.dataset.ground_truth) if not gt]
                
                enrolled_probes = self.probes_processed[enrolled_indices]
                non_enrolled_probes = self.probes_processed[non_enrolled_indices]
                
                # Préparer les données pour Eigenfaces (aplatir les images)
                gallery_flat = self.gallery_processed.reshape(self.gallery_processed.shape[0], -1)
                enrolled_flat = enrolled_probes.reshape(enrolled_probes.shape[0], -1)
                non_enrolled_flat = non_enrolled_probes.reshape(non_enrolled_probes.shape[0], -1)
                
                # Trouver le meilleur rayon et entraîner le modèle
                radius, self.eigenfaces_model = eigenfaces.find_best_radius(
                    gallery_flat,
                    enrolled_flat,
                    non_enrolled_flat
                )
                
                # Évaluer les performances
                metrics = evaluate_authentication_method(
                    lambda p, g, **kwargs: eigenfaces.authenticate(p, g, radius, self.eigenfaces_model),
                    gallery_flat,
                    self.probes_processed.reshape(self.probes_processed.shape[0], -1),
                    self.dataset.ground_truth
                )
                
                self.result_text.insert(tk.END, f"Rayon optimal: {radius:.4f}\n")
                self.result_text.insert(tk.END, f"Nombre de composantes utilisées: {self.eigenfaces_model.n_components}\n\n")
                self.result_text.insert(tk.END, str(metrics))
                
            elif method == "cnn":
                # Créer un modèle CNN (dans un cas réel, on l'entraînerait ici)
                # Pour simplifier, nous utiliserons un modèle non entraîné avec un seuil arbitraire
                messagebox.showinfo("Information", "L'entraînement d'un CNN prend du temps et n'est pas inclus dans cette démo.\nUn modèle de test sera utilisé.")
                
                self.cnn_model = deep_learning.CNNAuthenticator()
                
                self.result_text.insert(tk.END, "Méthode CNN non implémentée dans cette démo interactive.\n")
                self.result_text.insert(tk.END, "Dans un cas réel, le modèle serait entraîné ici.\n")
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'évaluation: {str(e)}")
            raise
    
    def _select_image(self):
        """Sélectionne une image à authentifier."""
        if self.dataset is None:
            messagebox.showwarning("Attention", "Veuillez d'abord charger les données.")
            return
        
        # Pour simplifier, on choisit une image aléatoire des probes
        idx = np.random.randint(0, len(self.dataset.probes))
        self.current_probe = self.dataset.probes[idx]
        self.current_probe_identity = self.dataset.probe_identities[idx]
        self.current_ground_truth = self.dataset.ground_truth[idx]
        
        # Afficher l'image
        img = Image.fromarray(self.current_probe).resize((150, 150))
        self.probe_photo = ImageTk.PhotoImage(img)
        self.canvas_probe.create_image(75, 75, image=self.probe_photo)
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Image sélectionnée.\n")
        self.result_text.insert(tk.END, f"Identité: {self.current_probe_identity}\n")
        self.result_text.insert(tk.END, f"Devrait être authentifié: {self.current_ground_truth}\n")
    
    def _authenticate(self):
        """Authentifie l'image sélectionnée."""
        if not hasattr(self, 'current_probe') or self.current_probe is None:
            messagebox.showwarning("Attention", "Veuillez d'abord sélectionner une image.")
            return
        
        method = self.method.get()
        
        try:
            if method == "brute_force":
                # Aplatir les images
                probe_flat = self.current_probe.flatten() / 255.0
                gallery_flat = self.gallery_processed.reshape(self.gallery_processed.shape[0], -1)
                
                # Trouver le rayon optimal
                enrolled_indices = [i for i, gt in enumerate(self.dataset.ground_truth) if gt]
                non_enrolled_indices = [i for i, gt in enumerate(self.dataset.ground_truth) if not gt]
                
                enrolled_probes = self.probes_processed[enrolled_indices].reshape(len(enrolled_indices), -1)
                non_enrolled_probes = self.probes_processed[non_enrolled_indices].reshape(len(non_enrolled_indices), -1)
                
                radius = brute_force.find_best_radius(
                    gallery_flat,
                    enrolled_probes,
                    non_enrolled_probes
                )
                
                # Authentifier
                result = brute_force.authenticate(probe_flat, gallery_flat, radius)
                
                # Afficher le résultat
                self.result_text.insert(tk.END, "\nRésultat d'authentification (Force Brute):\n")
                self.result_text.insert(tk.END, f"Accès {'autorisé' if result else 'refusé'}\n")
                
            elif method == "eigenfaces":
                if self.eigenfaces_model is None:
                    messagebox.showinfo("Information", "Le modèle Eigenfaces n'est pas encore entraîné. Évaluez d'abord les performances.")
                    return
                
                # Aplatir l'image
                probe_flat = self.current_probe.flatten() / 255.0
                gallery_flat = self.gallery_processed.reshape(self.gallery_processed.shape[0], -1)
                
                # Trouver le rayon optimal
                enrolled_indices = [i for i, gt in enumerate(self.dataset.ground_truth) if gt]
                non_enrolled_indices = [i for i, gt in enumerate(self.dataset.ground_truth) if not gt]
                
                enrolled_probes = self.probes_processed[enrolled_indices].reshape(len(enrolled_indices), -1)
                non_enrolled_probes = self.probes_processed[non_enrolled_indices].reshape(len(non_enrolled_indices), -1)
                
                radius, _ = eigenfaces.find_best_radius(
                    gallery_flat,
                    enrolled_probes,
                    non_enrolled_probes
                )
                
                # Authentifier
                result = eigenfaces.authenticate(probe_flat, gallery_flat, radius, self.eigenfaces_model)
                
                # Afficher le résultat
                self.result_text.insert(tk.END, "\nRésultat d'authentification (Eigenfaces):\n")
                self.result_text.insert(tk.END, f"Accès {'autorisé' if result else 'refusé'}\n")
                
            elif method == "cnn":
                self.result_text.insert(tk.END, "\nMéthode CNN non implémentée dans cette démo interactive.\n")
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'authentification: {str(e)}")
            raise
    
    def _visualize_metrics(self):
        """Visualise les métriques de performance."""
        messagebox.showinfo("Information", "Cette fonctionnalité serait implémentée en production.")
    
    def _visualize_eigenfaces(self):
        """Visualise les Eigenfaces."""
        if self.eigenfaces_model is None:
            messagebox.showwarning("Attention", "Veuillez d'abord évaluer les performances de la méthode Eigenfaces.")
            return
        
        # Visualiser les eigenfaces
        self.eigenfaces_model.visualize_eigenfaces((150, 150))
        
        # Visualiser la variance expliquée
        self.eigenfaces_model.visualize_variance()
    
    def _visualize_cnn_activations(self):
        """Visualise les activations des couches CNN."""
        if self.cnn_model is None:
            messagebox.showwarning("Attention", "Veuillez d'abord évaluer les performances de la méthode CNN.")
            return
        
        messagebox.showinfo("Information", "Cette fonctionnalité serait implémentée en production.")


def main():
    """Fonction principale pour lancer l'application."""
    root = tk.Tk()
    app = FaceAuthenticationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
