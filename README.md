# Système d'Authentification Faciale

Cette application démontre deux approches d'authentification faciale : la méthode brute force et la méthode des eigenfaces.

## Installation

1. Clonez ce dépôt :
```
git clone https://github.com/votre-username/auth_faciale.git
cd auth_faciale
```

2. Installez les dépendances requises :
```
pip install -r requirements.txt
```

## Lancement de l'application

Exécutez l'application en utilisant la commande suivante :
```
python main.py
```

Vous pouvez également lancer directement le module de l'application :
```
python -m src.ui.app
```

L'application sera accessible à l'adresse [http://127.0.0.1:8050/](http://127.0.0.1:8050/).

## Structure du projet

```
auth_faciale/
├── assets/            # Fichiers statiques (images, CSS)
├── data/              # Dossiers de données
├── src/
│   ├── core/          # Fonctionnalités principales d'authentification
│   │   ├── __init__.py
│   │   ├── brute_force.py
│   │   ├── dataset.py
│   │   ├── eigenfaces.py
│   │   └── evaluation.py
│   └── ui/            # Interface utilisateur Dash
│       ├── __init__.py
│       ├── app.py
│       ├── callbacks.py
│       ├── layout.py
│       └── components/
│           ├── __init__.py
│           ├── brute_force.py
│           ├── common.py
│           ├── comparison.py
│           ├── eigenfaces.py
│           └── home.py
├── main.py            # Script principal pour lancer l'application
└── requirements.txt   # Dépendances Python
```

## Fonctionnalités

1. **Page d'accueil** : Chargement et visualisation du jeu de données
2. **Méthode brute force** : Authentification par comparaison directe d'images
3. **Méthode eigenfaces** : Authentification par analyse en composantes principales (PCA)
4. **Comparaison** : Évaluation et comparaison des performances des deux méthodes

## Technologies utilisées

- Dash et Dash Bootstrap Components pour l'interface utilisateur
- NumPy, scikit-learn pour le traitement des données et l'apprentissage
- Plotly pour les visualisations
- OpenCV et Pillow pour le traitement d'images
