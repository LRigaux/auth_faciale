# Système d'Authentification Faciale

Ce projet implémente un système d'authentification faciale en utilisant trois approches différentes:

1. **Approche par Force Brute** - Comparaison directe des images
2. **Eigenfaces** - Utilisation d'ACP pour réduire la dimensionnalité
3. **Deep Learning** - Utilisation de réseaux de neurones convolutifs (CNN)

## Structure du Projet

```
auth_faciale/
│
├── data/                    # Données d'entrée
│   ├── dataset1/           # Premier jeu de données
│   └── dataset2/           # Deuxième jeu de données
│
├── src/                     # Code source
│   ├── brute_force/        # Implémentation force brute
│   ├── eigenfaces/         # Implémentation eigenfaces
│   ├── deep_learning/      # Implémentation CNN
│   ├── utils/              # Utilitaires communs
│   ├── ui/                 # Interface utilisateur
│   └── api/                # API pour intégration web
│
├── tests/                   # Tests unitaires
│
├── requirements.txt         # Dépendances Python
└── README.md                # Ce fichier
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python src/ui/app.py
```

## Fonctionnalités

- Chargement et prétraitement des images faciales
- Séparation des données en gallery et probes
- Authentification par comparaison directe (force brute)
- Authentification par Eigenfaces (ACP)
- Authentification par réseau de neurones (CNN)
- Visualisation des couches du CNN
- Interface graphique pour tester et comparer les méthodes
- Évaluation des performances (précision, rappel, temps d'exécution)
