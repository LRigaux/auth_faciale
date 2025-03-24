"""
Page d'accueil de l'application.

Ce module contient la mise en page et les callbacks pour la page d'accueil.
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

def create_layout():
    """
    Crée la mise en page de la page d'accueil.
    
    Returns:
        dbc.Container: Mise en page de la page d'accueil
    """
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Système d'Authentification Faciale", className="display-4 mb-4"),
                html.P([
                    "Bienvenue dans le système d'authentification faciale. ",
                    "Ce système implémente plusieurs méthodes d'authentification ",
                    "et permet de comparer leurs performances."
                ], className="lead mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H2("Méthodes d'authentification", className="mb-3"),
                html.P([
                    "Ce système propose plusieurs méthodes d'authentification faciale. ",
                    "Choisissez une méthode pour explorer ses caractéristiques et performances."
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardImg(src="/assets/img/eigenfaces_icon.png", top=True, style={"opacity": 0.7}),
                            dbc.CardBody([
                                html.H4("Eigenfaces", className="card-title"),
                                html.P([
                                    "Méthode basée sur l'analyse en composantes principales (PCA) ",
                                    "pour réduire la dimensionnalité des images."
                                ], className="card-text"),
                                dbc.Button("Explorer", color="primary", href="/eigenfaces", className="mt-2")
                            ])
                        ], className="h-100")
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardImg(src="/assets/img/deeplearning_icon.png", top=True, style={"opacity": 0.7}),
                            dbc.CardBody([
                                html.H4("Deep Learning", className="card-title"),
                                html.P([
                                    "Méthode utilisant des réseaux de neurones profonds pour ",
                                    "extraire des caractéristiques discriminantes."
                                ], className="card-text"),
                                dbc.Button("Explorer", color="primary", href="/deep-learning", className="mt-2", disabled=True)
                            ])
                        ], className="h-100")
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardImg(src="/assets/img/bruteforce_icon.png", top=True, style={"opacity": 0.7}),
                            dbc.CardBody([
                                html.H4("Brute Force", className="card-title"),
                                html.P([
                                    "Méthode de base comparant directement les pixels ",
                                    "des images sans réduction de dimensionnalité."
                                ], className="card-text"),
                                dbc.Button("Explorer", color="primary", href="/brute-force", className="mt-2", disabled=True)
                            ])
                        ], className="h-100")
                    ], width=4)
                ], className="mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H2("À propos du projet", className="mb-3"),
                html.P([
                    "Ce projet a été développé dans le cadre d'un projet académique ",
                    "pour explorer différentes méthodes d'authentification faciale ",
                    "et comparer leurs performances sur différents jeux de données."
                ], className="mb-4"),
                html.P([
                    "Les fonctionnalités principales incluent :"
                ], className="mb-2"),
                html.Ul([
                    html.Li("Chargement de différents jeux de données"),
                    html.Li("Évaluation des performances des méthodes d'authentification"),
                    html.Li("Visualisation des résultats et métriques"),
                    html.Li("Tests d'authentification sur des images individuelles"),
                    html.Li("Comparaison des méthodes sur différentes métriques")
                ], className="mb-4"),
                html.P([
                    "Pour commencer, sélectionnez une méthode d'authentification ci-dessus."
                ], className="mb-3")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Footer([
                    html.Hr(),
                    html.P([
                        "© 2023 Système d'Authentification Faciale - ",
                        html.A("Documentation", href="#", className="text-decoration-none")
                    ], className="text-center text-muted")
                ])
            ], width=12)
        ])
    ], fluid=True, className="py-4") 