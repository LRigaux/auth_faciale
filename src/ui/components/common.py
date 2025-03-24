"""
Composants communs pour l'interface utilisateur.

Ce module fournit des composants réutilisables pour créer l'interface utilisateur.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

def create_navbar(app_title: str = "Authentification Faciale") -> dbc.Navbar:
    """
    Crée une barre de navigation.
    
    Args:
        app_title (str): Titre de l'application
        
    Returns:
        dbc.Navbar: Composant barre de navigation
    """
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(app_title, className="ms-2"),
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Accueil", href="/")),
                        dbc.NavItem(dbc.NavLink("Force Brute", href="/brute-force")),
                        dbc.NavItem(dbc.NavLink("Eigenfaces", href="/eigenfaces")),
                        dbc.NavItem(dbc.NavLink("CNN", href="/cnn")),
                        dbc.NavItem(dbc.NavLink("Comparaison", href="/compare"))
                    ],
                    className="ms-auto",
                    navbar=True
                ),
            ]
        ),
        color="primary",
        dark=True,
        className="mb-4"
    )

def create_footer() -> html.Footer:
    """
    Crée un pied de page.
    
    Returns:
        html.Footer: Composant pied de page
    """
    return html.Footer(
        dbc.Container(
            [
                html.Hr(),
                html.P(
                    "Système d'authentification faciale © 2025",
                    className="text-center text-muted"
                )
            ]
        ),
        className="mt-4"
    )

def create_card(title: str, children=None, className: str = "mb-4") -> dbc.Card:
    """
    Crée une carte avec un titre et un contenu.
    
    Args:
        title (str): Titre de la carte
        children: Contenu de la carte
        className (str): Classes CSS additionnelles
        
    Returns:
        dbc.Card: Composant carte
    """
    return dbc.Card(
        [
            dbc.CardHeader(title),
            dbc.CardBody(children or [])
        ],
        className=className
    )

def create_progress_card() -> dbc.Card:
    """
    Crée une carte avec une barre de progression.
    
    Returns:
        dbc.Card: Composant carte avec barre de progression
    """
    return create_card(
        "Progression",
        [
            dbc.Progress(id="progress-bar", value=0, style={"height": "20px", "marginBottom": "10px"}),
            html.P(id="progress-text", children="")
        ]
    )

def create_dataset_selector() -> dbc.Card:
    """
    Crée une carte avec un sélecteur de dataset.
    
    Returns:
        dbc.Card: Composant carte avec sélecteur de dataset
    """
    return create_card(
        "Configuration",
        [
            html.Div([
                html.Label("Dataset:"),
                dbc.RadioItems(
                    id="dataset-select",
                    options=[
                        {"label": "Dataset 1", "value": 1},
                        {"label": "Dataset 2", "value": 2}
                    ],
                    value=1,
                    inline=True
                )
            ], className="mb-3"),
            dbc.Button("Charger les données", id="load-data-btn", color="primary")
        ]
    )

def create_method_selector() -> dbc.Card:
    """
    Crée une carte avec un sélecteur de méthode.
    
    Returns:
        dbc.Card: Composant carte avec sélecteur de méthode
    """
    return create_card(
        "Méthode",
        [
            html.Div([
                html.Label("Méthode:"),
                dbc.RadioItems(
                    id="method-select",
                    options=[
                        {"label": "Force Brute", "value": "brute_force"},
                        {"label": "Eigenfaces", "value": "eigenfaces"},
                        {"label": "CNN", "value": "cnn"}
                    ],
                    value="brute_force",
                    inline=True
                )
            ], className="mb-3"),
            dbc.Button("Évaluer les performances", id="evaluate-btn", color="success", disabled=True)
        ]
    )

def create_authentication_test_card() -> dbc.Card:
    """
    Crée une carte pour tester l'authentification.
    
    Returns:
        dbc.Card: Composant carte pour le test d'authentification
    """
    return create_card(
        "Test d'authentification",
        [
            html.Div([
                html.Img(id="probe-image", style={
                    "width": "150px", 
                    "height": "150px", 
                    "marginBottom": "10px",
                    "border": "1px solid #ddd"
                }),
                html.Div([
                    dbc.Button("Sélectionner une image", id="select-image-btn", color="info", 
                              className="me-2", disabled=True),
                    dbc.Button("Authentifier", id="authenticate-btn", color="danger", disabled=True)
                ], style={"marginTop": "10px"})
            ], style={"textAlign": "center"})
        ]
    )

def create_results_card() -> dbc.Card:
    """
    Crée une carte pour afficher les résultats.
    
    Returns:
        dbc.Card: Composant carte pour les résultats
    """
    return create_card(
        "Résultats",
        [
            html.Pre(id="result-text", style={
                "whiteSpace": "pre-wrap", 
                "wordBreak": "break-all",
                "maxHeight": "200px",
                "overflowY": "auto"
            })
        ]
    )

def create_evaluation_results_card() -> dbc.Card:
    """
    Crée une carte pour afficher les résultats d'évaluation.
    
    Returns:
        dbc.Card: Composant carte pour les résultats d'évaluation
    """
    return create_card(
        "Résultats d'évaluation",
        [
            html.Pre(id="evaluation-result-text", style={
                "whiteSpace": "pre-wrap", 
                "wordBreak": "break-all",
                "maxHeight": "200px",
                "overflowY": "auto"
            })
        ]
    )

def create_authentication_results_card() -> dbc.Card:
    """
    Crée une carte pour afficher les résultats d'authentification.
    
    Returns:
        dbc.Card: Composant carte pour les résultats d'authentification
    """
    return create_card(
        "Résultat du test d'authentification",
        [
            html.Pre(id="authentication-result-text", style={
                "whiteSpace": "pre-wrap", 
                "wordBreak": "break-all",
                "maxHeight": "200px",
                "overflowY": "auto"
            })
        ]
    )

def create_visualization_card() -> dbc.Card:
    """
    Crée une carte pour la visualisation.
    
    Returns:
        dbc.Card: Composant carte pour la visualisation
    """
    return create_card(
        "Visualisation",
        [
            dbc.ButtonGroup([
                dbc.Button("Visualiser les métriques", id="viz-metrics-btn", 
                          color="secondary", className="me-2", disabled=True),
                dbc.Button("Visualiser les Eigenfaces", id="viz-eigenfaces-btn", 
                          color="secondary", className="me-2", disabled=True),
                dbc.Button("Visualiser la matrice de confusion", id="viz-confmat-btn", 
                          color="secondary", disabled=True)
            ]),
            html.Div(id="visualization-container", className="mt-3")
        ]
    )

def create_modal(id_base: str, title: str = "Information") -> dbc.Modal:
    """
    Crée une boîte de dialogue modale.
    
    Args:
        id_base (str): Préfixe d'identifiant pour la modale
        title (str): Titre de la modale
        
    Returns:
        dbc.Modal: Composant modale
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(title),
            dbc.ModalBody(id=f"{id_base}-body"),
            dbc.ModalFooter(
                dbc.Button("Fermer", id=f"close-{id_base}", className="ms-auto", n_clicks=0)
            ),
        ],
        id=f"{id_base}-modal",
        is_open=False,
    )

def create_graph_display() -> html.Div:
    """
    Crée un conteneur pour afficher un graphique.
    
    Returns:
        html.Div: Conteneur pour graphique
    """
    return html.Div([
        dcc.Graph(id="visualization-graph")
    ])

def create_image_display(image_id: str, caption_id: str = None, width: str = "100%") -> html.Div:
    """
    Crée un conteneur pour afficher une image.
    
    Args:
        image_id (str): Identifiant de l'image
        caption_id (str): Identifiant de la légende (optionnel)
        width (str): Largeur de l'image
        
    Returns:
        html.Div: Conteneur pour image
    """
    children = [html.Img(id=image_id, style={"width": width})]
    
    if caption_id:
        children.append(html.P(id=caption_id, className="text-center"))
        
    return html.Div(children, className="text-center")

def create_info_row(title: str, value_id: str) -> dbc.Row:
    """
    Crée une ligne d'information avec un titre et une valeur.
    
    Args:
        title (str): Titre de l'information
        value_id (str): Identifiant de l'élément contenant la valeur
        
    Returns:
        dbc.Row: Ligne d'information
    """
    return dbc.Row([
        dbc.Col(html.Strong(f"{title}:"), width=4),
        dbc.Col(html.Span(id=value_id), width=8)
    ], className="mb-2")

def create_timing_card() -> dbc.Card:
    """
    Crée une carte pour afficher les temps d'exécution.
    
    Returns:
        dbc.Card: Composant carte pour les temps d'exécution
    """
    return create_card(
        "Temps d'exécution",
        [
            create_info_row("Chargement des données", "loading-time"),
            create_info_row("Prétraitement", "preprocessing-time"),
            create_info_row("Évaluation des performances", "evaluation-time"),
            create_info_row("Authentification", "authentication-time")
        ]
    )

def create_tab_layout(tabs_content: list) -> html.Div:
    """
    Crée une mise en page à onglets.
    
    Args:
        tabs_content (list): Liste de tuples (label, content) pour chaque onglet
        
    Returns:
        html.Div: Mise en page à onglets
    """
    tabs = []
    contents = []
    
    for i, (label, content) in enumerate(tabs_content):
        tab_id = f"tab-{i}"
        tabs.append(dbc.Tab(label=label, tab_id=tab_id))
        contents.append(html.Div(content, id=f"content-{tab_id}", style={"display": "none"}))
    
    return html.Div([
        dbc.Tabs(tabs, id="tabs", active_tab="tab-0"),
        html.Div(contents, id="tabs-content", className="mt-3")
    ])

def create_figure_display(figure_id: str) -> html.Div:
    """
    Crée un conteneur pour afficher une figure enregistrée.
    
    Args:
        figure_id (str): Identifiant de la figure
        
    Returns:
        html.Div: Conteneur pour figure
    """
    return html.Div([
        html.Img(id=figure_id, style={"width": "100%"}),
        dbc.Button("Actualiser", id=f"refresh-{figure_id}", color="link", className="mt-2")
    ], className="text-center") 