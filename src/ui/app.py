"""
Application Dash pour le système d'authentification faciale.

Ce module initialise l'application Dash avec une structure multi-pages
et configure les routes pour les différentes fonctionnalités.
"""

import os
import sys
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import des composants et pages
from src.ui.pages.home_page import create_layout as create_home_layout
from src.ui.pages.eigenfaces_page import create_layout as create_eigenfaces_layout
from src.ui.pages.brute_force_page import create_layout as create_brute_force_layout
from src.ui.utils.async_utils import clear_queues

# Créer les répertoires nécessaires s'ils n'existent pas
os.makedirs('assets/img', exist_ok=True)
os.makedirs('figures/eigenfaces', exist_ok=True)
os.makedirs('figures/brute_force', exist_ok=True)

# Initialisations
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
)

server = app.server

# Définir la mise en page de l'application
app.layout = html.Div([
    # Navbar
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Authentification Faciale", href="/"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Accueil", href="/")),
                dbc.NavItem(dbc.NavLink("Eigenfaces", href="/eigenfaces")),
                dbc.NavItem(dbc.NavLink("Force Brute", href="/brute-force")),
                dbc.NavItem(dbc.NavLink("Documentation", href="/documentation", disabled=True)),
            ]),
        ], fluid=True),
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    # Contenu principal qui sera mis à jour par le callback
    html.Div(id="page-content"),
    
    # Location pour stocker l'URL actuelle
    dcc.Location(id="url", refresh=False)
])

# Callback pour mettre à jour le contenu en fonction de l'URL
@callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    """
    Met à jour le contenu de la page en fonction de l'URL.
    
    Args:
        pathname: Chemin de l'URL actuelle
        
    Returns:
        Layout: Mise en page correspondant à l'URL
    """
    # Réinitialiser les files d'attente pour les opérations asynchrones
    clear_queues()
    
    # Rediriger vers la page appropriée
    if pathname == "/eigenfaces":
        return create_eigenfaces_layout()
    elif pathname == "/brute-force":
        return create_brute_force_layout()
    elif pathname == "/deep-learning":
        return html.Div([
            html.H1("Deep Learning - En développement", className="text-center mt-5"),
            dbc.Button("Retour à l'accueil", href="/", color="primary", className="mt-3 d-block mx-auto")
        ])
    elif pathname == "/documentation":
        return html.Div([
            html.H1("Documentation - En développement", className="text-center mt-5"),
            dbc.Button("Retour à l'accueil", href="/", color="primary", className="mt-3 d-block mx-auto")
        ])
    
    # Par défaut, afficher la page d'accueil
    return create_home_layout()

# Exécuter l'application si ce fichier est exécuté directement
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, dev_tools_hot_reload=True, dev_tools_ui=True, port=8050) 