"""
Point d'entrée principal pour l'application d'authentification faciale.

Ce script initialise et lance l'application Dash.
"""

import os
import sys

# S'assurer que les modules du package sont importables
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer et lancer l'application
from src.ui.app import app

if __name__ == "__main__":
    app.run(debug=True, port=8050) 