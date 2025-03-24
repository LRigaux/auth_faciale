"""
Script principal pour le système d'authentification faciale.

Ce script lance l'application d'authentification faciale basée sur Dash.
"""

import sys
import os
import traceback

def main():
    """
    Fonction principale qui exécute l'application Dash.
    """
    try:
        print("Démarrage du système d'authentification faciale...")
        
        # Import différé pour éviter les problèmes d'initialisation précoce
        from src.ui.dash_app import app
        
        print("Interface accessible à l'adresse: http://127.0.0.1:8050")
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        print(f"Erreur lors du démarrage de l'application: {e}")
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 