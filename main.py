#!/usr/bin/env python
"""
Script principal pour lancer l'application d'authentification faciale.

Pour exécuter l'application, lancez simplement:
    python main.py
"""

from src.ui.app import app

if __name__ == "__main__":
    print("Démarrage de l'application d'authentification faciale...")
    print("Accédez à l'application sur http://127.0.0.1:8050/")
    app.run(debug=True, port=8050) 