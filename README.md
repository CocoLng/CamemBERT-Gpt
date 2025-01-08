# CamemBERT-Gpt

Projet Final pour Sorbonne Université, machine learning avancé

## Description du projet

Ce projet vise à recréer le modèle CamemBERT pour une utilisation dans un projet universitaire à Sorbonne Université. CamemBERT est un modèle de langage basé sur RoBERTa, pré-entraîné sur un large corpus de texte en français. Le projet inclut également une interface utilisateur interactive utilisant Gradio.

## Instructions d'installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/CocoLng/CamemBERT-Gpt.git
   cd CamemBERT-Gpt
   ```

2. Créez un environnement virtuel et activez-le :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
   ```

3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Instructions d'utilisation

1. Lancez l'interface utilisateur Gradio :
   ```bash
   python main.py
   ```

2. Ouvrez votre navigateur et accédez à l'URL fournie par Gradio (par défaut, http://127.0.0.1:7860).

3. Suivez les instructions à l'écran pour charger les données, configurer le modèle, entraîner le modèle et tester les prédictions.

## Structure du dépôt

- `data/` : Contient les scripts pour le chargement et le prétraitement des données.
- `model/` : Contient les configurations et les scripts liés au modèle CamemBERT.
- `process/` : Contient les scripts pour l'entraînement et le fine-tuning du modèle.
- `Overleaf/` : Contient les fichiers LaTeX pour la documentation du projet.
- `interface.py` : Script principal pour l'interface utilisateur Gradio.
- `main.py` : Point d'entrée principal pour lancer l'interface utilisateur.

## Modèle

Les poids du modèle peuvent être trouvés à l'adresse suivante : https://huggingface.co/CocoLng/CamemBERT-Gpt
