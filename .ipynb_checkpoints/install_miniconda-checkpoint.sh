#!/bin/bash

# script va télécharger et installer Miniconda sur Linux.

# Télécharger le script d'installation de Miniconda
echo "Téléchargement de Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh

# Donner les permissions d'exécution au fichier téléchargé
echo "Donner les permissions d'exécution au script..."
chmod +x Miniconda3-latest-Linux-x86_64.sh


# Lancer le script d'installation de Miniconda
# -b : mode "batch", cela accepte automatiquement la licence et installe sans interaction
# -p : spécifie le répertoire d'installation (ici, dans le dossier $HOME/miniconda3)
echo "Installation de Miniconda..."
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

Conda dans le shell (permet de pouvoir utiliser `conda activate` et autres commandes Conda)
echo "Initialisation de Conda dans votre terminal..."
$HOME/miniconda3/bin/conda init


# ajouter Miniconda au PATH pour utiliser conda dans le terminal
# permet de reconnaître la commande 'conda' après l'installation
echo "Ajout de Miniconda au PATH..."
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc


# applique les modifications du fichier .bashrc
# actualise les changements sans avoir besoin de redémarrer le terminal
source ~/.bashrc


echo "Vérification de l'installation de Conda..."
conda --version

echo "Miniconda a été installé avec succès !"

echo "Suppression du fichier d'installation"
rm Miniconda3-latest-Linux-x86_64.sh