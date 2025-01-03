#!/bin/bash

ENV_NAME="camemBERTenv"

echo "### Initialisation de Conda dans le terminal ###"
if ! command -v conda &> /dev/null; then
  echo "Conda not found! Please install Miniconda or Anaconda first."
  exit 1
fi
conda init

echo "### Création de l'environnement Conda $ENV_NAME ###"
conda create -y -n $ENV_NAME python

echo "### Activation de l'environnement $ENV_NAME ###"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME
echo ">>> Environnement actif"

echo "### Installation des packages ###"
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia
conda install -y pytorch-lightning transformers datasets sentencepiece -c conda-forge
conda install -y scikit-learn

echo "### Installation d'ipykernel pour intégrer cet environnement dans Jupyter ###"
conda install -y ipykernel

echo "### Enregistrement de l'environnement $ENV_NAME dans Jupyter comme noyau ###"
python -m ipykernel install --user --name $ENV_NAME --display-name "Python ($ENV_NAME)"

echo "### Vérification de l'installation de PyTorch CUDA ###"
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

echo "### L'environnement $ENV_NAME a été créé et les packages ont été installés ###"

echo "### Liste des environnements Conda : ###"
conda env list

echo "############################################"
echo "### Activer l'environnement avec la commande : ###"
echo "conda activate $ENV_NAME"
echo "############################################"
