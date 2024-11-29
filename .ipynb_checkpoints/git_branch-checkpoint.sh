#!/bin/bash

# Demander le nom de la branche
echo "Entrez le nom de la branche (existant ou nouveau) :"
read branch_name

# Vérifier si la branche existe déjà
git branch --list "$branch_name" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "La branche '$branch_name' existe déjà. Passage à cette branche."
    git checkout "$branch_name"
else
    echo "La branche '$branch_name' n'existe pas. Création de la branche."
    git checkout -b "$branch_name"
fi

# Demander le message de commit
echo "Entrez le message du commit :"
read commit_message

# Ajouter tous les fichiers modifiés
git add .

# Effectuer le commit
git commit -m "$commit_message"

# Pousser la branche sur le dépôt distant
git push origin "$branch_name"

echo "Changements poussés sur la branche '$branch_name'."

