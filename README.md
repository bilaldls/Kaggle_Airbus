# Surrogate Model Challenge - Hackathon 2iA 2025 (Airbus)

Ce projet implémente un modèle de substitution (Surrogate Model) pour Airbus, visant à prédire la `satisfaction` des processus de production à partir de données simulées complexes.

## 1. Prétraitement des Données (`pretraitement.py`)

Étant donné la très forte dimensionnalité du dataset, un preprocessing rigoureux a été mis en place pour optimiser la performance des modèles.

### 1.1 Chargement et Optimisation Mémoire
* Les fichiers `train.csv` et `test.csv` sont chargés via Pandas.
* Une optimisation automatique des types numériques (downcasting) est appliquée pour réduire l'empreinte mémoire et accélérer les calculs.

### 1.2 Suppression des Colonnes Inutiles
* Exclusion des colonnes `id` et des cibles (`wip`, `investissement`, `satisfaction`).
* Suppression des colonnes constantes (ne contenant qu'une seule valeur).
* Résultat : **1 484 colonnes constantes** ont été retirées du dataset initial.

### 1.3 Réduction de la Colinéarité
* Analyse de corrélation absolue sur les variables numériques par blocs de 500 colonnes.
* Élimination des variables présentant une corrélation supérieure à **0.95** afin de limiter la redondance d'information et le sur-apprentissage.
* Résultat : **579 colonnes** fortement corrélées ont été supprimées.

### 1.4 Jeu de Données Final
* Nombre final de variables explicatives : **5 528**.
* Alignement strict : mêmes colonnes et même ordre appliqués aux jeux d'entraînement et de test.
* Sauvegarde des datasets au format **Parquet** pour une efficacité maximale.

## 2. Architecture de Modélisation (`ensemble+residualLGBM.py`)

Le pipeline repose sur une approche ensembliste et corrective :

* **Base Ensemble** : Moyenne de 5 modèles LightGBM avec des graines (seeds) différentes pour stabiliser les prédictions.
* **Modèle Résiduel** : Un second étage de modèles (LightGBM) est entraîné pour apprendre et corriger les erreurs résiduelles du premier étage.
* **Calibration par Shift** : Recherche du décalage optimal (`shift`) pour maximiser l'exactitude dans la plage de tolérance de $\pm 0.05$.

## 3. Prérequis Techniques

* **Langage** : Python 3.10+
* **Bibliothèques principales** : `polars`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`.
* **Stockage** : Environ 5 Go d'espace libre pour les fichiers Parquet et les artifacts.

## 4. Organisation des Fichiers

* `/data` : Données sources (`train.csv`, `test.csv`) et fichiers Parquet nettoyés.
* `/artifacts` : Modèles entraînés, fichiers de calibration (`shift`) et rapports de performance.
* `pretraitement.py` : Script de nettoyage et réduction de dimension.
* `ensemble+residualLGBM.py` : Script d'entraînement, de stacking résiduel et de prédiction.

## 5. Procédure d'Exécution

Le pipeline doit être exécuté séquentiellement :

1. **Génération des données propres** :
   ```bash
   python pretraitement.py

2. **Entraînement et génération de la soumissions** :
   ```bash
   python ensemble+residualLGBM.py