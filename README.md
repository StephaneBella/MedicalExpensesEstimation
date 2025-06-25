#  Estimation Équitable des Charges Médicales Futures

##  Contexte du Projet

Ce projet s'inscrit dans le cadre d’une collaboration avec une **compagnie d'assurance santé privée**.  
L’objectif est de développer un modèle prédictif capable d’estimer les **charges médicales futures** des assurés, afin d’**ajuster les primes d’assurance** de manière cohérente, transparente et équitable.

---

##  Problématique

> Comment notre compagnie d'assurance santé privée peut-elle estimer de manière fiable les charges médicales futures de ses clients afin d'ajuster ses primes d'assurance, tout en garantissant l'équité et la soutenabilité financière du système ?

---

##  Contraintes et Engagements Éthiques

L’implémentation du modèle doit respecter plusieurs principes fondamentaux :

###  Non-discrimination
Aucune discrimination basée sur des **critères sensibles**, notamment :
- le **sexe**
- le **tabagisme**
- l’**âge**

###  Équité (Fairness)
- Éviter les **biais algorithmiques**
- Garantir une **égalité de traitement** entre les individus
- Suivre des **métriques de fairness** (parité démographique, égalité des chances, etc.)

###  Transparence
- Les décisions (ex. prime d’assurance élevée) doivent être **explicables**
- Utilisation de techniques d’**interprétabilité** (SHAP, LIME)

###  Robustesse aux biais
- Le modèle doit être **robuste face aux biais historiques** dans les données
- Utilisation de méthodes de **correction des biais** : prétraitement, régularisation, post-traitement

---

##  Structure du Projet

```bash
Medical_expenses_estimation_project/
│
├── README.md               # Ce fichier
├── requirements.txt        # Librairies nécessaires
├── data/                   # Données brutes et nettoyées
├── notebooks/              # Jupyter notebooks (EDA, modèle, etc.)
├── src/                    # Code source : loading, features, training
├── scripts/                # Scripts exécutables (CLI)
├── config/                 # Paramètres YAML et JSON
├── models/                 # Fichiers de modèles entraînés
├── outputs/                # Logs, graphes, métriques
└── tests/                  # Tests unitaires