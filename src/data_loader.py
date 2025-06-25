import pandas as pd
import yaml
import os


# fonction pour charger les config du fichier config.yaml
def load_config(config_path='C:/Users/DELL/Documents/VEMV/pycaret/work/Projets_professionnels/Medical_expenses_estimation_project/config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        return config
    
# fonction de chargement des données proprement dite
def load_data(path):
    """Cette fonction charge les données depuis le chemin spécifié"""
    return pd.read_csv(path)