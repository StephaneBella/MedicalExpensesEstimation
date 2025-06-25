import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_config
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# fonction de gestion des valeurs manquantes
def gestion_nan(df):
    return df.dropna()


# fonction pour la gestion des doublons
def gestion_doublons(df):
    return df.drop_duplicates()


# fonction pour l'encodage catégoriel
def encodage_catégoriel(df):
    cols = df.select_dtypes(include='object').drop(['region', 'sex'], axis=1).columns
    dummies = pd.get_dummies(df[cols], dtype=int)
    df = df.drop(columns=cols)
    df = df.drop(['region', 'sex'], axis=1)
    df = pd.concat([df, dummies], axis=1)
    return df

# fonction pour la normalisation
def normalisation(df):
    scaler = MinMaxScaler()
    cols_to_scale = df.select_dtypes(include=['number']).drop(['charges'], axis=1).columns
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df


# fonction pour la séparation en ensembles d'entrainement et de test
config = load_config()
def save_and_split(df, processed_path=config['data']['processed_path']):
    df.to_csv(processed_path, index=False)
    x = df.drop('charges', axis=1)
    y = df['charges']
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=0)
    return x_train, x_test, y_train, y_test


# fonction pour le preprocessing complet
def preprocessing(df):
    df = gestion_nan(df)
    df = gestion_doublons(df)
    df = encodage_catégoriel(df)
    x_train, x_test, y_train, y_test = save_and_split(df)
    return x_train, x_test, y_train, y_test


