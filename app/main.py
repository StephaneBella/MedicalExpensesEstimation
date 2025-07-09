# import des bibliotheques
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import uvicorn
import os
import src
from typing import List

# creation de l'objet API
app = FastAPI()

# chargement du modele
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'medical_cost_estimator.pkl')

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# creation de la classe des features
class chargesfeatures(BaseModel):
    age: float
    sex: object
    bmi: float
    children: float
    smoker: object


# chemin pour afficher le message de bienvenue
@app.get("/")
def read_root():
    return {"message": "API House price Prediction ready!"}

# chemin pour la prediction
@app.post("/predict")
def predict(data: chargesfeatures):
    # creation d'un DataFrame avec les colonnes dans le bon ordre
    input_df = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "bmi": data.bmi,
        "children": data.children,
        "smoker": data.smoker        
    }])


    # prediction
    log_prediction = model.predict(input_df)

    prediction = np.exp(log_prediction)

    return {
        "Les charges en dollars sont estimées à": prediction
    }

@app.post("/predict_batch")
def predict_batch(features_list: List[chargesfeatures]):

    # Construire un dataframe à partir de la liste d'objets chargesfeatures
    input_data = []

    for features in features_list:
        input_data.append({
            "age": features.age,
            "sex": features.sex,
            "bmi": features.bmi,
            "children": features.children,
            "smoker": features.smoker
        })

    input_df = pd.DataFrame(input_data)

    # prediction batch
    predictions = model.predict(input_df)

    # retourner la liste des predictions
    return {'charges estimées': predictions.tolist()}











