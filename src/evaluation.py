from sklearn.metrics import mean_absolute_error, root_mean_squared_log_error, r2_score, root_mean_squared_error
from sklearn.model_selection import learning_curve
from src.data_loader import load_config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import json


# fonction pour sauvegarder le model
config = load_config()
def save_model(model, output_path='/models/medical_cost_estimator.pkl'): #output_path=config['model']['model_path']
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Modèle sauvegardé dans: {output_path}')


# fonction pour sauvegarder les paramètres du modèle
def save_model_params(model, output_path=config['model']['model_params_path']):
    # récuperer les coefficients
    coefficients = model.named_steps['linearregression'].coef_.flatten()

    # recuperer tous les biais
    intercept = model.named_steps['linearregression'].intercept_

    # recupérer tous les noms des features (anciennes + nouvelles) obtenues avec polynomialfeatures
    feature_names = model.named_steps['poly'].get_feature_names_out()

    params_dict = {'intercept': float(intercept)}
    for feat, coef in zip(feat, coefficients):
        params_dict[feat] = float(coef)
    
    with open(output_path, 'w') as f:
        json.dump(params_dict, f, indent=4)


# fonction pour charger le modèle
def load_model(path=config['model']['model_path']):
    with open(path, "r") as f:
        model = pickle.load(path)
    
    return model


# fonction d'evaluation
def evaluation(model, x_train, x_test, y_train, y_test, 
               learning_curve_fig_path=config['outputs']['learning_curve&residuals'],
               metrics_path=config['outputs']['metrics_path']):

    y_train_log = np.log(y_train)
    model = model.fit(x_train, y_train_log)
    y_pred_log = model.predict(x_test)
    y_pred = np.exp(y_pred_log)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmsle = root_mean_squared_log_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #recuperer les metriques
    results = {
        "R2": r2,
        "MAE": mae,
        "RMSLE": rmsle,
        "RMSE": rmse
    }
    
    #sauvegarde des metriques
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[✔] Résultats sauvegardés dans :\n- {learning_curve_fig_path}\n- {metrics_path}")
    

    # receuil des residus
    residus = (y_test - y_pred)
    residus = pd.DataFrame(residus)

    # learning curve
    N, train_score, val_score = learning_curve(model, x_train, y_train_log,
                                               cv=4, scoring='neg_root_mean_squared_error', train_sizes=np.linspace(0.1, 1, 10))
    
    # affichage de la courbe d'apprentissage et des residus
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(N, train_score.mean(axis=1), label='Training')
    plt.plot(N, val_score.mean(axis=1), label='Validation')
    plt.ylabel('RMSLE')
    plt.savefig(learning_curve_fig_path)
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(residus, label='Distribution des Residus', kde=True)
    plt.legend()
    plt.show()

    return model


