from src.data_loader import load_data, load_config
from src.preprocessing import preprocessing
from src.modeling import initialisation
from src.evaluation import save_model, save_model_params, evaluation



def main():
    # Recuperation des configurations
    config = load_config()

    # chargement des données
    df = load_data(path=config['data']['raw_path'])

    # preprocessing
    x_train, x_test, y_train, y_test = preprocessing(df)

    # initialisation des estimateurs et choix du modèle
    models = initialisation(x_train=x_train)
    model = models['Polynomial Regression']

    # entrinement et evaluation
    final_model = evaluation(model=model, x_train=x_train, x_test=x_test,
                             y_train=y_train, y_test=y_test, learning_curve_fig_path=config['outputs']['learning_curve&residuals'],
                             metrics_path=config['outputs']['metrics_path'])
    
    # sauvegarde du modele
    save_model(model=final_model, output_path=config['model']['model_path'])

    # sauvegarde des paramètres du modèle
    #save_model_params(final_model, output_path=config['model']['model_params_path'])

if __name__ == "__main__":
    main()