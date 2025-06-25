from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression


# fonction pour l'initialisation des estimateurs/pipelines
def initialisation():
    PR = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), SelectKBest(f_regression, k=19), LinearRegression())
    DTR = DecisionTreeRegressor()
    RG = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), SelectKBest(f_regression, k=19), Ridge())
    RFR = RandomForestRegressor(n_estimators=50, random_state=2)
    SVM = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), SVR(kernel='linear'))
    
    
    models = {
        "Polynomial Regression": PR,
        "Decision Tree Regressor": DTR,
        "Ridge": RG,
        "RandomForestRegressor": RFR,
        "SVM": SVM
    }
    return models
