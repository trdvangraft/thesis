from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

HYPERPARAMETERS = {
    'SVR': {
        'regressor': SVR(),
        'regressor__kernel': ['rbf', 'sigmoid'],
        'regressor__gamma': ['auto', 'scale'],
        'regressor__epsilon': [0.1, 0.01, 0.001, 0.0001],
        'regressor__C': [10, 100, 1000],
        'preprocessor__num__scaler': [MinMaxScaler(), StandardScaler()],
    },
    'RandomForestRegressor': {
        'regressor': RandomForestRegressor(),
        'regressor__n_estimators': [10, 25, 50, 75, 100],
        'regressor__criterion': ['squared_error', 'friedman_mse'],
        'regressor__max_depth': [5, 10, 20],
        'regressor__max_features': [1.0],
        'preprocessor__num__scaler': [MinMaxScaler(), StandardScaler()]
    },
    'ElasticNet': {
        'regressor': ElasticNet(),
        'regressor__l1_ratio': [0.01, 0.25, 0.5, 0.75, 1],
        'regressor__tol': [0.01],
        'preprocessor__num__scaler': [MinMaxScaler(), StandardScaler()]
    },
    'MLPRegressor': {
        'regressor': MLPRegressor(),
        'regressor__hidden_layer_sizes':  [[128, 32, 32], [64, 32]],
        'regressor__batch_size': [8, 16, 4, 2],
    }
}
