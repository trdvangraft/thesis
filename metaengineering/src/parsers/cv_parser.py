from typing import Any, Dict, List
import math

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import make_column_transformer, TransformedTargetRegressor

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

def to_cv_params(params: Dict[str, Dict[str, Any]]):
    return [{
        **model_param,
        'regressor__regressor': [model_param['regressor__regressor']]
    } for model_param in params.values()]

def parse_cv_result(model: TransformedTargetRegressor, c_best_model: pd.DataFrame):
    model: TransformedTargetRegressor = _cv_result_to_preprocessor(model, c_best_model)
    if any([name.startswith('param_regressor__pca__') for name in c_best_model.columns]):
        model = _cv_result_to_pca(model, c_best_model)
    model = _cv_result_to_model(model, c_best_model)
    return model

def _cv_result_to_preprocessor(model: TransformedTargetRegressor, c_best_model: pd.DataFrame):
    prefix = 'param_regressor__preprocessor__'

    names = ['MinMaxScaler', 'RobustScaler', 'StandardScaler']
    scalers = [MinMaxScaler(), RobustScaler(), StandardScaler()]

    for name, scaler in zip(names, scalers):
        if all(c_best_model[f'{prefix}num__scaler'].str.contains(name)):
            model.regressor.set_params(
                preprocessor__num__scaler=scaler
            )
    
    return model

def _cv_result_to_pca(model: TransformedTargetRegressor, c_best_model: pd.DataFrame):
    prefix= 'param_regressor__pca__'
    model.regressor.set_params(
        pca__n_components=c_best_model[f'{prefix}n_components'].values[0]
    )

    return model
    

def _cv_result_to_model(model: TransformedTargetRegressor, c_best_model: pd.DataFrame):
    prefix = 'param_regressor__'
    if all(c_best_model[f'{prefix}regressor'].str.contains('DecisionTreeRegressor')):
        print('DecisionRegressor model')
        # set parameters for decision tree
        model.regressor.set_params(
            regressor=DecisionTreeRegressor(),
            regressor__criterion=c_best_model[f'{prefix}regressor__criterion'].values[0],
            regressor__max_depth=None if math.isnan(r := c_best_model[f'{prefix}regressor__max_depth'].values[0]) else int(r)
        )
    elif all(c_best_model[f'{prefix}regressor'].str.contains('SVR')):
        print('SVR model')
        model.regressor.set_params(
            regressor=SVR(),
            regressor__kernel=c_best_model[f'{prefix}regressor__kernel'].values[0],
        )
    elif all(c_best_model[f'{prefix}regressor'].str.contains('ElasticNet')):
        print('ElasticNet model')
        model.regressor.set_params(
            regressor=ElasticNet(),
            regressor__l1_ratio=c_best_model[f'{prefix}regressor__l1_ratio'].values[0],
        )
    elif all(c_best_model[f'{prefix}regressor'].str.contains('RandomForestRegressor')):
        print('RandomForest model')
        model.regressor.set_params(
            regressor=RandomForestRegressor(),
            regressor__n_estimators=int(c_best_model[f'{prefix}regressor__n_estimators'].values[0]),
            regressor__criterion=c_best_model[f'{prefix}regressor__criterion'].values[0],
            regressor__max_depth=None if math.isnan(r := c_best_model[f'{prefix}regressor__max_depth'].values[0]) else int(r)
        )
    else:
        raise NotImplementedError(c_best_model[f'{prefix}regressor'])

    return model

def fmt_cv_results(df: pd.DataFrame):
    _df = df.copy()
    _df = _fmt_regressor(_df)
    _df['params_fmt'] = _df.apply(_fmt_param, axis=1)    
    _df.columns = _df.columns.map(lambda name: _rename(name))
    _df['mean'] = -1 * _df['mean']
    _df['std'] = -1 * _df['std']

    _df = _df[_df['params_fmt'] != "-"]
    # filter depth none 
    _df = _df[(_df['regressor'] == 'DecisionTreeRegressor()') & (_df['regressor__max_depth'] > 0) | (_df['regressor'] != 'DecisionTreeRegressor()')]
    return _df

def get_architectures(df: pd.DataFrame):
    return ['all'] + df['param_regressor__regressor'].dropna().unique().tolist()

def _fmt_param(param: pd.DataFrame):
    series = param[param.index.str.startswith('param_')].dropna()
    result = [f"{name.rsplit('__', 1)[1]}: {value}" for name, value in zip(series.index, series.values)]
    return '\n'.join(result)

def _fmt_regressor(df: pd.DataFrame):
    _df = df.copy()
    _df['param_regressor__regressor'] = _df['param_regressor__regressor'].replace(to_replace=r'^DecisionTreeRegressor.*', value="DecisionTreeRegressor()", regex=True)
    _df['param_regressor__regressor'] = _df['param_regressor__regressor'].replace(to_replace=r'^ElasticNet.*', value="ElasticNet()", regex=True)
    _df['param_regressor__regressor'] = _df['param_regressor__regressor'].replace(to_replace=r'^SVR.*', value="SVR()", regex=True)
    _df['param_regressor__regressor'] = _df['param_regressor__regressor'].replace(to_replace=r'^RandomForestRegressor.*', value="RandomForestRegressor()", regex=True)
    return _df

def _rename(name: str):
    name = name.removeprefix('param_regressor__')
    name = name.removesuffix('_test_score')
    return name


