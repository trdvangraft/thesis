import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

def median_absolute_percentage_error(y_true, y_pred):
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    ape = np.abs(y_pred - y_true) / np.abs(y_true)
    return np.median(ape)

def mean_absolute_percentage_error(y_true, y_pred):
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    ape = np.abs(y_pred - y_true) / np.abs(y_true)
    return np.mean(ape)

def adjusted_r2(y_true, y_pred):
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    r2 = r2_score(y_true, y_pred)
    r2 = r2 if r2 > 0 else 0
    n = len(y_pred)
    k = 1
    return 1 - ((1 - r2)*(n - 1)/(n - k - 1))