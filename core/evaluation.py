import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score, accuracy_score, log_loss

def get_score(actual, predicted, mode):
    score = {}
    if mode == "regr":
        score["mae"] = mean_absolute_error(actual, predicted)
        score["rmse"] = np.sqrt(mean_squared_error(actual, predicted))
        if np.mean(actual) != 0:
            score["nmae"] = score["mae"] / np.mean(actual)
            score["nrmse"] = score["rmse"] / np.mean(actual)
    else:
        predicted_labels = predicted.idxmax(axis=1)
        score["acc"] = accuracy_score(actual, predicted_labels)
        score["f1"] = f1_score(actual, predicted_labels, average='weighted')
        try:
            score["logloss"] = log_loss(actual, predicted)
            # score["logloss"] = log_loss(actual, predicted, labels=predicted.columns)
        except ValueError:
            print("logloss cannot be calculated")
    return score

def get_agg_score(actual, predicted, mode):
    score = {}
    if mode == "regr":
        score["metric"] = "mae"
        score["value"] = mean_absolute_error(actual, predicted)
    else:
        score["metric"] = "acc"
        score["value"] = accuracy_score(actual, predicted)
    return score
