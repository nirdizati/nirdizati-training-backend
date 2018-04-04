import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score, accuracy_score, log_loss

def get_score(actual, predicted, mode):
    score = {}
    if mode == "regr":
        score["mae"] = mean_absolute_error(actual, predicted)
        score["rmse"] = np.sqrt(mean_squared_error(actual, predicted))
        score["nmae"] = score["mae"] / np.mean(actual)
        score["nrmse"] = score["rmse"] / np.mean(actual)
    elif len(set(actual)) < 2:
        score = {"acc": 0, "f1": 0, "logloss": 0}
    else:
        predicted_labels = predicted.idxmax(axis=1)
        score["acc"] = accuracy_score(actual, predicted_labels)
        score["f1"] = f1_score(actual, predicted_labels, average='weighted')
        try:
            score["logloss"] = log_loss(actual, predicted, labels=predicted.columns)
        except ValueError:
            print("logloss cannot be calculated")
    return score
