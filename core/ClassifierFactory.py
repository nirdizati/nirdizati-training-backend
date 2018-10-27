from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor

from ClassifierWrapper import ClassifierWrapper


def get_classifier(method, mode, max_features=None, n_estimators=None, learning_rate=None, random_state=None, min_cases_for_training=30, max_depth=None, subsample=None, colsample_bytree=None):

    if method == "xgb" and mode == "regr":
        return ClassifierWrapper(
            cls=XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample,
                                     max_depth=max_depth, colsample_bytree=colsample_bytree, n_jobs=-1, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    elif method == "xgb" and mode == "class":
        return ClassifierWrapper(
            cls=XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample,
                                     max_depth=max_depth, colsample_bytree=colsample_bytree, n_jobs=-1, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    elif method == "rf" and mode == "regr":
        return ClassifierWrapper(
            cls=RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, n_jobs=-1, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    elif method == "rf" and mode == "class":
        return ClassifierWrapper(
            cls=RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=-1, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)
               
    elif method == "gbm" and mode == "regr":
        return ClassifierWrapper(
            cls=GradientBoostingRegressor(n_estimators=n_estimators, max_features=max_features, learning_rate=learning_rate, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    elif method == "gbm" and mode == "class":
        return ClassifierWrapper(
            cls=GradientBoostingClassifier(n_estimators=n_estimators, max_features=max_features, learning_rate=learning_rate, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    elif method == "dt" and mode == "regr":
        return ClassifierWrapper(
            cls=DecisionTreeRegressor(max_depth=max_depth, max_features=max_features, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    elif method == "dt" and mode == "class":
        return ClassifierWrapper(
            cls=DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    else:
        print("Invalid classifier type")
        return None