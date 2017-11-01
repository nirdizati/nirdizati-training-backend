from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from ClassifierWrapper import ClassifierWrapper


def get_classifier(method, mode, n_estimators, max_features, gbm_learning_rate=None, random_state=None, min_cases_for_training=30):

    if method == "rf" and mode == "regr":
        return ClassifierWrapper(
            cls=RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    elif method == "rf" and mode == "class":
        return ClassifierWrapper(
            cls=RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)
               
    elif method == "gbm" and mode == "regr":
        return ClassifierWrapper(
            cls=GradientBoostingRegressor(n_estimators=n_estimators, max_features=max_features, learning_rate=gbm_learning_rate, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    elif method == "gbm" and mode == "class":
        return ClassifierWrapper(
            cls=GradientBoostingClassifier(n_estimators=n_estimators, max_features=max_features, learning_rate=gbm_learning_rate, random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    elif method == "dt" and mode == "regr":
        return ClassifierWrapper(
            cls=DecisionTreeRegressor(random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    elif method == "dt" and mode == "class":
        return ClassifierWrapper(
            cls=DecisionTreeClassifier(random_state=random_state),
            min_cases_for_training=min_cases_for_training, mode=mode)

    else:
        print("Invalid classifier type")
        return None