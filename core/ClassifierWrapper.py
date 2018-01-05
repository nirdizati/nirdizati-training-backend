import numpy as np
from numpy import array

class ClassifierWrapper(object):
    
    def __init__(self, cls, mode, min_cases_for_training=30):
        self.cls = cls
        
        self.min_cases_for_training = min_cases_for_training
        self.hardcoded_prediction = None
        self.mode = mode

        
    def fit(self, X, y):
        # if all the training instances are of the same class, use this class as prediction
        if len(set(y)) < 2 and self.mode == "class":
            print("All samples are of one class. Defaulting to hardcoded predictions")
            self.hardcoded_prediction = y[0]

        # if there are too few training instances, use the mean
        elif X.shape[0] < self.min_cases_for_training:
            print("Too few samples. Defaulting to average predictions")
            if self.mode == "regr":
                self.hardcoded_prediction = np.mean(y)
            elif self.mode == "class":
                class_freqs = y.value_counts().sort_index()
                self.hardcoded_prediction = class_freqs / class_freqs.sum()

        else:
            self.cls.fit(X, y)
        return self
    
    
    def predict_proba(self, X, y=None):

        if self.hardcoded_prediction is not None:
            return array([self.hardcoded_prediction] * X.shape[0])
                        
        elif self.mode == "regr":
            preds = self.cls.predict(X)
            return preds

        elif self.mode == "class":
            # preds_pos_label_idx = np.where(self.cls.classes_ == 1)[0][0]
            # preds = self.cls.predict_proba(X)[:,preds_pos_label_idx]
            preds = self.cls.predict_proba(X)
            return preds

        else:
            print("Unrecognized training mode")
            return None
        
    
    def fit_predict(self, X, y):
        
        self.fit(X, y)
        return self.predict_proba(X)