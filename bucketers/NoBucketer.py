# implementation based on https://github.com/irhete/predictive-monitoring-benchmark and https://github.com/nirdizati/nirdizati-training-backend

import pandas as pd
import numpy as np
import sys

class NoBucketer(object):
    
    def __init__(self, case_id_col):
        self.n_states = 1
        self.case_id_col = case_id_col
        
    
    def fit(self, X, y=None):
        
        return self
    
    
    def predict(self, X, y=None):
        
        return np.ones(len(X[self.case_id_col].unique()), dtype=np.int)
    
    
    def fit_predict(self, X, y=None):
        
        self.fit(X)
        return self.predict(X)