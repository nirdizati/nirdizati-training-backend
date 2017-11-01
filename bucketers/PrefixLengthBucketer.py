# implementation based on https://github.com/irhete/predictive-monitoring-benchmark and https://github.com/nirdizati/nirdizati-training-backend

import pandas as pd
import numpy as np
import sys

class PrefixLengthBucketer(object):
    
    def __init__(self, case_id_col):
        self.n_states = 0
        self.case_id_col = case_id_col
        
    
    def fit(self, X, y=None):
        
        sizes = X.groupby(self.case_id_col).size()
        self.n_states = sizes.unique()
        
        return self
    
    
    def predict(self, X, y=None):
        
        return X.groupby(self.case_id_col).size().as_matrix()
    
    
    def fit_predict(self, X, y=None):
        
        self.fit(X)
        return self.predict(X)