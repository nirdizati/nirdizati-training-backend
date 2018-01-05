# implementation based on https://github.com/irhete/predictive-monitoring-benchmark and https://github.com/nirdizati/nirdizati-training-backend

import pandas as pd
import numpy as np
import sys

class ClusterBasedBucketer(object):
    
    def __init__(self, encoder, clustering):
        self.encoder = encoder
        self.clustering = clustering
        
    
    def fit(self, X, y=None):
        
        dt_encoded = self.encoder.fit_transform(X)
        
        self.clustering.fit(dt_encoded)
        
        return self
    
    
    def predict(self, X, y=None):
        
        dt_encoded = self.encoder.transform(X)
        
        return self.clustering.predict(dt_encoded)
    
    
    def fit_predict(self, X, y=None):
        
        self.fit(X)
        return self.predict(X)