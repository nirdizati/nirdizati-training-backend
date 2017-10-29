import pickle
import sys
from sys import argv

import numpy as np
import pandas as pd

from DatasetManager import DatasetManager

test_file = argv[1]
dataset_ref = argv[2]
bucket_encoding = "agg"
bucket_method = argv[3]
cls_encoding = argv[4]
cls_method = argv[5]

##### MAIN PART ######

dataset_manager = DatasetManager(dataset_ref)
dtypes = {col: "object" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
          [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
    dtypes[col] = "float"

for col in dataset_manager.label_col:
    dtypes[col] = "float"

test = pd.read_csv(test_file, sep=";", dtype=dtypes)
test[dataset_manager.timestamp_col] = pd.to_datetime(test[dataset_manager.timestamp_col])

# read in pickle file with predictive model and bucketer
with open('remtime_%s_%s_%s_%s.pkl' % (bucket_method, cls_encoding, cls_method, dataset_ref), 'rb') as f:
    pipelines = pickle.load(f)
    bucketer = pickle.load(f)

# get bucket for the test case
bucket = np.asscalar(bucketer.predict(test))

# select relevant classifier
if bucket not in pipelines:  # TODO fix this
    sys.exit("No matching model has been trained!")

else:
    # make actual predictions
    preds = pipelines[bucket].predict_proba(test)
    preds = max(0, np.rint(np.asscalar(preds)))  # if remaining time is predicted to be negative, make it zero
print (preds)
