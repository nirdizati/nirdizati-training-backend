import os
import pickle
import sys
from sys import argv

import numpy as np
import pandas as pd

from DatasetManager import DatasetManager

test_file = argv[1]
pickle_model = argv[2]

logs_dir = "../logdata/"
pickles_dir = "../pkl/"

# read in pickle file with predictive model and metadata
with open(os.path.join(pickles_dir, '%s' % pickle_model), 'rb') as f:
    pipelines = pickle.load(f)
    bucketer = pickle.load(f)
    dataset_ref = pickle.load(f)
    label_col = pickle.load(f)

##### MAIN PART ######

dataset_manager = DatasetManager(dataset_ref, label_col)
dtypes = {col: "object" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
          [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
    dtypes[col] = "float"

if dataset_manager.mode == "regr":
    dtypes[dataset_manager.label_col] = "float"  # if regression, target value is float
else:
    dtypes[dataset_manager.label_col] = "object"  # if classification, preserve and do not interpret dtype of label

#test = pd.read_csv(os.path.join(logs_dir, test_file), sep=";", dtype=dtypes)
test = pd.read_json(os.path.join(logs_dir, test_file), orient='records', dtype=dtypes)
test[dataset_manager.timestamp_col] = pd.to_datetime(test[dataset_manager.timestamp_col])

# get bucket for the test case
bucket = np.asscalar(bucketer.predict(test))

# select relevant classifier
if bucket not in pipelines:  # TODO fix this
    sys.exit("No matching model has been trained!")

else:
    # make actual predictions
    preds = pipelines[bucket].predict_proba(test)
    preds = np.around(preds, decimals=3)
    preds = max(0, np.asscalar(preds))  # if remaining time is predicted to be negative, make it zero
    print(preds)
