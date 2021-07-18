import os
import pickle
import sys
from sys import argv

import numpy as np
import pandas as pd


test_file = argv[1]
pickle_model = argv[2]

# logs_dir = "../logdata/"
# pickles_dir = "../pkl/"

# read in pickle file with predictive model and metadata
with open(pickle_model, 'rb') as f:
    pipelines = pickle.load(f)
    bucketer = pickle.load(f)
    dataset_manager = pickle.load(f)

##### MAIN PART ######

dtypes = {col: "str" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
          [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
    dtypes[col] = "float"

# if dataset_manager.mode == "regr":
#     dtypes[dataset_manager.label_col] = "float"  # if regression, target value is float
# else:
#     dtypes[dataset_manager.label_col] = "str"  # if classification, preserve and do not interpret dtype of label

test = pd.read_json(test_file, orient='records', dtype=dtypes)
#test = test.drop(label_col, axis = 1)
test[dataset_manager.timestamp_col] = pd.to_datetime(test[dataset_manager.timestamp_col])

# get bucket for the test case
bucket = bucketer.predict(test).item()

# select relevant classifier
if bucket not in pipelines:  # TODO fix this
    sys.exit("No matching model has been trained!")

else:
    # make actual predictions
    preds = pipelines[bucket].predict_proba(test)
    if preds.ndim == 1:  #regression
        preds = pd.DataFrame(preds.clip(min=0), columns=[dataset_manager.label_col])

    preds = preds.to_json(orient='records')
    print(preds)
