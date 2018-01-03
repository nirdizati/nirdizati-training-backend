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

detailed_results_file = "results_%s_%s.csv" % (os.path.basename(test_file), dataset_manager.label_col)

##### MAIN PART ######

dtypes = {col: "str" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
          [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
    dtypes[col] = "float"

# if dataset_manager.mode == "regr":
#     dtypes[dataset_manager.label_col] = "float"  # if regression, target value is float
# else:
#     dtypes[dataset_manager.label_col] = "str"  # if classification, preserve and do not interpret dtype of label

test = pd.read_csv(test_file, sep=";", dtype=dtypes)
#test = test.drop(label_col, axis = 1)
test[dataset_manager.timestamp_col] = pd.to_datetime(test[dataset_manager.timestamp_col])

# get bucket for each test case
bucket_assignments_test = bucketer.predict(test)

detailed_results = pd.DataFrame()

# use appropriate classifier for each bucket of test cases
for bucket in set(bucket_assignments_test):
    relevant_cases_bucket = dataset_manager.get_indexes(test)[bucket_assignments_test == bucket]
    dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(test, relevant_cases_bucket)

    if len(relevant_cases_bucket) == 0:
        continue

    elif bucket not in pipelines:
        sys.exit("No matching model has been trained!")

    else:
        # make actual predictions
        preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)

    preds_bucket = preds_bucket.clip(min=0)  # if remaining time is predicted to be negative, make it zero

    if preds_bucket.shape[1] > 1:  # classification
        preds_bucket = pipelines[bucket]._final_estimator.cls.classes_[preds_bucket.argmax(axis=1)]

    case_ids = list(dt_test_bucket.groupby(dataset_manager.case_id_col).first().index)
    current_results = pd.DataFrame({"%s"%dataset_manager.label_col: preds_bucket, "%s"%dataset_manager.case_id_col: case_ids})
    detailed_results = pd.concat([detailed_results, current_results])

detailed_results.to_csv(detailed_results_file, sep=",", index=False)
