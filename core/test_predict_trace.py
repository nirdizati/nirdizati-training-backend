import unittest

import pickle
import sys

import numpy as np
import pandas as pd


class TestBpi17Predictions(unittest.TestCase):

    def test_remtime(self):
        test_file = "../logdata/1170931347.json"
        pickle_model = "../pkl/bpi17_sample_myconfig_remtime.pkl"

        # read in pickle file with predictive model and metadata
        with open(pickle_model, 'rb') as f:
            pipelines = pickle.load(f)
            bucketer = pickle.load(f)
            dataset_manager = pickle.load(f)

        dtypes = {col: "str" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
                  [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
        for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
            dtypes[col] = "float"

        test = pd.read_json(test_file, orient='records', dtype=dtypes)
        test[dataset_manager.timestamp_col] = pd.to_datetime(test[dataset_manager.timestamp_col])

        # get bucket for the test case
        bucket = bucketer.predict(test).item()

        # select relevant classifier and make prediction
        if bucket not in pipelines:  # TODO fix this
            sys.exit("No matching model has been trained!")

        else:
            # make actual predictions
            preds = pipelines[bucket].predict_proba(test)
            preds = preds.clip(min=0)[0]

        self.assertAlmostEqual(preds, 15.96, delta=0.01)

    def test_label(self):
        test_file = "../logdata/1170931347.json"
        pickle_model = "../pkl/bpi17_sample_myconfig_label.pkl"

        # read in pickle file with predictive model and metadata
        with open(pickle_model, 'rb') as f:
            pipelines = pickle.load(f)
            bucketer = pickle.load(f)
            dataset_manager = pickle.load(f)

        dtypes = {col: "str" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
                  [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
        for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
            dtypes[col] = "float"

        test = pd.read_json(test_file, orient='records', dtype=dtypes)
        test[dataset_manager.timestamp_col] = pd.to_datetime(test[dataset_manager.timestamp_col])

        # get bucket for the test case
        bucket = bucketer.predict(test).item()

        # select relevant classifier and make prediction
        if bucket not in pipelines:  # TODO fix this
            sys.exit("No matching model has been trained!")

        else:
            # make actual predictions
            preds = pipelines[bucket].predict_proba(test)
            preds = preds[False][0]

        self.assertAlmostEqual(preds, 0.79, delta=0.01)


if __name__ == '__main__':
    unittest.main()
