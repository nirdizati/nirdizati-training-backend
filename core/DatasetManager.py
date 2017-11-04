import sys
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold

home_dir = os.environ['PYTHONPATH']
dataset_params_dir = os.path.join(home_dir, "core/dataset_params/")

class DatasetManager:
    
    def __init__(self, dataset_name, label_col):
        self.dataset_name = dataset_name
        self.label_col = label_col

        dataset_params = pd.read_json(os.path.join(dataset_params_dir, "%s.json" % self.dataset_name), orient="index", typ="series")

        self.case_id_col = dataset_params[u'case_id_col']
        self.activity_col = dataset_params[u'activity_col']
        self.timestamp_col = dataset_params[u'timestamp_col']

        self.dynamic_cat_cols = dataset_params[u'dynamic_cat_cols']
        self.static_cat_cols = dataset_params[u'static_cat_cols']
        self.dynamic_num_cols = dataset_params[u'dynamic_num_cols']
        self.static_num_cols = dataset_params[u'static_num_cols']

        # attributes that generally become known only after case completes, should not be used for predictions
        self.label_cat_cols = dataset_params[u'label_cat_cols']
        self.label_num_cols = dataset_params[u'label_num_cols']

        if label_col in self.label_cat_cols:
            print("Your prediction target is categorical, classification will be applied")
            self.mode = "class"
            self.pos_label = "true"
        elif label_col in self.label_num_cols:
            print("Your prediction target is numeric, regression will be applied")
            self.mode = "regr"
        else:
            sys.exit("This label column is undefined in dataset params")


    def split_data(self, data, train_ratio):  
        # split into train and test using temporal split

        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')

        return (train, test)


    def generate_prefix_data(self, data, min_length, max_length):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
        for nr_events in range(min_length+1, max_length+1):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
        dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id_col)[self.activity_col].transform(len)
        
        return dt_prefixes


    def get_pos_case_length_quantile(self, data, quantile=0.90):
        if self.mode == "regr":
            return int(np.ceil(data.groupby(self.case_id_col).size().quantile(quantile)))
        else:
            return int(np.ceil(data[data[self.label_col]==self.pos_label].groupby(self.case_id_col).size().quantile(quantile)))


    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data):
        if self.mode == "regr":
            return data.groupby(self.case_id_col).min()[self.label_col]
        else:
            return data.groupby(self.case_id_col).first()[self.label_col]
    
    def get_label_numeric(self, data):
        y = self.get_label(data) # one row per case
        if self.mode == "regr":
            return y
        elif self.mode == "class":
            return [1 if label == self.pos_label else 0 for label in y]
        else:
            print("Unrecognized training mode")
            return None
    
    def get_class_ratio(self, data):
        class_freqs = data[self.label_col].value_counts()
        return class_freqs[self.pos_label] / class_freqs.sum()
    
    def get_stratified_split_generator(self, data, n_splits=5, shuffle=True, random_state=22):
        grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
        skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state) if self.mode == "regr" else StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
            current_train_names = grouped_firsts[self.case_id_col][train_index]
            train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            yield (train_chunk, test_chunk)
