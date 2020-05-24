import json
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold

home_dirs = os.environ['PYTHONPATH'].split(":")
home_dir = home_dirs[0]

dataset_params_dir = Path.cwd().parent / "core/dataset_params/"
#dataset_params_dir = os.path.join(home_dir, "core/dataset_params/")
unique_values_threshold = 10  # threshold to distinguish classification vs regression

class DatasetManager:
    
    def __init__(self, dataset_name, label_col):
        self.dataset_name = dataset_name
        self.label_col = label_col

        with open(dataset_params_dir / ("%s.json" % self.dataset_name)) as f:
            dataset_params = json.load(f)
            
        self.case_id_col = dataset_params['case_id_col']
        self.activity_col = dataset_params['activity_col']
        self.timestamp_col = dataset_params['timestamp_col']

        # define features for predictions
        predictor_cols = ["dynamic_cat_cols", "static_cat_cols", "dynamic_num_cols", "static_num_cols"]
        for predictor_col in predictor_cols:
            if label_col in dataset_params[predictor_col]:
                print("%s found in %s, it will be removed (not a feature)" % (label_col, predictor_col))
                dataset_params[predictor_col].remove(label_col)  # exclude label attributes from features
            setattr(self, predictor_col, dataset_params[predictor_col])

        if self.activity_col not in self.dynamic_cat_cols:
            self.dynamic_cat_cols += [self.activity_col]

    def determine_mode(self, data):
        if data[self.label_col].nunique() < unique_values_threshold:
            print("less than %s unique values in target variable, classification will be applied" % unique_values_threshold)
            mode = "class"
        else:
            try:
                data[self.label_col].astype(float)
                print("%s or more unique numeric values, regression will be applied" % unique_values_threshold)
                mode = "regr"
            except ValueError:
                print("%s or more unique categorical values, classification will be applied" % unique_values_threshold)
                mode = "class"
        return mode

    def add_remtime(self, group):
        group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        end_date = group[self.timestamp_col].iloc[-1]
        tmp = end_date - group[self.timestamp_col]
        tmp = tmp.fillna(pd.Timedelta(seconds=0))
        group["remtime"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 's')))  # 's' is for seconds
        return group

    def get_mean_case_duration(self, data):
        case_durations = data.groupby(self.case_id_col)['remtime'].max()
        return np.mean(case_durations)

    def assign_label(self, group, threshold):
        group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        case_duration = group["remtime"].iloc[0]
        group[self.label_col] = "false" if case_duration < threshold else "true"
        return group

    def get_next_activity(self, group):
        group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        group[self.label_col] = group[self.activity_col].shift(-1)
        group[self.label_col] = group[self.label_col].fillna("PROCESS_END")
        return group

    def split_data(self, data, train_ratio):
        # split into train and test using temporal split

        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')

        return (train, test)


    def generate_prefix_data(self, data, min_length, max_length, comparator, gap=1):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        dt_prefixes = data[comparator(data['case_length'], min_length)].groupby(self.case_id_col).head(min_length)
        for nr_events in range(min_length+gap, max_length+1, gap):
            tmp = data[comparator(data['case_length'], nr_events)].groupby(self.case_id_col).head(nr_events)
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
        dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id_col)[self.activity_col].transform(len)
        
        return dt_prefixes


    def get_case_length_quantile(self, data, quantile=0.90):
        return int(np.floor(data.groupby(self.case_id_col).size().quantile(quantile)))


    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data, mode):
        if self.label_col == "remtime":
            # remtime is a dynamic label (changes with each executed event), take the latest (smallest) value
            return data.groupby(self.case_id_col).min()[self.label_col]
        else:
            # static labels - take any value throughout the case (e.g. the last one)
            return data.groupby(self.case_id_col).last()[self.label_col]
    
    def get_class_ratio(self, data):
        class_freqs = data[self.label_col].value_counts()
        return class_freqs / class_freqs.sum()
    
    def get_stratified_split_generator(self, data, mode, n_splits=5, shuffle=True, random_state=22):
        grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
        skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state) if mode == "regr" else StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
            current_train_names = grouped_firsts[self.case_id_col][train_index]
            train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            yield (train_chunk, test_chunk)
