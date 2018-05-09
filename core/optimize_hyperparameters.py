import os
import itertools
import sys
import operator
from pathlib import Path

import numpy as np
from numpy import array
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.pipeline import Pipeline, FeatureUnion

import BucketFactory
import ClassifierFactory
import EncoderFactory
from DatasetManager import DatasetManager

train_file = sys.argv[1]
bucket_encoding = "agg"
bucket_method = sys.argv[2]
cls_encoding = sys.argv[3]
cls_method = sys.argv[4]
label_col = sys.argv[5]

dataset_ref = os.path.splitext(train_file)[0]
home_dirs = os.environ['PYTHONPATH'].split(":")
home_dir = home_dirs[0]
logs_dir = "logdata/"
results_dir = "results/CV/"

if not os.path.exists(Path.cwd().parent / results_dir):
    os.makedirs(Path.cwd().parent / results_dir)
# if not os.path.exists(os.path.join(home_dir, results_dir)):
#     os.makedirs(os.path.join(home_dir, results_dir))


encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}

method_name = "%s_%s" % (bucket_method, cls_encoding)
methods = encoding_dict[cls_encoding]

# bucketing params to optimize 
if bucket_method == "cluster":
    bucketer_params = {'n_clusters':[2, 5, 10, 20]}
else:
    bucketer_params = {'n_clusters':[1]}

# classification params to optimize
if cls_method == "rf":
    cls_params = {'n_estimators':[100],
                  'max_features':["sqrt", 0.1, 0.5]}

elif cls_method == "gbm":
    cls_params = {'n_estimators':[100],
                  'max_features':["sqrt", 0.25],
                  'learning_rate':[0.1, 0.2]}

elif cls_method == "dt":
    cls_params = {'max_features':[0.1, 0.6, 0.9],
                  'max_depth':[5, 10, 20]}

elif cls_method == "xgb":
    cls_params = {'n_estimators':[300, 500],
                  'learning_rate':[0.02, 0.04, 0.06],
                  'subsample': [0.5, 0.8],
                  'max_depth': [2,4,6],
                  'colsample_bytree':[0.5, 0.8]}

bucketer_params_names = list(bucketer_params.keys())
cls_params_names = list(cls_params.keys())


outfile = Path.cwd().parent / results_dir / ("CV_%s_%s_%s_%s.csv" %
                                             (dataset_ref, method_name, cls_method, label_col))

# outfile = os.path.join(home_dir, results_dir, "CV_%s_%s_%s_%s.csv"%(dataset_ref, method_name, cls_method, label_col))


train_ratio = 0.8
random_state = 22
fillna = True
n_min_cases_in_bucket = 30


##### MAIN PART ######    
with open(outfile, 'w') as fout:

    fout.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%("part", "label_col", "method", "cls", ",".join(bucketer_params_names), ",".join(cls_params_names), "nr_events", "metric", "score"))

    dataset_manager = DatasetManager(dataset_ref, label_col)

    # read the data
    dtypes = {col: "str" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
              [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
    for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
        dtypes[col] = "float"

    # if dataset_manager.mode == "regr":
    #     dtypes[dataset_manager.label_col] = "float" # if regression, target value is float
    # else:
    #     dtypes[dataset_manager.label_col] = "str" # if classification, preserve and do not interpret dtype of label

    data = pd.read_csv(Path.cwd().parent / logs_dir / train_file, sep=",|;", dtype=dtypes, engine="python")
    # data = pd.read_csv(os.path.join(home_dir, logs_dir, train_file), sep=",|;", dtype=dtypes, engine="python")
    #data = data.tail(10000)
    data[dataset_manager.timestamp_col] = pd.to_datetime(data[dataset_manager.timestamp_col])

    # add remaining time column to the dataset if it does not exist yet
    if "remtime" not in data.columns:
        print("Remaining time column is not found, will be added now")
        data = data.groupby(dataset_manager.case_id_col, as_index=False).apply(dataset_manager.add_remtime)

    try:
        threshold = float(label_col)
        mode = "class"
        if threshold == -1:
            # prediction of a label wrt mean case duration
            mean_case_duration = dataset_manager.get_mean_case_duration(data)
            data = data.groupby(dataset_manager.case_id_col, as_index=False).apply(dataset_manager.assign_label,
                                                                                   mean_case_duration)
        elif threshold > 0:
            # prediction of a label wrt arbitrary threshold on case duration
            data = data.groupby(dataset_manager.case_id_col, as_index=False).apply(dataset_manager.assign_label,
                                                                                   threshold)
        else:
            sys.exit("Wrong value for case duration threshold")

    except ValueError:
        if label_col == "remtime":  # prediction of remaining time
            mode = "regr"
        elif label_col == "next":  # prediction of the next activity
            mode = "class"
            data = data.groupby(dataset_manager.case_id_col, as_index=False).apply(dataset_manager.get_next_activity)
        elif label_col in data.columns:  # prediction of existing column
            mode = dataset_manager.determine_mode(data)
        else:
            sys.exit("Undefined target variable")

    # split data into train and validation
    train, _ = dataset_manager.split_data(data, train_ratio)

    # consider prefix lengths until 90% of positive cases have finished
    min_prefix_length = 1
    max_prefix_length = min(25, dataset_manager.get_case_length_quantile(data, 0.90))
    del data

    part = 0
    for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, mode, n_splits=5):
        part += 1
        print("Starting chunk %s..."%part)
        sys.stdout.flush()

        # create prefix logs
        dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, min_prefix_length, max_prefix_length, comparator=operator.ge)
        dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length, comparator=operator.gt)

        print(dt_train_prefixes.shape)
        print(dt_test_prefixes.shape)


        for bucketer_params_combo in itertools.product(*(bucketer_params.values())):
            for cls_params_combo in itertools.product(*(cls_params.values())):
                print("Bucketer params are: %s"%str(bucketer_params_combo))
                print("Cls params are: %s"%str(cls_params_combo))

                # extract arguments
                bucketer_args = {'encoding_method':bucket_encoding,
                                 'case_id_col':dataset_manager.case_id_col,
                                 'cat_cols':[dataset_manager.activity_col],
                                 'num_cols':[],
                                 'random_state':random_state}
                for i in range(len(bucketer_params_names)):
                    bucketer_args[bucketer_params_names[i]] = bucketer_params_combo[i]

                cls_encoder_args = {'case_id_col':dataset_manager.case_id_col,
                                    'static_cat_cols':dataset_manager.static_cat_cols,
                                    'static_num_cols':dataset_manager.static_num_cols,
                                    'dynamic_cat_cols':dataset_manager.dynamic_cat_cols,
                                    'dynamic_num_cols':dataset_manager.dynamic_num_cols,
                                    'fillna':fillna}

                cls_args = {'mode':mode,
                            'random_state':random_state,
                            'min_cases_for_training':n_min_cases_in_bucket}
                for i in range(len(cls_params_names)):
                    cls_args[cls_params_names[i]] = cls_params_combo[i]


                # Bucketing prefixes based on control flow
                print("Bucketing prefixes...")
                bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
                bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)

                pipelines = {}

                # train and fit pipeline for each bucket
                for bucket in set(bucket_assignments_train):
                    print("Fitting pipeline for bucket %s..."%bucket)
                    relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
                    dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket) # one row per event
                    train_y = dataset_manager.get_label(dt_train_bucket, mode=mode)

                    feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
                    pipelines[bucket] = Pipeline([('encoder', feature_combiner), ('cls', ClassifierFactory.get_classifier(cls_method, **cls_args))])
                    pipelines[bucket].fit(dt_train_bucket, train_y)


                # if the bucketing is prefix-length-based, then evaluate for each prefix length separately, otherwise evaluate all prefixes together
                max_evaluation_prefix_length = max_prefix_length if bucket_method == "prefix" else min_prefix_length

                prefix_lengths_test = dt_test_prefixes.groupby(dataset_manager.case_id_col).size()

                for nr_events in range(min_prefix_length, max_evaluation_prefix_length+1):
                    print("Predicting for %s events..."%nr_events)

                    if bucket_method == "prefix":
                        # select only prefixes that are of length nr_events
                        relevant_cases_nr_events = prefix_lengths_test[prefix_lengths_test == nr_events].index

                        if len(relevant_cases_nr_events) == 0:
                            break

                        dt_test_nr_events = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_cases_nr_events)
                        del relevant_cases_nr_events
                    else:
                        # evaluate on all prefixes
                        dt_test_nr_events = dt_test_prefixes.copy()

                    # get predicted cluster for each test case
                    bucket_assignments_test = bucketer.predict(dt_test_nr_events)

                    # use appropriate classifier for each bucket of test cases
                    # for evaluation, collect predictions from different buckets together
                    preds = [] if mode == "regr" else pd.DataFrame()
                    test_y = []
                    for bucket in set(bucket_assignments_test):
                        relevant_cases_bucket = dataset_manager.get_indexes(dt_test_nr_events)[bucket_assignments_test == bucket]
                        dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_nr_events, relevant_cases_bucket) # one row per event

                        if len(relevant_cases_bucket) == 0:
                            continue

                        elif bucket not in pipelines:
                            # regression - use mean value (in training set) as prediction
                            # classification - use the historical class ratio
                            if mode == "regr":
                                avg_target_value = [np.mean(train_chunk[dataset_manager.label_col])]
                                preds_bucket = array(avg_target_value * len(relevant_cases_bucket))
                            else:
                                avg_target_value = [dataset_manager.get_class_ratio(train_chunk)]
                                preds_bucket = pd.DataFrame(avg_target_value * len(relevant_cases_bucket))

                        else:
                            # make actual predictions
                            preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)

                        if mode == "regr":
                            # if remaining time is predicted to be negative, make it zero
                            preds_bucket = preds_bucket.clip(min=0)
                            preds.extend(preds_bucket)
                        else:
                            # if some label values were not present in the training set, thus are never predicted
                            classes_as_is = preds_bucket.columns
                            for class_to_be in train[dataset_manager.label_col].unique():
                                if class_to_be not in classes_as_is:
                                    preds_bucket[class_to_be] = 0
                            preds = pd.concat([preds, preds_bucket])

                        # extract actual label values
                        test_y_bucket = dataset_manager.get_label(dt_test_bucket, mode=mode) # one row per case
                        test_y.extend(test_y_bucket)

                    score = {}
                    if mode == "regr":
                        score["mae"] = mean_absolute_error(test_y, preds)
                        score["rmse"] = np.sqrt(mean_squared_error(test_y, preds))
                        score["nmae"] = score["mae"] / train[dataset_manager.label_col].mean()
                        score["nrmse"] = score["rmse"] / train[dataset_manager.label_col].mean()
                    elif len(set(test_y)) < 2:
                        score = {"acc": 0, "f1": 0, "logloss": 0}
                    else:
                        preds_labels = preds.idxmax(axis=1)
                        score["acc"] = accuracy_score(test_y, preds_labels)
                        score["f1"] = f1_score(test_y, preds_labels, average='weighted')
                        try:
                            score["logloss"] = log_loss(test_y, preds, labels=preds.columns)
                        except ValueError:
                            print("logloss cannot be calculated")

                    bucketer_params_str = ",".join([str(param) for param in bucketer_params_combo])
                    cls_params_str = ",".join([str(param) for param in cls_params_combo])

                    for k, v in score.items():
                        fout.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(part, label_col, method_name, cls_method, bucketer_params_str, cls_params_str, nr_events, k, v))

                print("\n")
