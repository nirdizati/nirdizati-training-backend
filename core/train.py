import json
import os
import pickle
import sys
import operator
from pathlib import Path

import numpy as np
from numpy import array
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion

import BucketFactory
import ClassifierFactory
import EncoderFactory
from DatasetManager import DatasetManager
import evaluation


config_file = sys.argv[1]
bucket_encoding = "agg"
home_dirs = os.environ['PYTHONPATH'].split(":")
#home_dir = home_dirs[0] # if there are multiple PYTHONPATHs, choose the first
training_params_dir =  Path("core/training_params/")
results_dir = Path("results/validation/")
detailed_results_dir = Path("results/detailed/")
feature_importance_dir = Path("results/feature_importance/")
pickles_dir = Path("pkl/")

path_to_open = Path.cwd().parent / training_params_dir / ("%s.json" % config_file)
with open(path_to_open) as f:
    config = json.load(f)

train_file = config["ui_data"]["log_file"]
dataset_ref = os.path.splitext(os.path.basename(train_file))[0]

for k, v in config.items():
    if k not in ['ui_data'] and k not in ['evaluation']:
        label_col=k
        for k1, v1 in v.items():
            bucket_method = k1
            for k2, v2 in v1.items():
                cls_encoding=k2
                for k3, v3 in v2.items():
                    cls_method=k3

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}

methods = encoding_dict[cls_encoding]

pickle_file = Path.cwd().parent / pickles_dir /  ('%s_%s.pkl' % (dataset_ref, config_file))
#pickle_file = os.path.join(home_dir, pickles_dir, '%s_%s.pkl' % (dataset_ref, config_file))

random_state = 22
fillna = True
n_min_cases_in_bucket = 30


##### MAIN PART ######
dataset_manager = DatasetManager(dataset_ref, label_col)
dtypes = {col: "str" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
          [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
    dtypes[col] = "float"

data = pd.read_csv(train_file, sep=",", dtype=dtypes)

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
        data = data.groupby(dataset_manager.case_id_col, as_index=False).apply(dataset_manager.assign_label, mean_case_duration)
    elif threshold > 0:
        # prediction of a label wrt arbitrary threshold on case duration
        data = data.groupby(dataset_manager.case_id_col, as_index=False).apply(dataset_manager.assign_label, threshold)
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

outfile = Path.cwd().parent / results_dir / ("validation_%s_%s_%s.csv" % (dataset_ref, config_file, mode))
#outfile = os.path.join(home_dir, results_dir,
#                       "validation_%s_%s_%s.csv" % (dataset_ref, config_file, mode))

detailed_results_file = Path.cwd().parent / detailed_results_dir / \
                        ("detailed_%s_%s_%s.csv" % (dataset_ref, config_file, mode))
#detailed_results_file = os.path.join(home_dir, detailed_results_dir,
#                       "detailed_%s_%s_%s.csv" % (dataset_ref, config_file, mode))
detailed_results = pd.DataFrame()

with open(str(outfile), 'w') as fout:
    fout.write("%s,%s,%s,%s,%s,%s,%s\n" % ("label_col", "bucket_method", "feat_encoding", "cls", "nr_events", "metric", "score"))

    # split data into training and test sets
    train, test = dataset_manager.split_data(data, train_ratio=0.80)

    # consider prefix lengths until 90th percentile of case length
    min_prefix_length = 1
    max_prefix_length = min(25, dataset_manager.get_case_length_quantile(data, 0.9))
    del data

    # create prefix logs
    # cases that have just finished are included in the training set, but not in the test
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length, comparator=operator.ge, gap=2)
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length, comparator=operator.gt)

    print(dt_train_prefixes.shape)
    print(dt_test_prefixes.shape)

    # extract arguments
    bucketer_args = {'encoding_method': bucket_encoding,
                     'case_id_col': dataset_manager.case_id_col,
                     'cat_cols': [dataset_manager.activity_col],
                     'num_cols': [],
                     'n_clusters': None,
                     'random_state': random_state}
    if bucket_method == "cluster":
        bucketer_args['n_clusters'] = config[label_col][bucket_method][cls_encoding][cls_method]['n_clusters']

    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols,
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                        'fillna': fillna}

    # Bucketing prefixes based on control flow
    print("Bucketing prefixes...")
    bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
    bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)

    pipelines = {}

    # train and fit pipeline for each bucket
    for bucket in set(bucket_assignments_train):
        print("Fitting pipeline for bucket %s..." % bucket)

        # set optimal params for this bucket
        if bucket_method == "prefix":
            cls_args = {k: v for k, v in config[label_col][bucket_method][cls_encoding][cls_method][u'%s' % bucket].items() if
                        k not in ['n_clusters']}
        else:
            cls_args = {k: v for k, v in config[label_col][bucket_method][cls_encoding][cls_method].items() if
                        k not in ['n_clusters']}
        cls_args['mode'] = mode
        cls_args['random_state'] = random_state
        cls_args['min_cases_for_training'] = n_min_cases_in_bucket

        # select relevant cases
        relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
        dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                       relevant_cases_bucket)  # one row per event
        train_y = dataset_manager.get_label(dt_train_bucket, mode=mode)

        feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
        pipelines[bucket] = Pipeline([('encoder', feature_combiner), ('cls', ClassifierFactory.get_classifier(cls_method, **cls_args))])

        pipelines[bucket].fit(dt_train_bucket, train_y)

        if pipelines[bucket].named_steps.cls.hardcoded_prediction is not None:
            print("Hardcoded predictions were used in bucket %s, no feature importance available" % bucket)
            continue

        if np.isnan(np.sum(pipelines[bucket].named_steps.cls.cls.feature_importances_)):
            print("No Feature importance available for bucket %d" % bucket)
            continue

        feature_set = []
        for feature_set_this_encoding in pipelines[bucket].steps[0][1].transformer_list:
            for feature in feature_set_this_encoding[1].columns.tolist():
                feature_set.append(feature)

        feats = {}  # a dict to hold feature_name: feature_importance
        for feature, importance in zip(feature_set, pipelines[bucket].named_steps.cls.cls.feature_importances_):
            feats[feature] = importance  # add the name/value pair

        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
        importances = importances.sort_values(by='Gini-importance', ascending=False)
        importances.head(20).to_csv(Path.cwd().parent / feature_importance_dir / ("feat_importance_%s_%s_%s.csv" %
                                                 (dataset_ref, config_file, bucket)))

#        importances.head(20).to_csv(os.path.join(home_dir, feature_importance_dir, "feat_importance_%s_%s_%s.csv" %
 #                                       (dataset_ref, config_file, bucket)))

    with open(str(pickle_file), 'wb') as f:
        pickle.dump(pipelines, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(bucketer, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset_manager, f, protocol=pickle.HIGHEST_PROTOCOL)

    prefix_lengths_test = dt_test_prefixes.groupby(dataset_manager.case_id_col).size()

    # test separately for each prefix length
    for nr_events in range(min_prefix_length, max_prefix_length + 1):
        print("Predicting for %s events..." % nr_events)

        # select only cases that are at least of length nr_events
        relevant_cases_nr_events = prefix_lengths_test[prefix_lengths_test == nr_events].index

        if len(relevant_cases_nr_events) == 0:
            break

        dt_test_nr_events = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_cases_nr_events)
        del relevant_cases_nr_events

        # get predicted cluster for each test case
        bucket_assignments_test = bucketer.predict(dt_test_nr_events)

        # use appropriate classifier for each bucket of test cases
        # for evaluation, collect predictions from different buckets together
        preds = [] if mode == "regr" else pd.DataFrame()
        test_y = []
        for bucket in set(bucket_assignments_test):
            relevant_cases_bucket = dataset_manager.get_indexes(dt_test_nr_events)[bucket_assignments_test == bucket]
            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_nr_events,
                                                                          relevant_cases_bucket)  # one row per event

            if len(relevant_cases_bucket) == 0:
                continue

            elif bucket not in pipelines:
                # regression - use mean value (in training set) as prediction
                # classification - use the historical class ratio
                if mode == "regr":
                    avg_target_value = [np.mean(train[dataset_manager.label_col])]
                    preds_bucket = array(avg_target_value * len(relevant_cases_bucket))
                else:
                    avg_target_value = [dataset_manager.get_class_ratio(train)]
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
            test_y_bucket = dataset_manager.get_label(dt_test_bucket, mode=mode)  # one row per case
            test_y.extend(test_y_bucket)

            if nr_events == 1:
                continue
            elif mode == "class":
                preds_bucket = preds_bucket.idxmax(axis=1)

            case_ids = list(dt_test_bucket.groupby(dataset_manager.case_id_col).first().index)
            current_results = pd.DataFrame({"label_col": label_col, "bucket_method": bucket_method, "feat_encoding": cls_encoding, "cls": cls_method, "nr_events": nr_events, "predicted": preds_bucket, "actual": test_y_bucket.values, "case_id": case_ids})
            detailed_results = pd.concat([detailed_results, current_results])

        # get average scores for this prefix length
        score = evaluation.get_score(test_y, preds, mode=mode)
        for k, v in score.items():
            fout.write("%s,%s,%s,%s,%s,%s,%s\n" % (label_col, bucket_method, cls_encoding, cls_method, nr_events, k, v))


    # get average scores across all evaluated prefix lengths
    config["evaluation"] = evaluation.get_agg_score(detailed_results.actual, detailed_results.predicted, mode=mode)
    with open(Path.cwd().parent / training_params_dir / ("%s.json" % config_file), 'w') as f:
        json.dump(config, f)

    print("\n")

if mode == "class":
    confusion_matrix = pd.crosstab(detailed_results.actual, detailed_results.predicted, rownames=['Actual'], colnames=['Predicted'], margins=False)
    confusion_matrix.to_csv(detailed_results_file, sep=",")
else:
    detailed_results.to_csv(detailed_results_file, sep=",", index=False)
