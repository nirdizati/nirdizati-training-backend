import os
import pickle
from sys import argv

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion

import BucketFactory
import ClassifierFactory
import EncoderFactory
from DatasetManager import DatasetManager

train_file = argv[1]
bucket_encoding = "agg"
bucket_method = argv[2]
cls_encoding = argv[3]
cls_method = argv[4]
label_col = argv[5]

dataset_ref = os.path.splitext(train_file)[0]
home_dir = ""
logs_dir = "../logdata/"
training_params_dir = "training_params/"
results_dir = "../results/"
feature_importance_dir = "../results/feature_importance"
pickles_dir = "../pkl/"

best_params = pd.read_json(os.path.join(home_dir, training_params_dir, "%s.json" % dataset_ref), typ="series")

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}

method_name = "%s_%s" % (bucket_method, cls_encoding)
methods = encoding_dict[cls_encoding]

outfile = os.path.join(home_dir, results_dir,
                       "validation_results_%s_%s_%s_%s.csv" % (dataset_ref, method_name, cls_method, label_col))

pickle_file = os.path.join(pickles_dir, '%s_%s_%s_%s.pkl' % (dataset_ref, method_name, cls_method, label_col))

random_state = 22
fillna = True
n_min_cases_in_bucket = 30

##### MAIN PART ######    
with open(outfile, 'w') as fout:
    fout.write("%s,%s,%s,%s,%s,%s\n" % ("label_col", "method", "cls", "nr_events", "metric", "score"))

    dataset_manager = DatasetManager(dataset_ref, label_col)
    dtypes = {col: "str" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
              [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
    for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
        dtypes[col] = "float"

    if dataset_manager.mode == "regr":
        dtypes[dataset_manager.label_col] = "float"  # if regression, target value is float
    else:
        dtypes[dataset_manager.label_col] = "str"  # if classification, preserve and do not interpret dtype of label

    data = pd.read_csv(os.path.join(logs_dir, train_file), sep=";", dtype=dtypes)
    data[dataset_manager.timestamp_col] = pd.to_datetime(data[dataset_manager.timestamp_col])

    # split data into training and validation sets
    train, test = dataset_manager.split_data(data, train_ratio=0.80)
    # train = train.sort_values(dataset_manager.timestamp_col, ascending=True, kind='mergesort')

    # consider prefix lengths until 90th percentile of case length
    min_prefix_length = 1
    max_prefix_length = min(15, dataset_manager.get_pos_case_length_quantile(data, 0.9))
    del data

    # create prefix logs
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

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
        bucketer_args['n_clusters'] = best_params[label_col][method_name][cls_method]['n_clusters']

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
            cls_args = {k: v for k, v in best_params[label_col][method_name][cls_method][u'%s' % bucket].items() if
                        k not in ['n_clusters']}
        else:
            cls_args = {k: v for k, v in best_params[label_col][method_name][cls_method].items() if
                        k not in ['n_clusters']}
        cls_args['mode'] = dataset_manager.mode
        cls_args['random_state'] = random_state
        cls_args['min_cases_for_training'] = n_min_cases_in_bucket

        # select relevant cases
        relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
        dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                       relevant_cases_bucket)  # one row per event
        train_y = dataset_manager.get_label_numeric(dt_train_bucket)

        feature_combiner = FeatureUnion(
            [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
        pipelines[bucket] = Pipeline(
            [('encoder', feature_combiner), ('cls', ClassifierFactory.get_classifier(cls_method, **cls_args))])

        pipelines[bucket].fit(dt_train_bucket, train_y)

        feature_set = []
        for feature_set_this_encoding in pipelines[bucket].steps[0][1].transformer_list:
            for feature in feature_set_this_encoding[1].columns.tolist():
                feature_set.append(feature)

        feats = {}  # a dict to hold feature_name: feature_importance
        for feature, importance in zip(feature_set, pipelines[bucket].named_steps.cls.cls.feature_importances_):
            feats[feature] = importance  # add the name/value pair

        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
        importances = importances.sort_values(by='Gini-importance', ascending=False)
        importances.to_csv(os.path.join(home_dir, feature_importance_dir, "feat_importance_%s_%s_%s_%s_%s.csv" %
                                        (dataset_ref, method_name, cls_method, label_col, bucket)))

    with open(pickle_file, 'wb') as f:
        pickle.dump(pipelines, f)
        pickle.dump(bucketer, f)
        pickle.dump(dataset_manager, f)

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
        preds = []
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
                avg_target_value = [np.mean(train["remtime"])] if dataset_manager.mode == "regr" else [
                    dataset_manager.get_class_ratio(train)]
                preds_bucket = avg_target_value * len(relevant_cases_bucket)

            else:
                # make actual predictions
                preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)

            preds.extend(preds_bucket)

            # extract actual label values
            test_y_bucket = dataset_manager.get_label_numeric(dt_test_bucket)  # one row per case
            test_y.extend(test_y_bucket)

        score = {}
        if len(set(test_y)) < 2:
            score = {"score1": 0, "score2": 0}
        elif dataset_manager.mode == "regr":
            score["mae"] = mean_absolute_error(test_y, preds)
            score["r2"] = r2_score(test_y, preds)
        else:
            score["auc"] = roc_auc_score(test_y, preds)
            _, _, score["fscore"], _ = precision_recall_fscore_support(test_y,
                                                                       [0 if pred < 0.5 else 1 for pred in preds],
                                                                       average="binary")

        fout.write("%s,%s,%s,%s,%s,%s\n" % (label_col, method_name, cls_method, nr_events,
                                            list(score)[0], list(score.values())[0]))
        fout.write("%s,%s,%s,%s,%s,%s\n" % (label_col, method_name, cls_method, nr_events,
                                            list(score)[1], list(score.values())[1]))

    print("\n")
