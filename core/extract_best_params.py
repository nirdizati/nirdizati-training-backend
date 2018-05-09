import glob
from pathlib import Path

import pandas as pd
import os
import numpy as np
from sys import argv
import json

train_file = argv[1]

dataset_ref = os.path.splitext(train_file)[0]
home_dirs = os.environ['PYTHONPATH'].split(":")
home_dir = home_dirs[0]
cv_results_dir = Path.cwd().parent /  "results/CV/"
#cv_results_dir = os.path.join(home_dir, "results/CV/")
training_params_dir = "core/training_params/"

# read all files from directory to df       
files = glob.glob("%s/CV_%s_*" % (cv_results_dir, dataset_ref))
files = [file for file in files if os.path.getsize(file) > 0]

data = pd.read_csv(files[0], sep=",")
for file in files[1:]:
    tmp = pd.read_csv(file, sep=",")
    data = pd.concat([data, tmp], axis=0)

# fix cases where score is unknown
data["score"][pd.isnull(data["score"])] = 0

if data["score"].dtype != np.float64:
    data["score"][data["score"] == "None"] = 0

data["score"] = data["score"].astype(float)
data.fillna(0, inplace=True)

# extract columns that refer to parameters
params_cols = [col for col in data.columns if
               col not in ['cls', 'label_col', 'method', 'metric', 'nr_events', 'part', 'score']]

# aggregate data over all CV folds
data_agg = data.groupby(["cls", "label_col", "method", "metric", "nr_events"] + params_cols, as_index=False)[
    "score"].mean()
data_agg_over_all_prefixes = data.groupby(["cls", "label_col", "method", "metric"] + params_cols, as_index=False)[
    "score"].mean()

# select best params according to MAE only (for regression) and accuracy only (for classification)
data_regr_agg = data_agg[data_agg.metric == "mae"]
data_class_agg = data_agg[data_agg.metric == "acc"]
data_regr_agg_over_all_prefixes = data_agg_over_all_prefixes[data_agg_over_all_prefixes.metric == "mae"]
data_class_agg_over_all_prefixes = data_agg_over_all_prefixes[data_agg_over_all_prefixes.metric == "acc"]

# select the best params - lowest MAE or highest accuracy
data_regr_best = data_regr_agg.sort_values("score", ascending=True).groupby(["cls", "label_col", "method", "metric", "nr_events"],
                                                                  as_index=False).first()
data_regr_best_over_all_prefixes = data_regr_agg_over_all_prefixes.sort_values("score", ascending=True).groupby(
    ["cls", "label_col", "method", "metric"], as_index=False).first()
data_class_best = data_class_agg.sort_values("score", ascending=False).groupby(["cls", "label_col", "method", "metric", "nr_events"],
                                                                  as_index=False).first()
data_class_best_over_all_prefixes = data_class_agg_over_all_prefixes.sort_values("score", ascending=False).groupby(
    ["cls", "label_col", "method", "metric"], as_index=False).first()

data_best = pd.concat([data_regr_best, data_class_best], axis=0)
data_best_over_all_prefixes = pd.concat([data_regr_best_over_all_prefixes, data_class_best_over_all_prefixes], axis=0)

best_params = {}

# all except prefix length based
for row in data_best_over_all_prefixes[~data_best_over_all_prefixes.method.str.contains("prefix")][
            ["label_col", "method", "cls"] + params_cols].values:

    if row[0] not in best_params:
        best_params[row[0]] = {}
    if row[1] not in best_params[row[0]]:
        best_params[row[0]][row[1]] = {}
    if row[2] not in best_params[row[0]][row[1]]:
        best_params[row[0]][row[1]][row[2]] = {}

    for i, param in enumerate(params_cols):
        value = row[3 + i]
        if param == "max_features":
            value = value if value == "sqrt" else float(value)
        elif param in ["n_clusters", "n_estimators", "max_depth"]:
            value = int(value)
        elif param == "learning_rate":
            value = float(value)

        best_params[row[0]][row[1]][row[2]][param] = value

# only prefix length based
for row in data_best[data_best.method.str.contains("prefix")][
            ["label_col", "method", "cls", "nr_events"] + params_cols].values:

    if row[0] not in best_params:
        best_params[row[0]] = {}
    if row[1] not in best_params[row[0]]:
        best_params[row[0]][row[1]] = {}
    if row[2] not in best_params[row[0]][row[1]]:
        best_params[row[0]][row[1]][row[2]] = {}
    if row[3] not in best_params[row[0]][row[1]][row[2]]:
        best_params[row[0]][row[1]][row[2]][row[3]] = {}

    for i, param in enumerate(params_cols):
        value = row[4 + i]
        if param == "max_features":
            value = value if value == "sqrt" else float(value)
        elif param in ["n_clusters", "n_estimators", "max_depth"]:
            value = int(value)
        elif param == "learning_rate":
            value = float(value)

        best_params[row[0]][row[1]][row[2]][row[3]][param] = value

# write to file
# with open(os.path.join(home_dir, training_params_dir, "%s.json" % dataset_ref), "w") as fout:
#     json.dump(best_params, fout, indent=3)
with open(Path.cwd().parent /  training_params_dir / ("%s.json" % dataset_ref), "w") as fout:
    json.dump(best_params, fout, indent=3)
