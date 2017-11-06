## Training
```bash
export PYTHONPATH=....../PredictiveMethods/
cd PredictiveMethods/core/
python train.py log_name_csv bucketing_type encoding_type learner_type target 
```

* log_name_csv - name of the file as in `logdata` directory 
* bucketing_type - `zero`, `cluster`, `state` or `prefix`
* encoding_type - `agg`, `laststate`, `index` or `combined`
* learner_type - `rf` for random forest, `gbm` for gradient boosting, `dt` for decision tree or `xgb` for extreme gradient boosting
* target - name of the column that you need to predict, e.g. `remtime` or `label` (check if it exists in the log)

Example:

```bash
python train.py BPIC15_4.csv zero agg rf remtime

```

This script assumes that you have a training configuration file in `core/training_params/{log-name-without-extension}.json
` with the following structure:

```json
{
  "target": {
      "bucketing_encoding": {
         "leaner": {
            "n_clusters": ,
            "n_estimators": ,
            "max_features": ,
            "learning_rate": 
         }
      }
  }
}
```

Example:
```json
{
   "remtime": {
      "zero_agg": {
         "rf": {
            "n_clusters": 1,
            "n_estimators": 100,
            "max_features": 0.25,
            "learning_rate": 0.03
         }
      }
  }
}
```

Output of the training script:

* Fitted model - PredictiveMethods/pkl/{log-name-without-extension}_bucketing_encoding_learner_target.**pkl**
* Validation results - PredictiveMethods/results/validation/validation_{log-name-without-extension}_bucketing_encoding_learner_target.**csv**
* Data on feature importance - PredictiveMethods/results/feature_importance/feat_importance_{log-name-without-extension}_bucketing_encoding_learner_target.**csv**



## Test
```bash
export PYTHONPATH=....../PredictiveMethods/
cd PredictiveMethods/core/
python test.py path_to_prefix.csv path_to_pickle_model_filename 
```

Example:
```bash
python test.py test_BPIC15_4.json ../pkl/BPIC15_4_zero_agg_rf_remtime.pkl
```

The output should be printed to stdout
