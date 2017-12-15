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
* target - *name of the column* that you want to predict. The prediction problem type (classification or regression) is determine automatically based on the number of unique levels of a target variable and whether or not it can be parsed as a numeric series. Two special targets are `remtime` or `label` (for the remaining time and case outcome based on the median case duration). Note that keywords `remtime` and `label` are reserved, i.e if there is already a column with such names, it will be ignored, but remaining time and outcome will be predicted instead. 

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
            "learning_rate":  // used for gbm and xgb
            "max_depth": , // used for dt and xgb
            "n_estimators": , // used for rf, gbm and xgb
            "n_clusters": , // used for all (=1 if bucketing method != cluster, otherwise to be entered by user)
            "colsample_bytree": , // used for xgb
            "max_features": , // used for rf, gbm and dt
            "subsample": // used for xgb
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
            "n_estimators": 300,
            "max_features": 0.5,
            "n_clusters": 1
         }
      }
  }
}
```

Output of the training script:

* Fitted model - PredictiveMethods/pkl/{log-name-without-extension}_bucketing_encoding_learner_target.**pkl**
* Validation results - PredictiveMethods/results/validation/validation_{log-name-without-extension}_bucketing_encoding_learner_target.**csv**
* Data on feature importance - PredictiveMethods/results/feature_importance/feat_importance_{log-name-without-extension}_bucketing_encoding_learner_target.**csv**


## How to choose default training parameters
Bucketing - No bucketing (zero)
Encoding - Frequency (agg)
Predictor - XGBoost

Default hyperparameters:
* Random forest: Number of estimators 300, max_features 0.5
* Gradient boosting: Number of estimators 300, max_features 0.5, learning rate 0.1
* Decision tree: max_features 0.5, max_depth 5
* XGBoost: Number of estimators 300, learning rate 0.03, subsample row ration 0.7, subsample column ratio 0.7, max_depth 5

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
