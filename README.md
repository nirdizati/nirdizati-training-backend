## Training
```bash
cd PredictiveMethods/core/
python train.py log_name_csv bucketing_type encoding_type learner_type target 
```

* log_name_csv - name of the file as in `logdata` directory 
* bucketing_type - `single`, `cluster`, `state` or `prefix`
* encoding_type - `agg`, `laststate`, `index` or `combined`
* learner_type - `rf`, `gbm` or `dt`
* target - name of the column that you need to predict, e.g. `remtime` or `label` (check if it exists in the log)

Example:

```bash
python train.py BPIC15_4.csv single agg rf remtime

```

This script assumes that you have a training configuration file in `core/training_params/{log-name-without-extension}.json
` with the following structure:

```json
   target: {
      bucketing_encoding: {
         leaner: {
            "n_clusters": ,
            "n_estimators": ,
            "max_features": ,
            "gbm_learning_rate": 
         }
      }
  }
```

Example:
```json
   "remtime": {
      "single_agg": {
         "rf": {
            "n_clusters": 1,
            "n_estimators": 100,
            "max_features": 0.25,
            "gbm_learning_rate": 0.03
         }
      }
  }
```

Output of the training script:

* Fitted model - PredictiveMethods/pkl/{log-name-without-extension}_bucketing_encoding_learner_target.**pkl**
* Validation results - PredictiveMethods/results/final_results_{log-name-without-extension}_bucketing_encoding_learner_target.**csv**


## Test
```bash
cd PredictiveMethods/core/
python test.py pickle_model_filename 
```

`pickle_model_filename` should match an existing file under `pkl` folder

Example:
```bash
python test.py test_BPIC15_4.json BPIC15_4_single_agg_rf_remtime.pkl
```

The output should be printed to stdout
