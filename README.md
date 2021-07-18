This repo contains scripts to train models for [Nirdizati](http://nirdizati.com) predictive process monitoring engine. The following instructions are for the standalone use. For use with [Apromore](http://apromore.org/), please refer to [these instructions](https://github.com/nirdizati/nirdizati-training-backend/blob/master/apromore/README.md).

## Requirements
Tested with Python 3.5 and Python 3.6. Install the necessary packages with
```bash
pip install -r requirements.txt
```

## Training predictive models
```bash
export PYTHONPATH=$PYTHONPATH:/wherever/you/keep/nirdizati-training-backend
cd core/
python train.py training-config-ID 
```

* `training-config-ID` - JSON file (without extension) that contains training configuration (see below), must be placed under `core/training_params/` directory

Example:

```bash
python train.py myconfig_remtime
```

This script assumes that you have a training configuration file `core/training_params/myconfig_remtime.json
` with the following structure: 

### Training configuration structure

```json
{
  "target": {
    "bucketing_type": {
      "encoding_type": {
        "learner_type": {
          "learner_param1": "value1",
          "learner_param2": "value2",
          ...
        }
      }
    }
  },
  "ui_data": {
    "log_file": "/wherever/you/keep/your/log.csv"
  }
}
```

* `bucketing_type` - `zero`, `cluster`, `state` or `prefix`. Use tooltips in the Nirdizati tool (advanced training mode) for short explanation
* `encoding_type` - `agg`, `laststate`, `index` or `combined`
* `learner_type` - `rf` for random forest, `gbm` for gradient boosting, `dt` for decision tree or `xgb` for extreme gradient boosting
* `learner_param`'s - most important hyperparameters for each learner, taken from [sklearn](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble) lib: `n_estimators` and `max_features` for `rf`; `n_estimators`, `max_features` and `learning_rate` for `gbm`; `max_features` and `max_depth` for `dt`; `n_estimators`, `max_depth`, `learning_rate`, `colsample_bytree` and `subsample` for `xgb` 
* `target` - variable that you want to predict (see below). The prediction problem type (classification or regression) is determined automatically based on the number of unique levels of a target variable and whether or not it can be parsed as a numeric series.


[Example of a training configuration file](https://github.com/nirdizati/nirdizati-training-backend/blob/master/core/training_params/myconfig.json)


### What can be predicted?
* Remaining cycle time. Use `remtime` keyword as a `target`  argument for train.py
* Binary case outcome based on the expected case duration (whether case duration will exceed a specified threshold). Use a positive threshold value for `target` or "-1" if you want the labeling to be based on the *median* case duration.   
* Next activity to be executed. Use `next` for `target`
* Any static, i.e. case, attribute that is already available in the log as a column. In this case, `target` is the name of the corresponding column.

 

### Output of the training script:

* Fitted model - `pkl/`
* Validation results by prefix length - `results/validation/`
* Detailed validation results - `results/detailed/`
* Data on feature importance - `results/feature_importance/`


### How to choose default training parameters?
Bucketing - No bucketing (zero)

Encoding - Frequency (agg)

Predictor - XGBoost

Default hyperparameters for XGBoost predictor:
* Random forest: Number of estimators 300, max_features 0.5
* Gradient boosting: Number of estimators 300, max_features 0.5, learning rate 0.1
* Decision tree: max_features 0.8, max_depth 5
* XGBoost: Number of estimators 300, learning rate 0.04, subsample row ration 0.7, subsample column ratio 0.7, max_depth 5

## Test for an ongoing case
```bash
export PYTHONPATH=$PYTHONPATH:/wherever/you/keep/nirdizati-training-backend
cd core/
python predict_trace.py path_to_single_test_prefix.json path_to_pickle_model_filename 
```

Example:
```bash
python predict_trace.py ../logdata/1170931347.json ../pkl/bpi17_sample_myconfig_remtime.pkl
```

The output should be printed to stdout, for example:
```bash
[{"remtime":15.9565439224}]
```
