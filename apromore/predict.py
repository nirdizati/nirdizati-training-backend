"""
Copyright (c) 2016-2017 The Nirdizati Project.
This file is part of "Nirdizati".

"Nirdizati" is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 3 of the
License, or (at your option) any later version.

"Nirdizati" is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this program.
If not, see <http://www.gnu.org/licenses/lgpl.html>.
"""

import warnings
warnings.filterwarnings('ignore')

import pickle
import sys
from sys import argv
import time

import numpy as np
import pandas as pd

import json
import kafka
import tempfile
import urllib.request

if len(argv) != 6:
    sys.exit("Usage: python {} bootstrap-server:port apromore-server:port control-topic prefixes-topic predictions-topic".format(argv[0]))

bootstrap_server, apromore_server, control_topic, prefixes_topic, predictions_topic = argv[1], argv[2], argv[3], argv[4], argv[5]

print("Making predictions using case prefixes from \"{}\" into \"{}\" with control channel \"{}\"".format(prefixes_topic, predictions_topic, control_topic))
consumer = kafka.KafkaConsumer(prefixes_topic, group_id="predict", bootstrap_servers=bootstrap_server, auto_offset_reset='earliest')
controlConsumer = kafka.KafkaConsumer(control_topic, bootstrap_servers=bootstrap_server, auto_offset_reset='earliest')
producer = kafka.KafkaProducer(bootstrap_servers=bootstrap_server)

active_logs = None
predictor_cache = {}

def getPredictor(predictor_id):
    """ Fetch a predictor by id, either by downloading it from Apromore or from the local cache """
    # if the pickle is cached, we're done
    if predictor_id in predictor_cache:
        return predictor_cache[predictor_id]

    # if the pickle isn't cached, retrieve it from the Apromore server
    print("Downloading predictor {} from Apromore".format(predictor_id))
    f = urllib.request.urlopen("http://{}/predictiveMonitor/id?{}".format(apromore_server, predictor_id))
    pipelines = pickle.load(f)
    bucketer = pickle.load(f)
    dataset_manager = pickle.load(f)

    dtypes = {col: "str" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
          [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
    for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
        dtypes[col] = "float"

    predictor_cache[predictor_id] = pipelines, bucketer, dataset_manager, dtypes
    return predictor_cache[predictor_id]


""" As case prefixes arrive on prefixes_topic, execute each required predictor and forward the prediction to predictions_topic """
while True:
    rawControlMessage = controlConsumer.poll()
    for topicPartition, controlMessages in rawControlMessage.items():
        for controlMessage in controlMessages:
            event = json.loads(controlMessage.value.decode())
            " TODO: clear log_cases of any deleted logs "
            active_logs = set(event)
            print("Updated active logs to {}".format(active_logs))

    rawMessage = consumer.poll(timeout_ms=1.0, max_records=1)
    if not rawMessage.items():
        time.sleep(1.0)
    for topicPartition, messages in rawMessage.items():
        for message in messages:
            jsonValue = json.loads(message.value)
            log_id = jsonValue["log_id"]
            if active_logs is not None and not log_id in active_logs:
                print("Discarding case prefix from deleted monitor {}".format(log_id))
            else:
                predictor = jsonValue["predictor"]
                try:
                    pipelines, bucketer, dataset_manager, dtypes = getPredictor(predictor)
                    tag = dataset_manager.label_col

                    jsonPrefix = jsonValue["prefix"]
                    for event in jsonPrefix:
                        event[dataset_manager.case_id_col] = jsonValue["case_id"]
                        for case_attribute in jsonValue["case_attributes"]:
                            if not case_attribute in event:
                                event[case_attribute] = jsonValue["case_attributes"][case_attribute]

                    prefix = json.dumps(jsonPrefix)
                    """
                    print("-- INPUT --")
                    print(prefix)
                    """
                    test = pd.read_json(prefix, orient='records', dtype=dtypes)
                    test[dataset_manager.timestamp_col] = pd.to_datetime(test[dataset_manager.timestamp_col])

                    # get bucket for the test case
                    bucket = bucketer.predict(test).item()

                    # select relevant classifier
                    if bucket not in pipelines:  # state-based bucketing may fail, in which case no prediction is issued
                        print("-- No prediction issued, likely because a state-based bucketer never encountered this prefix in training --")
                    else:
                        # make actual predictions
                        preds = pipelines[bucket].predict_proba(test)
                        if preds.ndim == 1:  # regression
                            preds = pd.DataFrame(preds.clip(min=0), columns=[dataset_manager.label_col])

                        preds = preds.to_json(orient='records')

                        # emit the predictions to the predictions_topic
                        output = {
                            "log_id":      log_id,
                            "case_id":     jsonValue["case_id"],
                            "event_nr":    len(jsonValue["prefix"]),
                            "predictions": {tag: json.loads(preds)[0]}
                        }
                        print("-- OUTPUT --")
                        print(json.dumps(output))
                        """
                        print(json.dumps({ "input": jsonPrefix, "output": output }))
                        """
                        producer.send(predictions_topic, json.dumps(output).encode('utf-8'))

                except urllib.error.HTTPError as e:
                    print("Unable to download predictor: {}".format(e))

                except KeyError as e:
                    print("Log does not match predictor: {}".format(e))
