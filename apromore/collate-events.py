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

import sys
import json
import time
from kafka import KafkaProducer, KafkaConsumer
import os.path

if len(sys.argv) != 6:
    sys.exit("Usage: python {} bootstrap-server:port control-topic events-topic prefixes-topic delay".format(sys.argv[0]))

bootstrap_server, control_topic, source_topic, destination_topic, delay = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
state_dir = "collator_state"

print("Collating events from \"{}\" into \"{}\" with control channel \"{}\" every {} seconds".format(source_topic,
                                                                                                     destination_topic,
                                                                                                     control_topic,
                                                                                                     delay))
consumer = KafkaConsumer(source_topic, bootstrap_servers=bootstrap_server, auto_offset_reset='earliest')
controlConsumer = KafkaConsumer(control_topic, bootstrap_servers=bootstrap_server, auto_offset_reset='earliest')
producer = KafkaProducer(bootstrap_servers=bootstrap_server)

active_logs = None

""" This is a map keyed on string-valued case_id and containing sequences of event objects """
log_cases = {}

""" As events arrive on source_topic, collate case prefixes and forward them on destination_topic """
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
        time.sleep(delay)
    for topicPartition, messages in rawMessage.items():
        for message in messages:
            event = json.loads(message.value)
            log_id = event.get('log_id')
            if active_logs is not None and not log_id in active_logs:
                print("Discarding event from deleted monitor {}".format(log_id))
            else:
                case_id = event.get('case_id')
                event_nr = int(event.get('event_nr'))

                if not os.path.isdir(state_dir):
                    os.mkdir(state_dir)
                dir_path = os.path.join(state_dir, str(log_id))
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                file_path = os.path.join(dir_path, case_id)
                if not os.path.isfile(file_path):
                    case_prefix = []
                else:
                    case_prefix = json.loads(open(file_path).read())
                if event_nr is None or event_nr > len(case_prefix):
                    case_prefix.append(event.get('event_attributes'))
                    if event_nr is not None and event_nr != len(case_prefix):
                        print("Event is labeled as {} but case prefix only has {} events".format(event_nr,
                                                                                                 len(case_prefix)))
                    open(file_path, mode='w').write(json.dumps(case_prefix, indent="\t"))
                    print("Collated event {} of case {}".format(len(case_prefix), case_id))
                    for predictorId in event.get('predictors'):
                        prediction_job = {"log_id": log_id,
                                          "case_id": case_id,
                                          "event_nr": event_nr,
                                          "predictor": predictorId,
                                          "case_attributes": event.get('case_attributes'),
                                          "prefix": case_prefix}
                        producer.send(destination_topic, json.dumps(prediction_job).encode('utf-8'))
                    time.sleep(delay)
                else:
                    print("Skipped collating event {} of case {}".format(len(case_prefix), case_id))
