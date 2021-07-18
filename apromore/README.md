## Predictive monitoring setup for Apromore

[Apromore](http://apromore.org) is a business process analytics platform.
It has optional support for training [Nirdizati](http://nirdizati.org) models and using them for predictive monitoring.
Use the following procedure to set up a Nirdizati backend to work with Apromore:


* Obtain a Kafka distribution at version 2.2.0, for instance from the following [URL](https://archive.apache.org/dist/kafka/2.2.0/kafka_2.12-2.2.0.tgz)

* A minimal Kafka cluster requires at least one Zookeeper server and one Kafka broker.
From the top directory of the Kafka distribution, execute the following commands in separate shells to start Zookeeper and Kafka:

```bash
$ bin/zookeeper-server-start.sh config/zookeeper.properties
$ bin/kafka-server-start.sh config/server.properties
```

or under Windows:
```bat
> bin\windows\zookeeper-server-start.bat config\zookeeper.properties
> bin\windows\kafka-server-start.bat config\server.properties
```

* To create the default topics Apromore will use to communicate with Nirdizati, execute the following commands:

```bash
$ bin/kafka-topics.sh --zookeeper localhost:2181 --create --topic events --replication-factor 1 --partitions 1
$ bin/kafka-topics.sh --zookeeper localhost:2181 --create --topic prefixes --replication-factor 1 --partitions 1
$ bin/kafka-topics.sh --zookeeper localhost:2181 --create --topic predictions --replication-factor 1 --partitions 1
$ bin/kafka-topics.sh --zookeeper localhost:2181 --create --topic control --replication-factor 1 --partitions 1
```

or under Windows:
```bat
> bin\windows\kafka-topics.bat --zookeeper localhost:2181 --create --topic events --replication-factor 1 --partitions 1
> bin\windows\kafka-topics.bat --zookeeper localhost:2181 --create --topic prefixes --replication-factor 1 --partitions 1
> bin\windows\kafka-topics.bat --zookeeper localhost:2181 --create --topic predictions --replication-factor 1 --partitions 1
> bin\windows\kafka-topics.bat --zookeeper localhost:2181 --create --topic control --replication-factor 1 --partitions 1
```

* Ensure that Python 3 is available.  This procedure is known to work with Python versions 3.5 and 3.6.
From the `nirdizati-training-backend` directory, install additional required libraries using the following command:

```bash
$ pip install -r requirements.txt
```

* Modify `PYTHONPATH` in your shell environment to include the `nirdizati-training-backend` and `nirdizati-training-backend/core` directories.
For instance, in bash this can be achieved by running something like:

```bash
$ export PYTHONPATH="$HOME/Work/nirdizati-training-backend:$HOME/Work/nirdizati-training-backend/core"
```

or under Windows:
```bat
> set PYTHONPATH=%HOME%\Work\nirdizati-training-backend;%HOME%\Work\nirdizati-training-backend\core
```

* Let's presume that Apromore will be running at `localhost:9000` and the Kafka broker at `localhost:9092`.
Enter the `nirdizati-training-backend/apromore` subdirectory and start the collator and predictor Kafka processors by executing the following commands:

```bash
$ python collate-events.py localhost:9092 control events prefixes 2
```

* Expect a message like: Collating events from "events" into "prefixes" with control channel "control" every 2.0 seconds

```bash
$ python predict.py localhost:9092 localhost:9000 control prefixes predictions
```

* Expect a message like: Making predictions using case prefixes from "prefixes" into "predictions" with control channel "control"

* At this point the Apromore plugins for training predictors and performing predictive monitoring can be deployed.
Consult the top-level `ApromoreCode/README.md` file for configuration details on the Apromore side.
The Zookeeper, Kafka broker, collator, and predictor must all remain running to provide predictive monitoring.


To reset the system, delete the following files:

```bash
$ rm -rf apromore/collator_state core/dataset_params/* core/training_params/* logdata/* pkl/* results/*/*
```

and (from the Kafka distribution directory) delete the Kafka topics:

```bash
$ bin/kafka-topics.sh --zookeeper localhost:2181 --delete --topic events
$ bin/kafka-topics.sh --zookeeper localhost:2181 --delete --topic prefixes
$ bin/kafka-topics.sh --zookeeper localhost:2181 --delete --topic predictions
$ bin/kafka-topics.sh --zookeeper localhost:2181 --delete --topic control
```

or under Windows:
```bat
> bin\window\kafka-topics.bat --zookeeper localhost:2181 --delete --topic events
> bin\window\kafka-topics.bat --zookeeper localhost:2181 --delete --topic prefixes
> bin\window\kafka-topics.bat --zookeeper localhost:2181 --delete --topic predictions
> bin\window\kafka-topics.bat --zookeeper localhost:2181 --delete --topic control
```
