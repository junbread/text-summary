#!/usr/bin/env bash

if [ -z "$CLASSPATH" ]
then
    export CLASSPATH=$(pwd)/lib/stanford-corenlp-3.9.2.jar
else
    export CLASSPATH=$CLASSPATH:$(pwd)/lib/stanford-corenlp-3.9.2.jar
fi

if [ -z "$PYTHONPATH" ]
then
    export PYTHONPATH=$(pwd)/src/sum:$(pwd)/src/demo
else
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src/sum:$(pwd)/src/demo
fi

python src/demo/manage.py runserver