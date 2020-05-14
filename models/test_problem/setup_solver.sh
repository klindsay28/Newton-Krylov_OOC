#!/bin/bash

set -e

toplevel_dir=`git rev-parse --show-toplevel`

cd $toplevel_dir

if [ -z ${PYTHONPATH+x} ]; then
    export PYTHONPATH=models
else
    export PYTHONPATH=models:$PYTHONPATH
fi

python -m src.test_problem.setup_solver $@
# ./models/test_problem/src/setup_solver.py $@
