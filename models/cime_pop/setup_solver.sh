#!/bin/bash -i

set -e

toplevel_dir=`git rev-parse --show-toplevel`

cd $toplevel_dir

source src/newton_krylov_env_cmds

if [ -z ${PYTHONPATH+x} ]; then
    export PYTHONPATH=models
else
    export PYTHONPATH=models:$PYTHONPATH
fi

python -m src.cime_pop.setup_solver $@
# ./models/cime_pop/src/setup_solver.py $@
