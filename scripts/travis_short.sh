#!/bin/bash -i

source scripts/newton_krylov_env_cmds

echo running black
black --check .

echo running flake8
flake8

# setup solver, with default tracer modules
# persist not needed since driver is not invoked
# this does perform forward model runs, since fp_cnt=1
# also exercises passing --model_name to setup_solver.sh
echo running setup_solver.sh
./scripts/setup_solver.sh --fp_cnt 1 --nlevs 20 --persist \
    --model_name test_problem \
    --workdir $HOME/travis_short_workdir

# now that the travis_short_workdir is populated, pytest tests have what they need
echo running pytest
pytest
