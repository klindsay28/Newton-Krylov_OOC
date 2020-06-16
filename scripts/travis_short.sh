#!/bin/bash -i

source scripts/newton_krylov_env_cmds

err_cnt=0

echo running black
black --check . || err_cnt=$((err_cnt+1))

echo running flake8
flake8 || err_cnt=$((err_cnt+1))

# setup solver, with default tracer modules
# persist not needed since driver is not invoked
# this does perform forward model runs, since fp_cnt=1
# also exercises passing --model_name to setup_solver.sh
echo running setup_solver.sh
./scripts/setup_solver.sh --fp_cnt 1 --nlevs 20 --persist \
    --logging_reproducible \
    --logging_fname "%(workdir)s/newton_krylov_travis_short.log" \
    --model_name test_problem \
    --workdir $HOME/travis_short_workdir || err_cnt=$((err_cnt+1))

# now that the travis_short_workdir is populated, pytest tests have what they need
echo running pytest
pytest || err_cnt=$((err_cnt+1))

echo err_cnt=$err_cnt

exit $err_cnt
