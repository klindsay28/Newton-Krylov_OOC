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
    --model_name test_problem \
    --workdir $HOME/travis_short_workdir || err_cnt=$((err_cnt+1))

for fname in depth_axis_test.nc; do
    echo comparing $fname
    python -m src.baseline_cmp --fname $fname \
        --expr_dir $HOME/travis_short_workdir \
        --baseline_dir baselines/travis_short || err_cnt=$((err_cnt+1))
done

for fname in fcn_00.nc hist_00.nc init_iterate.nc init_iterate_00.nc; do
    echo comparing $fname
    python -m src.baseline_cmp --fname $fname \
        --expr_dir $HOME/travis_short_workdir/gen_init_iterate \
        --baseline_dir baselines/travis_short || err_cnt=$((err_cnt+1))
done

# now that the travis_short_workdir is populated, pytest tests have what they need
echo running pytest
pytest || err_cnt=$((err_cnt+1))

echo err_cnt=$err_cnt

exit $err_cnt
