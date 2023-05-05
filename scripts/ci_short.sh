#!/bin/bash -i

source scripts/newton_krylov_env_cmds
conda env export --no-builds

err_cnt=0

echo running isort
isort --diff nk_ooc tests || err_cnt=$((err_cnt+1))

echo running black
black --diff . || err_cnt=$((err_cnt+1))

echo running flake8
flake8 || err_cnt=$((err_cnt+1))

echo running pytest
pytest || err_cnt=$((err_cnt+1))

for model_dir in input/*; do
    model_name=$(basename $model_dir)
    echo checking variable usage in newton_krylov.cfg for $model_name
    ./scripts/check_cfg_var_usage.sh $model_name || err_cnt=$((err_cnt+1))
done

# setup solver, with default tracer modules
# persist not needed since driver is not invoked
# this does perform forward model runs, since fp_cnt=1
# also exercises passing --model_name to setup_solver.sh
echo running setup_solver.sh
./scripts/setup_solver.sh --fp_cnt 1 --depth_nlevs 20 --persist \
    --model_name test_problem \
    --workdir $HOME/ci_short_workdir --deprecation_warning_to_error \
    $@ || err_cnt=$((err_cnt+1))

for fname in depth_axis.nc; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $HOME/ci_short_workdir \
        --baseline_dir baselines/ci_short || err_cnt=$((err_cnt+1))
done

for fname in fcn_00.nc hist_00.nc init_iterate.nc init_iterate_00.nc; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $HOME/ci_short_workdir/gen_init_iterate \
        --baseline_dir baselines/ci_short || err_cnt=$((err_cnt+1))
done

echo err_cnt=$err_cnt

exit $err_cnt
