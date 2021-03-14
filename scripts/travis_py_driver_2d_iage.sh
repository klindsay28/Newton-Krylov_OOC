#!/bin/bash -i

source scripts/newton_krylov_env_cmds

err_cnt=0

workdir=$HOME/travis_py_driver_2d_iage_workdir
mkdir -p $workdir

# create override.cfg file with reduced grid size
cat > $workdir/override.cfg <<EOF
[modelinfo]
depth_nlevs = 30
ypos_nlevs = 30
EOF

input_dir=`pwd`/input/py_driver_2d
cfg_fnames=$input_dir/newton_krylov.cfg,$input_dir/model_params.cfg,$workdir/override.cfg

# setup solver, with default tracer modules
# persist not needed since driver is not invoked
# this does perform forward model runs, since fp_cnt=1
echo running setup_solver.sh
./scripts/setup_solver.sh --fp_cnt 1 \
    --model_name py_driver_2d --tracer_module_names iage \
    --cfg_fnames $cfg_fnames --workdir $workdir $@ || err_cnt=$((err_cnt+1))

for fname in grid_vars.nc; do
    echo comparing $fname
    python -m src.baseline_cmp --fname $fname \
        --expr_dir $workdir \
        --baseline_dir baselines/travis_py_driver_2d_iage || err_cnt=$((err_cnt+1))
done

for fname in fcn_0000.nc hist_0000.nc init_iterate.nc init_iterate_0000.nc; do
    echo comparing $fname
    python -m src.baseline_cmp --fname $fname --atol 4.0e-7 --rtol 1.0e-3 \
        --expr_dir $workdir/gen_init_iterate \
        --baseline_dir baselines/travis_py_driver_2d_iage || err_cnt=$((err_cnt+1))
done

echo err_cnt=$err_cnt

exit $err_cnt
