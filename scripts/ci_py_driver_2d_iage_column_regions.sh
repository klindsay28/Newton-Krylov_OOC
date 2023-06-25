#!/bin/bash -i

source scripts/newton_krylov_env_cmds

err_cnt=0

workdir=$HOME/ci_py_driver_2d_iage_column_regions_workdir
mkdir -p $workdir

# create override.cfg file with reduced grid size
# and eliminate lateral processes
cat > $workdir/override.cfg <<EOF
[modelinfo]
depth_nlevs = 20
ypos_nlevs = 3
max_abs_vvel = 0.0
horiz_mix_coeff = 0.0
EOF

input_dir=`pwd`/input/py_driver_2d
cfg_fnames=$input_dir/newton_krylov.cfg,$input_dir/model_params.cfg,$workdir/override.cfg

# setup solver, with default tracer modules
echo running setup_solver.sh
./scripts/setup_solver.sh --fp_cnt 1 \
    --model_name py_driver_2d --tracer_module_names iage --persist \
    --cfg_fnames $cfg_fnames --workdir $workdir --deprecation_warning_to_error \
    $@ || err_cnt=$((err_cnt+1))

for fname in grid_vars.nc; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $workdir \
        --baseline_dir baselines/ci_py_driver_2d_iage_column_regions || err_cnt=$((err_cnt+1))
done

for fname in fcn_0000.nc hist_0000.nc init_iterate.nc init_iterate_0000.nc; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname --atol 1.0e-6 --rtol 1.0e-3 \
        --expr_dir $workdir/gen_init_iterate \
        --baseline_dir baselines/ci_py_driver_2d_iage_column_regions || err_cnt=$((err_cnt+1))
done

echo running nk_driver.sh for py_driver_2d
$workdir/nk_driver.sh || err_cnt=$((err_cnt+1))

for fname in precond_00.nc ; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $workdir/krylov_00 \
        --baseline_dir baselines/ci_py_driver_2d_iage_column_regions || err_cnt=$((err_cnt+1))
done

for fname in precond_fcn_00.nc ; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $workdir/krylov_00 \
        --baseline_dir baselines/ci_py_driver_2d_iage_column_regions \
        --rtol 2.0e-3 || err_cnt=$((err_cnt+1))
done

for fname in basis_00.nc ; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $workdir/krylov_00 \
        --baseline_dir baselines/ci_py_driver_2d_iage_column_regions \
        --atol 5.0e-5 || err_cnt=$((err_cnt+1))
done

for fname in perturb_fcn_w_raw_00.nc ; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $workdir/krylov_00 \
        --baseline_dir baselines/ci_py_driver_2d_iage_column_regions \
        --atol 5.0e-6 || err_cnt=$((err_cnt+1))
done

for fname in krylov_res_00.nc ; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $workdir/krylov_00 \
        --baseline_dir baselines/ci_py_driver_2d_iage_column_regions \
        --rtol 1.9e-2 || err_cnt=$((err_cnt+1))
done

for fname in increment_00.nc iterate_01.nc ; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $workdir \
        --baseline_dir baselines/ci_py_driver_2d_iage_column_regions \
        --rtol 1.9e-2 || err_cnt=$((err_cnt+1))
done

echo comparing Newton_state.json to baseline
expr_dir=$workdir
baseline_dir=baselines/ci_py_driver_2d_iage_column_regions
diff -u -b <(sed "s%$HOME%HOME%g" $expr_dir/Newton_state.json) \
    $baseline_dir/Newton_state.json || err_cnt=$((err_cnt+1))

echo err_cnt=$err_cnt

exit $err_cnt
