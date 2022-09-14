#!/bin/bash -i

source scripts/newton_krylov_env_cmds

err_cnt=0

echo running setup_solver.sh for iage
./scripts/setup_solver.sh --fp_cnt 1 --depth_nlevs 20 --persist \
    --tracer_module_names iage \
    --workdir $HOME/ci_long_iage_workdir $@ || err_cnt=$((err_cnt+1))

echo comparing iage from gen_init_iterate fixed point iteration \
    to same from from ci_short
dir_iage_phosphorus=$HOME/ci_short_workdir/gen_init_iterate
dir_iage=$HOME/ci_long_iage_workdir/gen_init_iterate
diff -q -I 'po4' -I 'dop' -I 'pop' -I 'history =' \
    <(ncdump -v iage $dir_iage_phosphorus/hist_00.nc) \
    <(ncdump -v iage $dir_iage/hist_00.nc) || err_cnt=$((err_cnt+1))

echo running nk_driver.sh for iage
$HOME/ci_long_iage_workdir/nk_driver.sh || err_cnt=$((err_cnt+1))

for fname in precond_00.nc precond_fcn_00.nc basis_00.nc perturb_fcn_w_raw_00.nc ; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $HOME/ci_long_iage_workdir/krylov_00 \
        --baseline_dir baselines/ci_long_iage || err_cnt=$((err_cnt+1))
done

for fname in w_raw_00.nc w_00.nc krylov_res_00.nc ; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $HOME/ci_long_iage_workdir/krylov_00 \
        --baseline_dir baselines/ci_long_iage \
        --rtol 2.0e-4 || err_cnt=$((err_cnt+1))
done

for fname in increment_00.nc iterate_01.nc ; do
    echo comparing $fname
    python -m nk_ooc.baseline_cmp --fname $fname \
        --expr_dir $HOME/ci_long_iage_workdir \
        --baseline_dir baselines/ci_long_iage \
        --rtol 2.0e-4 || err_cnt=$((err_cnt+1))
done

echo comparing Newton_state.json to baseline
expr_dir=$HOME/ci_long_iage_workdir
baseline_dir=baselines/ci_long_iage
diff -u -b <(sed "s%$HOME%HOME%g" $expr_dir/Newton_state.json) \
    $baseline_dir/Newton_state.json || err_cnt=$((err_cnt+1))

echo err_cnt=$err_cnt

exit $err_cnt
