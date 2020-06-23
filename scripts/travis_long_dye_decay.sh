#!/bin/bash -i

source scripts/newton_krylov_env_cmds

err_cnt=0

echo running setup_solver.sh for dye_decay
./scripts/setup_solver.sh --fp_cnt 1 --nlevs 20 --persist \
    --tracer_module_names dye_decay_{suff}:001:010 \
    --newton_rel_tol "1.0e-6" \
    --workdir $HOME/travis_long_dye_decay_workdir || err_cnt=$((err_cnt+1))

echo running nk_driver.sh for dye_decay
$HOME/travis_long_dye_decay_workdir/nk_driver.sh || err_cnt=$((err_cnt+1))

echo comparing Newton_state.json to baseline
expr_dir=$HOME/travis_long_dye_decay_workdir
baseline_dir=baselines/travis_long_dye_decay
diff -b <(sed "s%$HOME%HOME%g" $expr_dir/Newton_state.json) \
    $baseline_dir/Newton_state.json || err_cnt=$((err_cnt+1))

echo err_cnt=$err_cnt

exit $err_cnt
