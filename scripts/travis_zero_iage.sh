#!/bin/bash -i

source scripts/newton_krylov_env_cmds

err_cnt=0

echo running setup_solver.sh for zero iage
./scripts/setup_solver.sh --fp_cnt 0 --nlevs 20 --persist \
    --tracer_module_names iage --init_iterate_opt zeros \
    --workdir $HOME/travis_zero_iage_workdir || err_cnt=$((err_cnt+1))

echo running nk_driver.sh for zero iage
$HOME/travis_zero_iage_workdir/nk_driver.sh || err_cnt=$((err_cnt+1))

echo err_cnt=$err_cnt

exit $err_cnt
