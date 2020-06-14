#!/bin/bash -i

source scripts/newton_krylov_env_cmds

err_cnt=0

echo running setup_solver.sh for iage
./scripts/setup_solver.sh --fp_cnt 1 --nlevs 20 --persist \
    --logging_reproducible \
    --logging_fname "%(workdir)s/newton_krylov_travis_long_iage.log" \
    --tracer_module_names iage \
    --workdir $HOME/travis_long_iage_workdir || err_cnt=$((err_cnt+1))

echo running nk_driver.sh for iage
$HOME/travis_long_iage_workdir/nk_driver.sh || err_cnt=$((err_cnt+1))

diff $HOME/travis_long_iage_workdir/newton_krylov_travis_long_iage.log \
    baselines || err_cnt=$((err_cnt+1))

echo err_cnt=$err_cnt

exit $err_cnt
