#!/bin/bash -i

source scripts/newton_krylov_env_cmds

err_cnt=0

echo running setup_solver.sh for iage
./scripts/setup_solver.sh --fp_cnt 1 --nlevs 20 --persist \
    --logging_reproducible \
    --logging_fname "%(workdir)s/newton_krylov_travis_long_iage.log" \
    --tracer_module_names iage \
    --workdir $HOME/travis_long_iage_workdir || err_cnt=$((err_cnt+1))

echo comparing iage from gen_init_iterate fixed point iteration \
    to same from from travis_short
dir_iage_phosphorus=$HOME/travis_short_workdir/gen_init_iterate
dir_iage=$HOME/travis_long_iage_workdir/gen_init_iterate
diff -q -I 'po4' -I 'dop' -I 'pop' -I 'history =' \
    <(ncdump -v iage $dir_iage_phosphorus/hist_00.nc) \
    <(ncdump -v iage $dir_iage/hist_00.nc) || err_cnt=$((err_cnt+1))

echo running nk_driver.sh for iage
$HOME/travis_long_iage_workdir/nk_driver.sh || err_cnt=$((err_cnt+1))

echo err_cnt=$err_cnt

exit $err_cnt
