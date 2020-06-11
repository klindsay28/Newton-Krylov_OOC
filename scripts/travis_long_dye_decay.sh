#!/bin/bash -i

source scripts/newton_krylov_env_cmds

# setup and run solver
echo running setup_solver.sh for dye_decay
./scripts/setup_solver.sh --fp_cnt 1 --nlevs 20 --persist \
    --tracer_module_names iage \
    --workdir $HOME/travis_long_iage_workdir
echo running nk_driver.sh for dye_decay
$HOME/travis_long_iage_workdir/nk_driver.sh
