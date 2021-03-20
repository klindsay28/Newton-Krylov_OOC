#!/bin/bash -i

set -e

source scripts/newton_krylov_env_cmds

workdir=$HOME/py_driver_2d_forced_iage
rm -Rf $workdir
mkdir -p $workdir

input_dir=`pwd`/input/py_driver_2d

# create override.cfg file
cat > $workdir/override.cfg <<EOF
[modelinfo]
forced_surf_restore_opt = const
forced_surf_restore_const = 0.0
forced_surf_restore_rate_10m = 1.0 / 3600.0

forced_sms_opt = const
forced_sms_const = 1.0 / (365.0 * 86400.0)
EOF

cfg_fnames=$input_dir/newton_krylov.cfg,$input_dir/model_params.cfg,$workdir/override.cfg

# setup solver
echo running setup_solver.sh
./scripts/setup_solver.sh --model_name py_driver_2d --cfg_fnames $cfg_fnames \
    --workdir $workdir --newton_max_iter 10 --persist --fp_cnt 1 \
    --tracer_module_names forced_{suff}:iage $@

# run solver
echo running nk_driver.sh
$workdir/nk_driver.sh
