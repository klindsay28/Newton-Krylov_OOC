#!/bin/bash -i

set -e

source scripts/newton_krylov_env_cmds

workdir=$HOME/py_driver_2d_forced_preformed_po4
rm -Rf $workdir
mkdir -p $workdir

input_dir=`pwd`/input/py_driver_2d

# create override.cfg file
cat > $workdir/override.cfg <<EOF
[modelinfo]
forced_surf_restore_opt = file
forced_surf_restore_fname = $input_dir/po4_surf.nc
forced_surf_restore_varname = po4
forced_surf_restore_rate_10m = 1.0 / 3600.0

forced_sms_opt = none
EOF

cfg_fnames=$input_dir/newton_krylov.cfg,$input_dir/model_params.cfg,$workdir/override.cfg

# setup solver
echo running setup_solver.sh
./scripts/setup_solver.sh --model_name py_driver_2d --cfg_fnames $cfg_fnames \
    --workdir $workdir --newton_max_iter 10 --tracer_module_names forced --persist --fp_cnt 1

# run solver
echo running nk_driver.sh
$workdir/nk_driver.sh
