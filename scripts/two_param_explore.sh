#!/bin/bash -i

set -e

cd `git rev-parse --show-toplevel`

source scripts/newton_krylov_env_cmds

root_dir=$HOME/py_driver_2d_param_explore

input_dir=`pwd`/input/py_driver_2d

max_uptake_rate_vals=(0.10 0.15 0.20 0.25)
pop_sink_vel_vals=(2.0 1.5 1.0 0.5)

for ((mur_ind=0;mur_ind<${#max_uptake_rate_vals[*]};mur_ind++)) ; do
    max_uptake_rate=${max_uptake_rate_vals[$mur_ind]}
    for ((psv_ind=0;psv_ind<${#pop_sink_vel_vals[*]};psv_ind++)) ; do
        pop_sink_vel=${pop_sink_vel_vals[$psv_ind]}

        wdir=$root_dir/max_uptake_rate_${max_uptake_rate}_pop_sink_vel_${pop_sink_vel}
        echo $wdir
        mkdir -p $wdir

        # create override.cfg file
        echo "[modelinfo]" > $wdir/override.cfg
        echo "max_uptake_rate = $max_uptake_rate / 86400.0" >> $wdir/override.cfg
        echo "pop_sink_vel = $pop_sink_vel / 86400.0" >> $wdir/override.cfg

        if [[ $psv_ind -gt 0 ]]; then
            init_iterate_opt=$iter_prev_psv
        else
            if [[ $mur_ind -gt 0 ]]; then
                init_iterate_opt=$iter_prev_mur
            fi
        fi

        ./scripts/setup_solver.sh --model_name py_driver_2d --workdir $wdir \
            --tracer_module_names phosphorus --fp_cnt 2 --persist \
            --init_iterate_opt $init_iterate_opt \
            --cfg_fnames $input_dir/newton_krylov.cfg,$input_dir/model_params.cfg,$wdir/override.cfg
        $wdir/nk_driver.sh

        # store final iterate for subsequent solver invocations
        iter_prev_psv=`ls $wdir/iterate_??.nc | tail -n 1`
        if [[ $psv_ind -eq 0 ]]; then
            iter_prev_mur=$iter_prev_psv
        fi

        # solve for preformed po4

        wdir_pf=${wdir}_pf
        echo $wdir_pf
        mkdir -p $wdir_pf

        # create surf_restore_fname
        infile=`ls $wdir/hist_??.nc | tail -n 1`
        ncwa -O -a depth -d depth,0 -v po4 $infile $wdir_pf/po4_surf.nc

        # create init_iterate_fname
        infile=`ls $wdir/iterate_??.nc | tail -n 1`
        mkdir -p $wdir_pf/gen_init_iterate
        ncks -O -v po4 $infile $wdir_pf/gen_init_iterate/po4_net.nc
        ncrename -v po4,po4_pf $wdir_pf/gen_init_iterate/po4_net.nc

        # create override.cfg file
        echo "[modelinfo]" > $wdir_pf/override.cfg
        echo "surf_restore_fname = $wdir_pf/po4_surf.nc" >> $wdir_pf/override.cfg
        echo "surf_restore_varname = po4" >> $wdir_pf/override.cfg
        echo "init_iterate_fname = $wdir_pf/gen_init_iterate/po4_net.nc" >> $wdir_pf/override.cfg

        ./scripts/setup_solver.sh --model_name py_driver_2d --workdir $wdir_pf \
            --tracer_module_names preformed --fp_cnt 2 --persist \
            --cfg_fnames $input_dir/newton_krylov.cfg,$input_dir/model_params.cfg,$wdir_pf/override.cfg
        $wdir_pf/nk_driver.sh
    done
done
