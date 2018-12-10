#!/bin/bash

set -x
set -e

newton_fcn_script=newton_fcn_test_problem.py
workdir=comp_fcn_fp_work

rm -Rf $workdir
mkdir $workdir

fp_ind=0
fp_cnt=3

cp iterate_test_00.nc $workdir/iterate_test_00_fp${fp_ind}.nc

while [ $fp_ind -lt $fp_cnt ]; do
    ./$newton_fcn_script comp_fcn --workdir $workdir \
        --hist_fname hist_00_fp${fp_ind}.nc \
        --in_fname iterate_test_00_fp${fp_ind}.nc \
        --res_fname fcn_test_00_fp${fp_ind}.nc
    ./$newton_fcn_script gen_precond_jacobian --workdir $workdir \
        --hist_fname hist_00_fp${fp_ind}.nc \
        --precond_fname precond_00_fp${fp_ind}.nc
    ./$newton_fcn_script apply_precond_jacobian --workdir $workdir \
        --in_fname fcn_test_00_fp${fp_ind}.nc \
        --precond_fname precond_00_fp${fp_ind}.nc \
        --res_fname w_test_00_fp${fp_ind}.nc

    fp_ind_p1=`expr $fp_ind + 1`

    ncwa -h -O -a time -d time,-1 \
        $workdir/hist_00_fp${fp_ind}.nc \
        $workdir/iterate_test_00_fp${fp_ind_p1}.nc

    fp_ind=$fp_ind_p1
done

cp $workdir/iterate_test_00_fp${fp_ind_p1}.nc .
