#!/bin/bash

set -x
set -e

toplevel_dir=`git rev-parse --show-toplevel`

cd $toplevel_dir
newton_fcn_module=src.test_problem.newton_fcn_test_problem

test_problem_dir=$toplevel_dir/models/test_problem
workdir=$test_problem_dir/comp_fcn_fp_work

rm -Rf $workdir
mkdir $workdir

fp_ind=0
fp_cnt=3

cp $test_problem_dir/iterate_test_00.nc \
    $workdir/iterate_test_00_fp${fp_ind}.nc

while [ $fp_ind -lt $fp_cnt ]; do
    python -m $newton_fcn_module comp_fcn --workdir $workdir \
        --cfg_fname $test_problem_dir/newton_krylov.cfg \
        --hist_fname hist_00_fp${fp_ind}.nc \
        --in_fname iterate_test_00_fp${fp_ind}.nc \
        --res_fname fcn_test_00_fp${fp_ind}.nc
    python -m $newton_fcn_module gen_precond_jacobian --workdir $workdir \
        --cfg_fname $test_problem_dir/newton_krylov.cfg \
        --hist_fname hist_00_fp${fp_ind}.nc \
        --in_fname iterate_test_00_fp${fp_ind}.nc \
        --precond_fname precond_00_fp${fp_ind}.nc
    python -m $newton_fcn_module apply_precond_jacobian --workdir $workdir \
        --cfg_fname $test_problem_dir/newton_krylov.cfg \
        --in_fname fcn_test_00_fp${fp_ind}.nc \
        --precond_fname precond_00_fp${fp_ind}.nc \
        --res_fname w_test_00_fp${fp_ind}.nc

    fp_ind_p1=`expr $fp_ind + 1`

    ncwa -h -O -a time -d time,-1 \
        $workdir/hist_00_fp${fp_ind}.nc \
        $workdir/iterate_test_00_fp${fp_ind_p1}.nc

    fp_ind=$fp_ind_p1
done

cp $workdir/iterate_test_00_fp${fp_ind_p1}.nc $test_problem_dir
