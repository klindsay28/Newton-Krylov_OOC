#!/bin/bash

set -x
set -e

repo_root=`git rev-parse --show-toplevel`

cd $repo_root
newton_fcn_module=src.test_problem.newton_fcn_test_problem

test_problem_dir=$repo_root/models/test_problem
fname_dir=$test_problem_dir/comp_fcn_fp_work

if [ -z ${PYTHONPATH+x} ]; then
    export PYTHONPATH=models
else
    export PYTHONPATH=models:$PYTHONPATH
fi

# clear out results from previous invocation
rm -Rf $fname_dir
mkdir $fname_dir

fp_ind=0
fp_cnt=3

python -m $newton_fcn_module gen_ic --fname_dir $fname_dir \
    --model test_problem \
    --res_fname iterate_test_00_fp${fp_ind}.nc

while [ $fp_ind -lt $fp_cnt ]; do
    python -m $newton_fcn_module comp_fcn --fname_dir $fname_dir \
        --model test_problem \
        --hist_fname hist_00_fp${fp_ind}.nc \
        --in_fname iterate_test_00_fp${fp_ind}.nc \
        --res_fname fcn_test_00_fp${fp_ind}.nc
    python -m $newton_fcn_module gen_precond_jacobian --fname_dir $fname_dir \
        --model test_problem \
        --hist_fname hist_00_fp${fp_ind}.nc \
        --in_fname iterate_test_00_fp${fp_ind}.nc \
        --precond_fname precond_00_fp${fp_ind}.nc
    python -m $newton_fcn_module apply_precond_jacobian --fname_dir $fname_dir \
        --model test_problem \
        --in_fname fcn_test_00_fp${fp_ind}.nc \
        --precond_fname precond_00_fp${fp_ind}.nc \
        --res_fname w_test_00_fp${fp_ind}.nc

    fp_ind_p1=`expr $fp_ind + 1`

    ncwa -h -O -a time -d time,-1 \
        $fname_dir/hist_00_fp${fp_ind}.nc \
        $fname_dir/iterate_test_00_fp${fp_ind_p1}.nc

    fp_ind=$fp_ind_p1
done

cp $fname_dir/iterate_test_00_fp${fp_ind_p1}.nc $test_problem_dir
