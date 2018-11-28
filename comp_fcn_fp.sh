#!/bin/bash

set -x

newton_fcn_script=newton_fcn_test_problem.py
workdir=comp_fcn_fp_work

rm -Rf $workdir
mkdir $workdir
cp iterate_test_00.nc $workdir

./$newton_fcn_script comp_fcn --workdir $workdir --hist_fname hist.nc --in_fname iterate_test_00.nc --res_fname fcn_test_00.nc
./$newton_fcn_script gen_precond_jacobian --workdir $workdir --hist_fname hist.nc
./$newton_fcn_script apply_precond_jacobian --workdir $workdir --in_fname fcn_test_00.nc --res_fname w_test_00.nc

ncwa -h -O -a time -d time,-1 $workdir/hist.nc $workdir/iterate_test_00_fp1.nc

./$newton_fcn_script comp_fcn --workdir $workdir --hist_fname hist_fp1.nc --in_fname iterate_test_00_fp1.nc --res_fname fcn_test_00_fp1.nc
./$newton_fcn_script gen_precond_jacobian --workdir $workdir --hist_fname hist_fp1.nc
./$newton_fcn_script apply_precond_jacobian --workdir $workdir --in_fname fcn_test_00_fp1.nc --res_fname w_test_00_fp1.nc

ncwa -h -O -a time -d time,-1 $workdir/hist_fp1.nc $workdir/iterate_test_00_fp2.nc

./$newton_fcn_script comp_fcn --workdir $workdir --hist_fname hist_fp2.nc --in_fname iterate_test_00_fp2.nc --res_fname fcn_test_00_fp2.nc
./$newton_fcn_script gen_precond_jacobian --workdir $workdir --hist_fname hist_fp2.nc
./$newton_fcn_script apply_precond_jacobian --workdir $workdir --in_fname fcn_test_00_fp2.nc --res_fname w_test_00_fp2.nc

ncwa -h -O -a time -d time,-1 $workdir/hist_fp2.nc $workdir/iterate_test_00_fp3.nc

cp $workdir/iterate_test_00_fp3.nc .
