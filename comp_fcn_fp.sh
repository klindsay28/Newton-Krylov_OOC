#!/bin/bash

set -x

newton_fcn_script=newton_fcn_test_problem.py

./$newton_fcn_script comp_fcn --hist_fname hist.nc iterate_test_00.nc fcn_test_00.nc
ncwa -O -a time -d time,-1 hist.nc iterate_test_00_fp1.nc

./$newton_fcn_script comp_fcn --hist_fname hist_fp1.nc iterate_test_00_fp1.nc fcn_test_00_fp1.nc
ncwa -O -a time -d time,-1 hist_fp1.nc iterate_test_00_fp2.nc

./$newton_fcn_script comp_fcn --hist_fname hist_fp2.nc iterate_test_00_fp2.nc fcn_test_00_fp2.nc
ncwa -O -a time -d time,-1 hist_fp2.nc iterate_test_00_fp3.nc

./$newton_fcn_script apply_precond_jacobian fcn_test_00.nc w_test_00.nc
./$newton_fcn_script apply_precond_jacobian fcn_test_00_fp1.nc w_test_00_fp1.nc
./$newton_fcn_script apply_precond_jacobian fcn_test_00_fp2.nc w_test_00_fp2.nc
