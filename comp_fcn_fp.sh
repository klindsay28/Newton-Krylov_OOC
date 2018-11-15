#!/bin/bash

set -x

./comp_fcn.py --hist_fname hist.nc iterate_00.nc fcn_00.nc
ncwa -O -a time -d time,-1 hist.nc iterate_00_fp1.nc

./comp_fcn.py --hist_fname hist_fp1.nc iterate_00_fp1.nc fcn_00_fp1.nc
ncwa -O -a time -d time,-1 hist_fp1.nc iterate_00_fp2.nc

./comp_fcn.py --hist_fname hist_fp2.nc iterate_00_fp2.nc fcn_00_fp2.nc
ncwa -O -a time -d time,-1 hist_fp2.nc iterate_00_fp3.nc

./comp_precond_jacobian_fcn_state_prod.py fcn_00.nc w_00.nc
./comp_precond_jacobian_fcn_state_prod.py fcn_00_fp1.nc w_00_fp1.nc
./comp_precond_jacobian_fcn_state_prod.py fcn_00_fp2.nc w_00_fp2.nc
