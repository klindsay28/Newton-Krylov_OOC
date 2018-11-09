#!/bin/bash

# compute product of preconditioner that approximates the inverse of the Jacobian of comp_fcn.sh
# with ModelState in in_frame

cfg_fname=$1
in_fname=$2
res_fname=$3

./comp_precond_jacobian_fcn_state_prod.py --cfg_fname $cfg_fname $in_fname $res_fname

./postrun.sh $cfg_fname
