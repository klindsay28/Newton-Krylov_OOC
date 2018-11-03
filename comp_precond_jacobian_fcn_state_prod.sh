#!/bin/bash

# compute product of preconditioner that approximates the inverse of the Jacobian of comp_fcn.sh
# with ModelState in in_frame

cfg_fname=$1
in_fname=$2
res_fname=$3

ncks -O $in_fname $res_fname

./postrun.sh $cfg_fname
