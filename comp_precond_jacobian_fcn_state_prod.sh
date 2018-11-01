#!/bin/bash

# compute product of preconditioner that approximates the inverse of the Jacobian of comp_fcn.sh
# with ModelState in in_frame

in_fname=$1
res_fname=$2

ncks -O $in_fname $res_fname

./postrun.sh
