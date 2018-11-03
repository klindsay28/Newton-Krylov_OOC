#!/bin/bash

# example function that Newton's method is being applied to

cfg_fname=$1
in_fname=$2
res_fname=$3

./comp_fcn.py --cfg_fname $cfg_fname $in_fname $res_fname

./postrun.sh $cfg_fname
