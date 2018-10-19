#!/bin/bash

# example function that Newton's method is being applied to

iterate_fname=$1
res_fname=$2

ncap2 -O -s "x=cos(x)-0.7*x" $iterate_fname $res_fname

./postrun.sh
