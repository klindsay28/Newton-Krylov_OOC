#!/bin/bash

# example function that Newton's method is being applied to

iterate_fname=$1
res_fname=$2

ncap2 -O -s "x1=cos(x1)-0.7*x1;x2=cos(x2)-0.7*x2" $iterate_fname $res_fname

./postrun.sh
