#!/bin/bash

# example function that Newton's method is being applied to

iterate_fname=$1
res_fname=$2

ncap2 -O -s "x1=cos(x1)-0.7*x1;x2=cos(x2)-0.8*x2;x3=cos(x3)-0.9*x3;x4=cos(x4)-x4" $iterate_fname $res_fname

./postrun.sh
