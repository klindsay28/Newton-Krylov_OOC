#!/bin/bash

# example function that Newton's method is being applied to

iterate_fname=$1
res_fname=$2

ncap2 -O -s "sx=array(0.7,0.05,x); x=cos(x)-sx*x; sy=array(1.0,.05,y); y=cos(y)-sy*y" $iterate_fname $res_fname

./postrun.sh
