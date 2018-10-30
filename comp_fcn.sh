#!/bin/bash

# example function that Newton's method is being applied to

iterate_fname=$1
res_fname=$2

ncap2 -O -s "sx=array(0.7,0.05,x1); x1=cos(x1)-sx*x1; x2=cos(x2)-sx*x2; sy=array(1.0,.05,y); y=cos(y)-sy*y" $iterate_fname $res_fname

./postrun.sh
