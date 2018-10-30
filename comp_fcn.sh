#!/bin/bash

# example function that Newton's method is being applied to

iterate_fname=$1
res_fname=$2

cat > comp_fcn.nco << EOF
sx1=array(0.7,0.02,x1);
x1=cos(x1)-sx1*x1;
sx2=array(0.8,.02,x2);
x2=cos(x2)-sx2*x2;
sy=array(0.9,.02,y);
y=cos(y)-sy*y;
EOF

ncap2 -O -S comp_fcn.nco $iterate_fname $res_fname

./postrun.sh
