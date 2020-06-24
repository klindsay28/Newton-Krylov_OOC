#!/bin/bash

model_name=$1

cfg_fname=input/$model_name/newton_krylov.cfg

echo cfg_fname=$cfg_fname

not_used_cnt=0

varnames=`cut -f1 -d'#' $cfg_fname | grep -i '^[a-z].*=' | cut -f1 -d=`
for varname in $varnames; do
    # check for usage outside of comments in src
    for fname in `ls src/*.py src/$model_name/*.py`; do
        cut -f1 -d'#' $fname | grep -q $varname
        if [ $? -eq 0 ]; then continue 2; fi
    done
    # check for interpolation usage outside of comments in cfg_fname
    cut -f1 -d'#' $cfg_fname | grep -q "%($varname)s"
    if [ $? -eq 0 ]; then continue; fi
    echo $varname not used
    not_used_cnt=$((not_used_cnt+1))
done

exit $not_used_cnt
