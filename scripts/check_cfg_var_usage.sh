#!/bin/bash

model_name=$1

not_used_cnt=0

for cfg_fname in `ls input/$model_name/*.cfg`; do

    echo cfg_fname=$cfg_fname

    varnames=`cut -f1 -d'#' $cfg_fname | grep -i '^[a-z].*=' | cut -f1 -d=`
    for varname in $varnames; do
        # check for usage outside of comments in nk_ooc
        for fname in `ls nk_ooc/*.py nk_ooc/$model_name/*.py`; do
            python -m tokenize $fname | awk '{print $2,$3}' | grep -E '^NAME|^STRING' \
                | awk '{print $2}' | cut -f2 -d\' | cut -f2 -d\"  | grep -q "^$varname$"
            if [ $? -eq 0 ]; then continue 2; fi
        done
        # check for interpolation usage outside of comments in cfg_fname
        cut -f1 -d'#' $cfg_fname | grep -q "%($varname)s"
        if [ $? -eq 0 ]; then continue; fi
        echo $varname not used
        not_used_cnt=$((not_used_cnt+1))
    done

done

echo not_used_cnt = $not_used_cnt

exit 0
