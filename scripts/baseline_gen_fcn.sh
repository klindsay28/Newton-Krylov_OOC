#!/bin/bash

set -e
set -x

for tracer_module_name in iage phosphorus "dye_decay_{suff}:100" "dye_decay_{suff}:010"; do
    workdir_baseline=$HOME/nk_fcn_baseline_$tracer_module_name
    rm -Rf $workdir_baseline
    ./scripts/setup_solver.sh --fp_cnt 1 --nlevs 20 \
        --tracer_module_names "$tracer_module_name" --workdir $workdir_baseline
    echo status=$?
done
