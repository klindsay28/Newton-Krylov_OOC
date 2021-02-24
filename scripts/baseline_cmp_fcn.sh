#!/bin/bash

set -e
set -x

for tracer_module_name in iage phosphorus "dye_decay_{suff}:100" "dye_decay_{suff}:010"; do
    workdir_baseline=$HOME/nk_fcn_baseline_$tracer_module_name
    workdir_expr=$HOME/nk_fcn_expr_$tracer_module_name
    rm -Rf $workdir_expr
    ./scripts/setup_solver.sh --fp_cnt 1 --nlevs 20 \
        --tracer_module_names "$tracer_module_name" --workdir $workdir_expr
    echo status=$?
    diff -I 'history =' \
        <(ncdump $workdir_baseline/gen_init_iterate/hist_00.nc) \
        <(ncdump $workdir_expr/gen_init_iterate/hist_00.nc)
    echo status=$?
done
