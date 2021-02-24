#!/bin/bash -i

set -e

cd `git rev-parse --show-toplevel`

source scripts/newton_krylov_env_cmds

# default model_name
model_name="test_problem"

# change model_name if it is specified with arguments
# do not pass model_name along to src.$model_name.setup_solver

args=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--model_name" ]]; then
        shift
        model_name="$1"
    else
        args+=("$1")
    fi
    shift
done

python -m src.$model_name.setup_solver "${args[@]}"
