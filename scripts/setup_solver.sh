#!/bin/bash -i

set -e

cd `git rev-parse --show-toplevel`

source scripts/newton_krylov_env_cmds

# default model_name
model_name="test_problem"

# see if model_name is specified in arguments
# do not alter arguments, e.g. with shift, so that they can be passed along
for (( j=0; j<$#; j++ )); do
    if [[ "${!j}" == "--model_name" ]]; then
        (( j++ ))
        model_name="${!j}"
    fi
done

python -m src.$model_name.setup_solver "$@"
