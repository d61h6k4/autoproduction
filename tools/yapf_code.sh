#!/bin/bash

set -ex

ACTION=${1:---in-place}
[[ $ACTION == '--in-place' ]] || [[ $ACTION == '--diff' ]]

# change to root directory
cd "$(dirname "${0}")/.."

DIRS=(
    '.'
)

PYTHON=python3
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --upgrade futures
"$PYTHON" -m pip install yapf

$PYTHON -m yapf $ACTION --recursive --style=setup.cfg "${DIRS[@]}"
