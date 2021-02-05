#!/bin/env bash

set -ex

# change to root directory
cd "$(dirname "$0")/.."

DIRS=(
    '.'
)


PYTHON=python

$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install --upgrade pylint


EXIT=0
find $DIRS \
  -iname "*.py" \
  | xargs $PYTHON -m pylint --rcfile=.pylintrc || EXIT=1

exit $EXIT
