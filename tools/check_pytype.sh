#!/bin/env bash

set -ex;

JOBS=$(nproc) || JOBS=4
python3 -m pip install pytype

python3 -m pytype --python-version 3.8 --keep-going -j "$JOBS" --strict-import --config "setup.cfg"
