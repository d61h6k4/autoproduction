#!/bin/env bash

set=-ex

PYTHON=`which python`
CONFIG_PATH=`pwd`/"$(dirname ${0})/bazel_style.cfg"

pushd "$(dirname "${0}")/.."
FILES=$(find . -path ./third_party -prune -o -name '*.bzl' -print)
echo "${FILES}"
echo "${FILES}" | xargs "$PYTHON" -m yapf -i --style="${CONFIG_PATH}"

if ! which buildifier &>/dev/null; then
  go get github.com/bazelbuild/buildtools/buildifier
fi

echo "${FILES}" | xargs buildifier --type=bzl

popd
