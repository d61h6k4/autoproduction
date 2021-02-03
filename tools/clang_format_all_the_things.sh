#!/bin/bash

set -ex

# change to root directory
cd "$(dirname "$0")/.."

# directories to run against
DIRS="."

# file matching patterns to check
GLOB="*.h *.cc"

# clang format command
CLANG_FORMAT=${CLANG_FORMAT:-clang-format}

files=
for dir in $DIRS
do
  for glob in $GLOB
  do
    files="$files `find $dir -name $glob`"
  done
done

if [ "$TEST" == "" ]
then
  echo $files | xargs $CLANG_FORMAT -i
else
  ok=yes
  for file in $files
  do
    tmp=`mktemp`
    $CLANG_FORMAT $file > $tmp
    diff -u $file $tmp || ok=no
    rm $tmp
  done
  if [ $ok == no ]
  then
    false
  fi
fi
