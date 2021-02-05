#!/usr/bin/env bash


set -o errexit
set -o nounset
set -o xtrace
set -o pipefail

declare -a bazel_build_flags

bazel_build_flags+=(
  # Useful to determine how long individual steps are taking in CI.
  '--show_timestamps'
)


# Generates a compile_commands.json file at $(bazel info execution_root) for
# your Clang tooling needs.

readonly FILE_NAME="compile_commands.json"
readonly BAZEL_ROOT="$(bazel info execution_root)"
cd "$BAZEL_ROOT" || exit 1
echo '[' > "$FILE_NAME"
COUNT=0
find . -name '*.compile_command.json' -print0 | while read -r -d '' fname; do
  if ((COUNT++)); then
    echo ',' >> "$FILE_NAME"
  fi
  sed -e "s|@BAZEL_ROOT@|$BAZEL_ROOT|g" < "$fname" | \
  # Clang doesn't recognize '-fno-canonical-system-headers'.
  sed -e "s|-fno-canonical-system-headers|-no-canonical-prefixes|g" >> "$FILE_NAME"
done
echo ']' >> "$FILE_NAME"
