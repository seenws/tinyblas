#!/usr/bin/env bash

set -euo pipefail

CC=${CC:-cc}
CFLAGS="-std=iso9899:1999 -Wall -Wextra -pedantic -Wshadow -Wcast-qual -Wpointer-arith -Wstrict-prototypes -Wmissing-prototypes -Wconversion"
INCLUDE="-Iinclude"

BUILD_DIR="build"

mkdir -p "$BUILD_DIR"

echo "[build] tinyblas tests"

for test_src in tests/*.c; do
    test_name=$(basename "$test_src" .c)
    test_bin="$BUILD_DIR/$test_name"

    echo "[build] $test_name"

    $CC $CFLAGS $INCLUDE \
        src/tinyblas_level1.c \
        "$test_src" \
        -lm \
        -o "$test_bin"

    echo "[run] $test_name"
    "$test_bin"

    echo "[ok] $test_name passed"
    echo
done

echo "[ok] all tests passed"

