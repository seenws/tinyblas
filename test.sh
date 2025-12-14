#!/usr/bin/env bash

set -euo pipefail

CC=${CC:-cc}
CFLAGS="-std=iso9899:1999 -Wall -Wextra -pedantic -Wshadow -Wcast-qual -Wpointer-arith -Wstrict-prototypes -Wmissing-prototypes -Wconversion"
INCLUDE="-Iinclude"

echo "[build] tinyblas ddot test"

$CC $CFLAGS $INCLUDE \
    src/tinyblas_level1.c \
    tests/test_ddot.c \
    -lm \
    -o test_ddot

echo "[run] ddot unit tests"
./test_ddot

echo "[ok] all tests passed"
