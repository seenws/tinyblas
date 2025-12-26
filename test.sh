#!/usr/bin/env bash

set -euo pipefail

CC=${CC:-cc}
CFLAGS="-std=iso9899:1999 -Wall -Wextra -pedantic -Wshadow -Wcast-qual -Wpointer-arith -Wstrict-prototypes -Wmissing-prototypes -Wconversion"
INCLUDE="-Iinclude"

BUILD_DIR="build"

GREEN='\033[0;32m'
RED='\033[0;31m'
RESET='\033[0m'

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

    if "$test_bin"; then
        echo -e "${GREEN}[Pass]${RESET} $test_name"
    else
        echo -e "${RED}[Fail]${RESET} $test_name"
        exit 1
    fi

    echo
done

echo -e "${GREEN}[PASS]${RESET} all tests passed"

