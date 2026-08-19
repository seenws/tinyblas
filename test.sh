#!/usr/bin/env bash
# Thin wrapper. The Makefile owns the build so the flags live in one place.
exec make -s test "$@"
