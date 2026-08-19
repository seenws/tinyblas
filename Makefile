CC     ?= cc
AR     ?= ar
PREFIX ?= /usr/local

# Reduction vectorization needs reassociation. Narrower than -ffast-math on
# purpose: NaN/Inf still propagate, which a BLAS caller is entitled to expect.
CFLAGS ?= -std=iso9899:1999 -O3 -march=native \
          -fassociative-math -fno-signed-zeros -fno-trapping-math

WARN    := -Wall -Wextra -pedantic -Wshadow -Wcast-qual -Wpointer-arith \
           -Wstrict-prototypes -Wmissing-prototypes -Wconversion
INCLUDE := -Iheaders

LIB      := build/libtinyblas.a
HEADERS  := $(wildcard headers/*.h)
# Private headers are a compile dependency but are never installed.
DEPS     := $(HEADERS) $(wildcard src/*.h)
SRC      := $(wildcard src/*.c)
OBJ      := $(SRC:src/%.c=build/%.o)
TESTS    := $(patsubst tests/%.c,build/%,$(wildcard tests/test_*.c))
BENCHES  := $(patsubst bench/%.c,build/%,$(wildcard bench/*.c))

# Optional: the bench builds and runs with or without OpenBLAS, it just loses
# the comparison column when the baseline is missing. Only probe pkg-config
# when the caller has not spoken, so that both `OB_LIBS= make bench` and
# `make OB_LIBS= bench` force the no-baseline path. A plain := here would
# silently ignore the environment, which makes the documented escape hatch a
# lie.
ifeq ($(origin OB_LIBS),undefined)
    OB_LIBS := $(shell pkg-config --libs openblas 2>/dev/null)
endif
OB_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
ifneq ($(strip $(OB_LIBS)),)
    OB_CFLAGS += -DTINYBLAS_HAVE_OPENBLAS
endif

GREEN := \033[0;32m
RED   := \033[0;31m
RESET := \033[0m

all: $(LIB)

lib: $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

build/%.o: src/%.c $(DEPS) | build
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(WARN) $(INCLUDE) -c $< -o $@

build/test_%: tests/test_%.c $(LIB) | build
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(WARN) $(INCLUDE) $< $(LIB) -lm -o $@

build/%: bench/%.c $(LIB) | build
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $(WARN) $(INCLUDE) $(OB_CFLAGS) $< $(LIB) $(OB_LIBS) -lm -o $@

build:
	@mkdir -p build

# An empty tests/ once passed silently. Never again.
test: $(TESTS)
	@test -n "$(TESTS)" || { printf '$(RED)[FAIL]$(RESET) no tests found\n'; exit 1; }
	@for t in $(TESTS); do \
	    printf '[run] %s\n' "$$t"; \
	    if "$$t"; then printf '$(GREEN)[Pass]$(RESET) %s\n\n' "$$t"; \
	    else printf '$(RED)[Fail]$(RESET) %s\n' "$$t"; exit 1; fi; \
	done
	@printf '$(GREEN)[PASS]$(RESET) all tests passed\n'

# Single-threaded on both sides: the point is kernel quality, not core count.
bench: $(BENCHES)
	@test -n "$(BENCHES)" || { printf '$(RED)[FAIL]$(RESET) no benchmarks found\n'; exit 1; }
	@for b in $(BENCHES); do OPENBLAS_NUM_THREADS=1 "$$b" $(ARGS) || exit 1; done

install: $(LIB)
	install -d $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include/tinyblas
	install -m 644 $(LIB) $(DESTDIR)$(PREFIX)/lib
	install -m 644 $(HEADERS) $(DESTDIR)$(PREFIX)/include/tinyblas

clean:
	rm -rf build

.PHONY: all lib test bench install clean
