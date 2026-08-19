 ```
 ______                     ____     __       ______  ____       
/\__  _\__                 /\  _`\  /\ \     /\  _  \/\  _`\     
\/_/\ \/\_\    ___   __  __\ \ \L\ \\ \ \    \ \ \L\ \ \,\L\_\   
   \ \ \/\ \ /' _ `\/\ \/\ \\ \  _ <'\ \ \  __\ \  __ \/_\__ \   
    \ \ \ \ \/\ \/\ \ \ \_\ \\ \ \L\ \\ \ \L\ \\ \ \/\ \/\ \L\ \ 
     \ \_\ \_\ \_\ \_\/`____ \\ \____/ \ \____/ \ \_\ \_\ `\____\
      \/_/\/_/\/_/\/_/`/___/> \\/___/   \/___/   \/_/\/_/\/_____/
                         /\___/                                  
                         \/__/                                   
```
## TinyBLAS
A minimal, dependency free, and performant BLAS implementation in C99.

This is meant as an educational BLAS-inspired library that does NOT include:
- historical conventions or semantics
- ABI compatibility
- Interop with other languages

## Building

```sh
make          # build/libtinyblas.a
make test     # build and run everything in tests/
make install  # PREFIX=/usr/local by default
```

`./test.sh` still works; it now just calls `make test`, so the compiler flags
live in exactly one place.

## Using it from another project

Installed:

```c
#include <tinyblas/tinyblas.h>
```
```sh
cc myprog.c -ltinyblas -lm
```

Or vendor it — the library is one translation unit, so dropping `src/` and
`headers/` into your tree and adding `-Iheaders` works just as well:

```c
#include "tinyblas.h"
```

`tinyblas.h` is the only header you need to include; it pulls in the rest.

## Implemented:
0.0.2 — levels 1, 2 and 3, dense, for all four types.

**Level 1** (42 routines) — everything except the modified Givens pair
(`rotm`/`rotmg`), which is exactly the historical semantics this library skips.

- dot: `ddot` `sdot` `dsdot` `sdsdot` `cdotu` `cdotc` `zdotu` `zdotc`
- norms: `snrm2` `dnrm2` `scnrm2` `dznrm2`
- abs sums: `sasum` `dasum` `scasum` `dzasum`
- max index: `isamax` `idamax` `icamax` `izamax`
- vector ops: `?swap` `?copy` `?axpy` `?scal` (+ `csscal` `zdscal`)
- rotations: `?rotg` `srot` `drot` `csrot` `zdrot`

**Level 2** (30 routines) — dense only.

- matrix-vector: `?gemv` `ssymv` `dsymv` `chemv` `zhemv`
- triangular: `?trmv` `?trsv`
- rank-1: `sger` `dger` `cgeru` `zgeru` `cgerc` `zgerc`
- symmetric and hermitian rank-1 and rank-2: `ssyr` `dsyr` `cher` `zher`
  `ssyr2` `dsyr2` `cher2` `zher2`

**Level 3** (30 routines) — dense only.

- general: `?gemm`
- symmetric and hermitian: `?symm` `chemm` `zhemm`
- rank-k and rank-2k: `?syrk` `cherk` `zherk` `?syr2k` `cher2k` `zher2k`
- triangular: `?trmm` `?trsm`

Not implemented, and not planned: **banded and packed storage**
(`?gbmv` `?tbmv` `?tpmv` `?sbmv` `?spmv` `?hbmv` `?hpmv` `?spr` `?hpr` …).
They are historical Fortran storage conventions, the same reason
`rotm`/`rotmg` is absent.

Where the C API departs from Fortran BLAS on purpose:

- `i?amax` returns a **0-based** index, and `-1` for an empty vector
- `?rotg` takes `a` and `b` by value and returns `c`, `s`, `r`; the packed `z`
  output is gone
- every vector and matrix argument is `restrict`: the operands must not overlap
- **row-major only.** `lda`/`ldb`/`ldc` are row strides, and there is no layout
  argument to get wrong
- the selectors are **enums, not `'N'`/`'T'`/`'U'` chars**. A compiler can check
  an enum; a mistyped char is a runtime bug that silently reads the wrong
  triangle
- `k == 0` still applies beta (`C := beta*C`), rather than being a no-op
- `beta == 0` means C is written and never read, so an uninitialised or
  NaN-filled C is legal input
- no `xerbla` and no error returns. Bad arguments are a programming error, not
  a runtime condition; `m`, `n <= 0` is simply a no-op

## Performance

The claim worth making is "comparable performance at a fraction of the code",
and that is only worth anything measured. `make bench` measures it against
OpenBLAS on the same machine, single threaded on both sides, and refuses to
print a number for a kernel that computes the wrong answer.

```sh
make bench                 # the whole table
./build/bench dgemm        # one routine
./build/bench gemm 512     # one family, one size
```

The `err` column is the difference from OpenBLAS divided by the naive forward
error bound, so one threshold works at every size. Correct implementations land
in single digits; the bench exits non-zero above 32. Without OpenBLAS installed
the comparison columns print `-` and `%peak` still prints, which is the honest
number and depends on nobody's package manager.

Measured on an i3-8350K at 4.0 GHz (4 cores, AVX2 + FMA, no AVX-512), GCC 13.3,
single thread. Peak is 64.0 GFLOP/s double, 128.0 single. The whole library is
about 6000 lines including headers and comments.

### gemm

GFLOP/s, tinyblas / OpenBLAS / percent of OpenBLAS:

| n | sgemm | dgemm | cgemm | zgemm |
|---|---|---|---|---|
| 512 | 100.7 / 111.1 / **91%** | 48.7 / 51.1 / **95%** | 90.1 / 113.8 / **79%** | 39.3 / 53.3 / **74%** |
| 1024 | 88.2 / 93.9 / **94%** | 37.8 / 46.2 / **82%** | 84.3 / 106.0 / **80%** | 33.8 / 51.4 / **66%** |
| 2048 | 98.7 / 113.6 / **87%** | 45.5 / 55.0 / **83%** | 81.5 / 106.6 / **77%** | 42.4 / 58.6 / **72%** |

The awkward sizes 257, 1000, 1023 and 1025 are in the table on purpose and show
no cliff, which is the zero-padded edge path doing its job. The remaining gap to
OpenBLAS is its hand-scheduled assembly with software pipelining and tuned
prefetch distances; GCC will not reliably match that from intrinsics.

### gemv

gemv moves `m*n` elements to do `2*m*n` flops, so it is a bandwidth benchmark in
a BLAS costume and GFLOP/s alone is misleading. The bench prints GB/s next to it
with a `memcpy` reference, which is the number that says whether there is room
left. Against a 32.9 GB/s memcpy, `dgemv` reaches 28.0 GB/s at n=2048 and tracks
OpenBLAS within 2% at every size — both are sitting on the same wall.

`cgemv` and `zgemv` were the exception. A single complex dot has only two
accumulator chains, which is not enough in flight to cover FMA latency, so they
sat at 12-13 and 16-24 GB/s against a ~33 GB/s ceiling: latency bound, not
bandwidth bound. Reading four rows at a time — the same trick the real gemv
already used — puts them at 20-34 and 22-43 GB/s, 40-80% and 50-86% of
OpenBLAS. Eight rows was tried and is slower; the accumulators spill. What is
left is the small in-cache sizes, where OpenBLAS runs a shuffle-free kernel
that GCC will not produce from interleaved complex loads.

### level 3, derived

Everything below gemm is the gemm core wearing a different hat, so the number
that matters is the fraction of gemm speed the wrapper keeps. At n=1024, side
LEFT, uplo UPPER, no transpose:

| routine | tinyblas | OpenBLAS | ratio | of gemm |
|---|---|---|---|---|
| dsymm  | 45.2 | 54.8 | 83% | 98% |
| dsyrk  | 38.1 | 48.8 | 78% | 83% |
| dsyr2k | 36.9 | 48.3 | 76% | 80% |
| dtrmm  | 22.4 | 50.9 | 44% | 49% |
| dtrsm  | 33.5 | 43.2 | 78% | 73% |
| zsymm  | 39.0 | 57.2 | 68% | 97% |
| zhemm  | 39.5 | 57.7 | 68% | 99% |
| zsyrk  | 32.9 | 54.9 | 60% | 82% |
| zherk  | 32.5 | 47.1 | 69% | 81% |
| zsyr2k | 30.5 | 54.8 | 56% | 76% |
| zher2k | 30.9 | 54.7 | 57% | 77% |
| ztrmm  | 19.6 | 53.4 | 37% | 49% |
| ztrsm  | 28.2 | 49.2 | 57% | 70% |

`trmm` is the one deliberate shortcut left: it expands the triangular operand
into a dense square and hands it to gemm, so it does twice the arithmetic a
triangular multiply needs. 49% of gemm is almost exactly what that predicts,
which means the expansion itself is nearly free and there is nothing left to win
short of a real triangular kernel. `trsm` is blocked for side LEFT; side RIGHT
still runs one `trsv` per row, because there it is B's columns that are strided
in row-major.

## references
- https://www.netlib.org/blas/blas.pdf
- https://github.com/Reference-LAPACK/lapack/tree/master/BLAS
- https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
- http://www.openmathlib.org/OpenBLAS/
- https://dl.acm.org/doi/abs/10.5555/578659
- https://scholar.google.com/citations?user=rldfxOMAAAAJ&hl=en
