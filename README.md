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
make bench    # build and run bench/bench.c
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
(`rotm`/`rotmg`)

- dot: `ddot` `sdot` `dsdot` `sdsdot` `cdotu` `cdotc` `zdotu` `zdotc`
- norms: `snrm2` `dnrm2` `scnrm2` `dznrm2`
- abs sums: `sasum` `dasum` `scasum` `dzasum`
- max index: `isamax` `idamax` `icamax` `izamax`
- vector ops: `?swap` `?copy` `?axpy` `?scal` (+ `csscal` `zdscal`)
- rotations: `?rotg` `srot` `drot` `csrot` `zdrot`

**Level 2** (30 routines) (dense only)

- matrix-vector: `?gemv` `ssymv` `dsymv` `chemv` `zhemv`
- triangular: `?trmv` `?trsv`
- rank-1: `sger` `dger` `cgeru` `zgeru` `cgerc` `zgerc`
- symmetric and hermitian rank-1 and rank-2: `ssyr` `dsyr` `cher` `zher`
  `ssyr2` `dsyr2` `cher2` `zher2`

**Level 3** (30 routines) (dense only)

- general: `?gemm`
- symmetric and hermitian: `?symm` `chemm` `zhemm`
- rank-k and rank-2k: `?syrk` `cherk` `zherk` `?syr2k` `cher2k` `zher2k`
- triangular: `?trmm` `?trsm`

Not implemented, and not planned: **banded and packed storage**
(`?gbmv` `?tbmv` `?tpmv` `?sbmv` `?spmv` `?hbmv` `?hpmv` `?spr` `?hpr` …).

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

```sh
make bench                 # the whole table
./build/bench dgemm        # one routine
./build/bench gemm 512     # one family with defined size
```

The `err` column is the difference from OpenBLAS divided by the naive forward
error bound. Correct implementations land
in single digits; the bench exits non-zero above 32. Without OpenBLAS installed
the comparison columns print `-` and `%peak` still prints.

Measured on an i3-8350K at 4.0 GHz (4 cores, AVX2 + FMA, no AVX-512), GCC 13.3,
single thread. Peak is 64.0 GFLOP/s double, 128.0 single. The whole library is
about 6900 lines including headers and comments.

Every number below is the **median of three full `make bench` runs**. One run is
not trustworthy on this box: sustained AVX2 throttles the chip, and a single
pass put `sgemm` at n=1025 at 41% of OpenBLAS where the median is 82%. Ratios to
OpenBLAS hold up far better than raw GFLOP/s, because both sides throttle
together.

### gemm

GFLOP/s, tinyblas / OpenBLAS / percent of OpenBLAS:

| n | sgemm | dgemm | cgemm | zgemm |
|---|---|---|---|---|
| 512 | 113.9 / 125.4 / **91%** | 51.9 / 59.0 / **87%** | 101.3 / 127.8 / **80%** | 45.0 / 61.2 / **73%** |
| 1024 | 102.5 / 118.2 / **87%** | 48.8 / 54.5 / **87%** | 93.1 / 119.7 / **78%** | 44.3 / 60.6 / **72%** |
| 2048 | 101.2 / 123.3 / **85%** | 49.1 / 59.6 / **82%** | 96.2 / 124.3 / **77%** | 43.0 / 62.2 / **69%** |

### gemv

gemv moves `m*n` elements to do `2*m*n` flops, so it is a bandwidth benchmark and GFLOP/s alone is misleading.
The bench prints GB/s next to it with a `memcpy` reference, which is the number that says whether there is room
left. Against a ~34 GB/s memcpy, `dgemv` reaches 29.0 GB/s at n=2048 and lands
between 96% and 125% of OpenBLAS across the sweep.

`cgemv` and `zgemv` were the exception. A single complex dot has only two
accumulator chains, which is not enough in flight to cover FMA latency, so they sat at 12-13 and 16-24 GB/s.
Reading four rows at a time puts them at 22-39 and 23-45 GB/s, 45-78% and 66-97% of OpenBLAS.
Eight rows was tried and is slower; the accumulators spill. What is left is the small in-cache sizes, where
OpenBLAS runs a shuffle-free kernel that GCC will not produce from interleaved complex loads.

### level 3, derived

| routine | tinyblas | OpenBLAS | ratio | of gemm |
|---|---|---|---|---|
| dsymm  | 43.1 | 56.7 | 76% | 92% |
| dsyrk  | 37.9 | 51.4 | 74% | 79% |
| dsyr2k | 36.8 | 49.6 | 72% | 78% |
| dtrmm  | 23.0 | 53.5 | 42% | 48% |
| dtrsm  | 33.0 | 46.1 | 71% | 68% |
| zsymm  | 43.4 | 61.3 | 72% | 96% |
| zhemm  | 43.7 | 61.5 | 72% | 97% |
| zsyrk  | 40.5 | 60.3 | 68% | 90% |
| zherk  | 40.5 | 59.0 | 69% | 89% |
| zsyr2k | 40.4 | 59.6 | 67% | 89% |
| zher2k | 40.4 | 59.5 | 67% | 88% |
| ztrmm  | 21.1 | 58.2 | 36% | 46% |
| ztrsm  | 34.8 | 54.5 | 63% | 76% |

## references
- https://www.netlib.org/blas/blas.pdf
- https://github.com/Reference-LAPACK/lapack/tree/master/BLAS
- https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
- http://www.openmathlib.org/OpenBLAS/
- https://dl.acm.org/doi/abs/10.5555/578659
- https://scholar.google.com/citations?user=rldfxOMAAAAJ&hl=en
