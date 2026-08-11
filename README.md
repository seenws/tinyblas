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

## Implemented:
0.0.1 Work in progress — all of level 1 except the modified Givens pair
(`rotm`/`rotmg`), which is exactly the historical semantics this library skips.

- dot: `ddot` `sdot` `dsdot` `sdsdot` `cdotu` `cdotc` `zdotu` `zdotc`
- norms: `snrm2` `dnrm2` `scnrm2` `dznrm2`
- abs sums: `sasum` `dasum` `scasum` `dzasum`
- max index: `isamax` `idamax` `icamax` `izamax`
- vector ops: `?swap` `?copy` `?axpy` `?scal` (+ `csscal` `zdscal`)
- rotations: `?rotg` `srot` `drot` `csrot` `zdrot`

Where the C API departs from Fortran BLAS on purpose:

- `i?amax` returns a **0-based** index, and `-1` for an empty vector
- `?rotg` takes `a` and `b` by value and returns `c`, `s`, `r`; the packed `z`
  output is gone
- every vector argument is `restrict`: x and y must not overlap

## references
- https://www.netlib.org/blas/blas.pdf
- https://github.com/Reference-LAPACK/lapack/tree/master/BLAS
- https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
- http://www.openmathlib.org/OpenBLAS/
- https://dl.acm.org/doi/abs/10.5555/578659
- https://scholar.google.com/citations?user=rldfxOMAAAAJ&hl=en
