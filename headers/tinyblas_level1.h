// MIT License
//
// Copyright (c) 2025 Sinan Olsson-Pasic
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TINYBLAS_LEVEL_1_H_
#define TINYBLAS_LEVEL_1_H_

#include <complex.h>
#include <stdint.h>

/* Conventions, deliberately un-Fortran:
 *   - any increment is allowed; a negative one walks the vector backwards
 *   - n <= 0 is a no-op: reductions return 0, i?amax returns -1
 *   - i?amax returns a 0-based index
 *   - ?rotg takes its inputs by value and drops the packed `z` output
 *   - x and y must not overlap (every vector argument is restrict)
 */

/* dot products */
double         tinyblas_ddot  (int32_t n, const double *restrict dx, int32_t incx, const double *restrict dy, int32_t incy);
float          tinyblas_sdot  (int32_t n, const float *restrict sx, int32_t incx, const float *restrict sy, int32_t incy);
double         tinyblas_dsdot (int32_t n, const float *restrict dx, int32_t incx, const float *restrict dy, int32_t incy);
float          tinyblas_sdsdot(int32_t n, float sb, const float *restrict sx, int32_t incx, const float *restrict sy, int32_t incy);
float complex  tinyblas_cdotu (int32_t n, const float complex *restrict zx, int32_t incx, const float complex *restrict zy, int32_t incy);
float complex  tinyblas_cdotc (int32_t n, const float complex *restrict zx, int32_t incx, const float complex *restrict zy, int32_t incy);
double complex tinyblas_zdotu (int32_t n, const double complex *restrict zx, int32_t incx, const double complex *restrict zy, int32_t incy);
double complex tinyblas_zdotc (int32_t n, const double complex *restrict zx, int32_t incx, const double complex *restrict zy, int32_t incy);

/* euclidean norms */
float          tinyblas_snrm2 (int32_t n, const float *restrict sx, int32_t incx);
double         tinyblas_dnrm2 (int32_t n, const double *restrict dx, int32_t incx);
float          tinyblas_scnrm2(int32_t n, const float complex *restrict cx, int32_t incx);
double         tinyblas_dznrm2(int32_t n, const double complex *restrict zx, int32_t incx);

/* sums of absolute values; the complex forms use |re| + |im|, not the modulus */
float          tinyblas_sasum (int32_t n, const float *restrict sx, int32_t incx);
double         tinyblas_dasum (int32_t n, const double *restrict dx, int32_t incx);
float          tinyblas_scasum(int32_t n, const float complex *restrict cx, int32_t incx);
double         tinyblas_dzasum(int32_t n, const double complex *restrict zx, int32_t incx);

/* index of the first element of largest absolute value, 0-based */
int32_t        tinyblas_isamax(int32_t n, const float *restrict sx, int32_t incx);
int32_t        tinyblas_idamax(int32_t n, const double *restrict dx, int32_t incx);
int32_t        tinyblas_icamax(int32_t n, const float complex *restrict cx, int32_t incx);
int32_t        tinyblas_izamax(int32_t n, const double complex *restrict zx, int32_t incx);

/* x <-> y */
void tinyblas_sswap(int32_t n, float *restrict sx, int32_t incx, float *restrict sy, int32_t incy);
void tinyblas_dswap(int32_t n, double *restrict dx, int32_t incx, double *restrict dy, int32_t incy);
void tinyblas_cswap(int32_t n, float complex *restrict cx, int32_t incx, float complex *restrict cy, int32_t incy);
void tinyblas_zswap(int32_t n, double complex *restrict zx, int32_t incx, double complex *restrict zy, int32_t incy);

/* y <- x */
void tinyblas_scopy(int32_t n, const float *restrict sx, int32_t incx, float *restrict sy, int32_t incy);
void tinyblas_dcopy(int32_t n, const double *restrict dx, int32_t incx, double *restrict dy, int32_t incy);
void tinyblas_ccopy(int32_t n, const float complex *restrict cx, int32_t incx, float complex *restrict cy, int32_t incy);
void tinyblas_zcopy(int32_t n, const double complex *restrict zx, int32_t incx, double complex *restrict zy, int32_t incy);

/* y <- alpha * x + y */
void tinyblas_saxpy(int32_t n, float sa, const float *restrict sx, int32_t incx, float *restrict sy, int32_t incy);
void tinyblas_daxpy(int32_t n, double da, const double *restrict dx, int32_t incx, double *restrict dy, int32_t incy);
void tinyblas_caxpy(int32_t n, float complex ca, const float complex *restrict cx, int32_t incx, float complex *restrict cy, int32_t incy);
void tinyblas_zaxpy(int32_t n, double complex za, const double complex *restrict zx, int32_t incx, double complex *restrict zy, int32_t incy);

/* x <- alpha * x */
void tinyblas_sscal (int32_t n, float sa, float *restrict sx, int32_t incx);
void tinyblas_dscal (int32_t n, double da, double *restrict dx, int32_t incx);
void tinyblas_cscal (int32_t n, float complex ca, float complex *restrict cx, int32_t incx);
void tinyblas_zscal (int32_t n, double complex za, double complex *restrict zx, int32_t incx);
void tinyblas_csscal(int32_t n, float sa, float complex *restrict cx, int32_t incx);
void tinyblas_zdscal(int32_t n, double da, double complex *restrict zx, int32_t incx);

/* givens rotation: build (c, s) with c*a + s*b == r and -conj(s)*a + c*b == 0 */
void tinyblas_srotg(float a, float b, float *restrict c, float *restrict s, float *restrict r);
void tinyblas_drotg(double a, double b, double *restrict c, double *restrict s, double *restrict r);
void tinyblas_crotg(float complex a, float complex b, float *restrict c, float complex *restrict s, float complex *restrict r);
void tinyblas_zrotg(double complex a, double complex b, double *restrict c, double complex *restrict s, double complex *restrict r);

/* apply a plane rotation */
void tinyblas_srot (int32_t n, float *restrict sx, int32_t incx, float *restrict sy, int32_t incy, float c, float s);
void tinyblas_drot (int32_t n, double *restrict dx, int32_t incx, double *restrict dy, int32_t incy, double c, double s);
void tinyblas_csrot(int32_t n, float complex *restrict cx, int32_t incx, float complex *restrict cy, int32_t incy, float c, float s);
void tinyblas_zdrot(int32_t n, double complex *restrict zx, int32_t incx, double complex *restrict zy, int32_t incy, double c, double s);

#endif
