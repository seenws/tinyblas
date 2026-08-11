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

#include <stdint.h>
#include <assert.h>
#include <math.h>

#include "../headers/tinyblas_level1.h"
#include "../headers/tinyblas_common.h"

/* The unit-stride branch exists so the compiler sees a contiguous reduction and
 * vectorizes it; see the -O3 -fassociative-math flags in test.sh. */

/*
 *  Double-precision dot product: ddot
 */
double
tinyblas_ddot(int32_t n,
        const double *restrict dx, int32_t incx,
        const double *restrict dy, int32_t incy)
{
    if (n <= 0) return 0.0;

    assert(dx && dy);

    if (incx < 0) dx += (1 - n) * incx;
    if (incy < 0) dy += (1 - n) * incy;

    double dot = 0.0;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            dot += dx[i] * dy[i];

        return dot;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        dot += dx[ix] * dy[iy];

    return dot;
}

/*
 *  Single-precision dot product: sdot
 */
float
tinyblas_sdot(int32_t n,
        const float *restrict sx, int32_t incx,
        const float *restrict sy, int32_t incy)
{
    if (n <= 0) return 0.0f;

    assert(sx && sy);

    if (incx < 0) sx += (1 - n) * incx;
    if (incy < 0) sy += (1 - n) * incy;

    float dot = 0.0f;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            dot += sx[i] * sy[i];

        return dot;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        dot += sx[ix] * sy[iy];

    return dot;
}

/*
 *  Mixed-precision dot product: single-precision inputs, double-precision accumulator
 */
double
tinyblas_dsdot(int32_t n,
        const float *restrict dx, int32_t incx,
        const float *restrict dy, int32_t incy)
{
    if (n <= 0) return 0.0;

    assert(dx && dy);

    if (incx < 0) dx += (1 - n) * incx;
    if (incy < 0) dy += (1 - n) * incy;

    double dot = 0.0;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            dot += (double)dx[i] * (double)dy[i];

        return dot;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        dot += (double)dx[ix] * (double)dy[iy];

    return dot;
}

/*
 *  Complex single-precision unconjugated dot product: cdotu
 */
float complex
tinyblas_cdotu(int32_t n,
        const float complex *restrict zx, int32_t incx,
        const float complex *restrict zy, int32_t incy)
{
    if (n <= 0) return 0.0f;

    assert(zx && zy);

    if (incx < 0) zx += (1 - n) * incx;
    if (incy < 0) zy += (1 - n) * incy;

    /* ponytail: complex `*` calls __mulsc3 for inf/nan handling and does not
     * vectorize. Add -fcx-limited-range if a profile says this path matters. */
    float complex dot = 0.0f;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            dot += zx[i] * zy[i];

        return dot;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        dot += zx[ix] * zy[iy];

    return dot;
}

/*
 *  Mixed-precision dot product plus a scalar: sdsdot
 */
float
tinyblas_sdsdot(int32_t n, float sb,
        const float *restrict sx, int32_t incx,
        const float *restrict sy, int32_t incy)
{
    return sb + (float)tinyblas_dsdot(n, sx, incx, sy, incy);
}

/*
 *  Complex single-precision conjugated dot product: cdotc (conj(x) . y)
 */
float complex
tinyblas_cdotc(int32_t n,
        const float complex *restrict zx, int32_t incx,
        const float complex *restrict zy, int32_t incy)
{
    if (n <= 0) return 0.0f;

    assert(zx && zy);

    if (incx < 0) zx += (1 - n) * incx;
    if (incy < 0) zy += (1 - n) * incy;

    float complex dot = 0.0f;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            dot += conjf(zx[i]) * zy[i];

        return dot;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        dot += conjf(zx[ix]) * zy[iy];

    return dot;
}

/*
 *  Complex double-precision unconjugated dot product: zdotu
 */
double complex
tinyblas_zdotu(int32_t n,
        const double complex *restrict zx, int32_t incx,
        const double complex *restrict zy, int32_t incy)
{
    if (n <= 0) return 0.0;

    assert(zx && zy);

    if (incx < 0) zx += (1 - n) * incx;
    if (incy < 0) zy += (1 - n) * incy;

    double complex dot = 0.0;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            dot += zx[i] * zy[i];

        return dot;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        dot += zx[ix] * zy[iy];

    return dot;
}

/*
 *  Complex double-precision conjugated dot product: zdotc (conj(x) . y)
 */
double complex
tinyblas_zdotc(int32_t n,
        const double complex *restrict zx, int32_t incx,
        const double complex *restrict zy, int32_t incy)
{
    if (n <= 0) return 0.0;

    assert(zx && zy);

    if (incx < 0) zx += (1 - n) * incx;
    if (incy < 0) zy += (1 - n) * incy;

    double complex dot = 0.0;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            dot += conj(zx[i]) * zy[i];

        return dot;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        dot += conj(zx[ix]) * zy[iy];

    return dot;
}

/* Textbook scaled sum of squares: keeps ||x|| finite when the squares are not.
 * Result is scale * sqrt(ssq), starting from scale = 0, ssq = 1.
 * ponytail: the branch blocks vectorization. A max-then-sum two-pass version is
 * faster if a profile ever says nrm2 matters. */
static void
dssq(double a, double *restrict scale, double *restrict ssq)
{
    a = fabs(a);

    if (a == 0.0) return;

    if (*scale < a) {
        double r = *scale / a;
        *ssq = 1.0 + *ssq * r * r;
        *scale = a;
    } else {
        double r = a / *scale;
        *ssq += r * r;
    }
}

/*
 *  Euclidean norm, single precision: snrm2
 */
float
tinyblas_snrm2(int32_t n, const float *restrict sx, int32_t incx)
{
    /* Squares of finite floats always fit in a double, so accumulating in one
     * is both scaling-free and exact enough. Aliasing sx twice is fine: dsdot
     * only reads. */
    return (float)sqrt(tinyblas_dsdot(n, sx, incx, sx, incx));
}

/*
 *  Euclidean norm, double precision: dnrm2
 */
double
tinyblas_dnrm2(int32_t n, const double *restrict dx, int32_t incx)
{
    if (n <= 0) return 0.0;

    assert(dx);

    if (incx < 0) dx += (1 - n) * incx;

    double scale = 0.0, ssq = 1.0;

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx)
        dssq(dx[ix], &scale, &ssq);

    return scale * sqrt(ssq);
}

/*
 *  Euclidean norm of a complex single-precision vector: scnrm2
 */
float
tinyblas_scnrm2(int32_t n, const float complex *restrict cx, int32_t incx)
{
    if (n <= 0) return 0.0f;

    assert(cx);

    if (incx < 0) cx += (1 - n) * incx;

    double ssq = 0.0;

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx) {
        double re = (double)crealf(cx[ix]), im = (double)cimagf(cx[ix]);
        ssq += re * re + im * im;
    }

    return (float)sqrt(ssq);
}

/*
 *  Euclidean norm of a complex double-precision vector: dznrm2
 */
double
tinyblas_dznrm2(int32_t n, const double complex *restrict zx, int32_t incx)
{
    if (n <= 0) return 0.0;

    assert(zx);

    if (incx < 0) zx += (1 - n) * incx;

    double scale = 0.0, ssq = 1.0;

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx) {
        dssq(creal(zx[ix]), &scale, &ssq);
        dssq(cimag(zx[ix]), &scale, &ssq);
    }

    return scale * sqrt(ssq);
}

/*
 *  Sum of absolute values, single precision: sasum
 */
float
tinyblas_sasum(int32_t n, const float *restrict sx, int32_t incx)
{
    if (n <= 0) return 0.0f;

    assert(sx);

    if (incx < 0) sx += (1 - n) * incx;

    float sum = 0.0f;

    if (incx == 1) {
        for (int32_t i = 0; i < n; ++i)
            sum += fabsf(sx[i]);

        return sum;
    }

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx)
        sum += fabsf(sx[ix]);

    return sum;
}

/*
 *  Sum of absolute values, double precision: dasum
 */
double
tinyblas_dasum(int32_t n, const double *restrict dx, int32_t incx)
{
    if (n <= 0) return 0.0;

    assert(dx);

    if (incx < 0) dx += (1 - n) * incx;

    double sum = 0.0;

    if (incx == 1) {
        for (int32_t i = 0; i < n; ++i)
            sum += fabs(dx[i]);

        return sum;
    }

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx)
        sum += fabs(dx[ix]);

    return sum;
}

/*
 *  Sum of |re| + |im|, complex single precision: scasum
 */
float
tinyblas_scasum(int32_t n, const float complex *restrict cx, int32_t incx)
{
    if (n <= 0) return 0.0f;

    assert(cx);

    if (incx < 0) cx += (1 - n) * incx;

    float sum = 0.0f;

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx)
        sum += fabsf(crealf(cx[ix])) + fabsf(cimagf(cx[ix]));

    return sum;
}

/*
 *  Sum of |re| + |im|, complex double precision: dzasum
 */
double
tinyblas_dzasum(int32_t n, const double complex *restrict zx, int32_t incx)
{
    if (n <= 0) return 0.0;

    assert(zx);

    if (incx < 0) zx += (1 - n) * incx;

    double sum = 0.0;

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx)
        sum += fabs(creal(zx[ix])) + fabs(cimag(zx[ix]));

    return sum;
}

/*
 *  Index of the largest |x[i]|, single precision: isamax
 */
int32_t
tinyblas_isamax(int32_t n, const float *restrict sx, int32_t incx)
{
    if (n <= 0) return -1;

    assert(sx);

    if (incx < 0) sx += (1 - n) * incx;

    int32_t best = 0;
    float largest = fabsf(sx[0]);

    for (int32_t i = 1, ix = incx; i < n; ++i, ix += incx) {
        float a = fabsf(sx[ix]);

        if (a > largest) {
            largest = a;
            best = i;
        }
    }

    return best;
}

/*
 *  Index of the largest |x[i]|, double precision: idamax
 */
int32_t
tinyblas_idamax(int32_t n, const double *restrict dx, int32_t incx)
{
    if (n <= 0) return -1;

    assert(dx);

    if (incx < 0) dx += (1 - n) * incx;

    int32_t best = 0;
    double largest = fabs(dx[0]);

    for (int32_t i = 1, ix = incx; i < n; ++i, ix += incx) {
        double a = fabs(dx[ix]);

        if (a > largest) {
            largest = a;
            best = i;
        }
    }

    return best;
}

/*
 *  Index of the largest |re| + |im|, complex single precision: icamax
 */
int32_t
tinyblas_icamax(int32_t n, const float complex *restrict cx, int32_t incx)
{
    if (n <= 0) return -1;

    assert(cx);

    if (incx < 0) cx += (1 - n) * incx;

    int32_t best = 0;
    float largest = fabsf(crealf(cx[0])) + fabsf(cimagf(cx[0]));

    for (int32_t i = 1, ix = incx; i < n; ++i, ix += incx) {
        float a = fabsf(crealf(cx[ix])) + fabsf(cimagf(cx[ix]));

        if (a > largest) {
            largest = a;
            best = i;
        }
    }

    return best;
}

/*
 *  Index of the largest |re| + |im|, complex double precision: izamax
 */
int32_t
tinyblas_izamax(int32_t n, const double complex *restrict zx, int32_t incx)
{
    if (n <= 0) return -1;

    assert(zx);

    if (incx < 0) zx += (1 - n) * incx;

    int32_t best = 0;
    double largest = fabs(creal(zx[0])) + fabs(cimag(zx[0]));

    for (int32_t i = 1, ix = incx; i < n; ++i, ix += incx) {
        double a = fabs(creal(zx[ix])) + fabs(cimag(zx[ix]));

        if (a > largest) {
            largest = a;
            best = i;
        }
    }

    return best;
}

/*
 *  Swap two vectors: sswap
 */
void
tinyblas_sswap(int32_t n, float *restrict sx, int32_t incx, float *restrict sy, int32_t incy)
{
    if (n <= 0) return;

    assert(sx && sy);

    if (incx < 0) sx += (1 - n) * incx;
    if (incy < 0) sy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i) {
            float t = sx[i];
            sx[i] = sy[i];
            sy[i] = t;
        }

        return;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy) {
        float t = sx[ix];
        sx[ix] = sy[iy];
        sy[iy] = t;
    }
}

/*
 *  Swap two vectors: dswap
 */
void
tinyblas_dswap(int32_t n, double *restrict dx, int32_t incx, double *restrict dy, int32_t incy)
{
    if (n <= 0) return;

    assert(dx && dy);

    if (incx < 0) dx += (1 - n) * incx;
    if (incy < 0) dy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i) {
            double t = dx[i];
            dx[i] = dy[i];
            dy[i] = t;
        }

        return;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy) {
        double t = dx[ix];
        dx[ix] = dy[iy];
        dy[iy] = t;
    }
}

/*
 *  Swap two vectors: cswap
 */
void
tinyblas_cswap(int32_t n, float complex *restrict cx, int32_t incx, float complex *restrict cy, int32_t incy)
{
    if (n <= 0) return;

    assert(cx && cy);

    if (incx < 0) cx += (1 - n) * incx;
    if (incy < 0) cy += (1 - n) * incy;

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy) {
        float complex t = cx[ix];
        cx[ix] = cy[iy];
        cy[iy] = t;
    }
}

/*
 *  Swap two vectors: zswap
 */
void
tinyblas_zswap(int32_t n, double complex *restrict zx, int32_t incx, double complex *restrict zy, int32_t incy)
{
    if (n <= 0) return;

    assert(zx && zy);

    if (incx < 0) zx += (1 - n) * incx;
    if (incy < 0) zy += (1 - n) * incy;

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy) {
        double complex t = zx[ix];
        zx[ix] = zy[iy];
        zy[iy] = t;
    }
}

/*
 *  Copy x into y: scopy
 */
void
tinyblas_scopy(int32_t n, const float *restrict sx, int32_t incx, float *restrict sy, int32_t incy)
{
    if (n <= 0) return;

    assert(sx && sy);

    if (incx < 0) sx += (1 - n) * incx;
    if (incy < 0) sy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            sy[i] = sx[i];

        return;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        sy[iy] = sx[ix];
}

/*
 *  Copy x into y: dcopy
 */
void
tinyblas_dcopy(int32_t n, const double *restrict dx, int32_t incx, double *restrict dy, int32_t incy)
{
    if (n <= 0) return;

    assert(dx && dy);

    if (incx < 0) dx += (1 - n) * incx;
    if (incy < 0) dy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            dy[i] = dx[i];

        return;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        dy[iy] = dx[ix];
}

/*
 *  Copy x into y: ccopy
 */
void
tinyblas_ccopy(int32_t n, const float complex *restrict cx, int32_t incx, float complex *restrict cy, int32_t incy)
{
    if (n <= 0) return;

    assert(cx && cy);

    if (incx < 0) cx += (1 - n) * incx;
    if (incy < 0) cy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            cy[i] = cx[i];

        return;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        cy[iy] = cx[ix];
}

/*
 *  Copy x into y: zcopy
 */
void
tinyblas_zcopy(int32_t n, const double complex *restrict zx, int32_t incx, double complex *restrict zy, int32_t incy)
{
    if (n <= 0) return;

    assert(zx && zy);

    if (incx < 0) zx += (1 - n) * incx;
    if (incy < 0) zy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            zy[i] = zx[i];

        return;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        zy[iy] = zx[ix];
}

/*
 *  y <- a*x + y: saxpy
 */
void
tinyblas_saxpy(int32_t n, float sa, const float *restrict sx, int32_t incx, float *restrict sy, int32_t incy)
{
    if (n <= 0 || sa == 0.0f) return;

    assert(sx && sy);

    if (incx < 0) sx += (1 - n) * incx;
    if (incy < 0) sy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            sy[i] += sa * sx[i];

        return;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        sy[iy] += sa * sx[ix];
}

/*
 *  y <- a*x + y: daxpy
 */
void
tinyblas_daxpy(int32_t n, double da, const double *restrict dx, int32_t incx, double *restrict dy, int32_t incy)
{
    if (n <= 0 || da == 0.0) return;

    assert(dx && dy);

    if (incx < 0) dx += (1 - n) * incx;
    if (incy < 0) dy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i)
            dy[i] += da * dx[i];

        return;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        dy[iy] += da * dx[ix];
}

/*
 *  y <- a*x + y: caxpy
 */
void
tinyblas_caxpy(int32_t n, float complex ca, const float complex *restrict cx, int32_t incx, float complex *restrict cy, int32_t incy)
{
    if (n <= 0 || ca == 0.0f) return;

    assert(cx && cy);

    if (incx < 0) cx += (1 - n) * incx;
    if (incy < 0) cy += (1 - n) * incy;

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        cy[iy] += ca * cx[ix];
}

/*
 *  y <- a*x + y: zaxpy
 */
void
tinyblas_zaxpy(int32_t n, double complex za, const double complex *restrict zx, int32_t incx, double complex *restrict zy, int32_t incy)
{
    if (n <= 0 || za == 0.0) return;

    assert(zx && zy);

    if (incx < 0) zx += (1 - n) * incx;
    if (incy < 0) zy += (1 - n) * incy;

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy)
        zy[iy] += za * zx[ix];
}

/*
 *  x <- a*x: sscal
 */
void
tinyblas_sscal(int32_t n, float sa, float *restrict sx, int32_t incx)
{
    if (n <= 0) return;

    assert(sx);

    if (incx < 0) sx += (1 - n) * incx;

    if (incx == 1) {
        for (int32_t i = 0; i < n; ++i)
            sx[i] *= sa;

        return;
    }

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx)
        sx[ix] *= sa;
}

/*
 *  x <- a*x: dscal
 */
void
tinyblas_dscal(int32_t n, double da, double *restrict dx, int32_t incx)
{
    if (n <= 0) return;

    assert(dx);

    if (incx < 0) dx += (1 - n) * incx;

    if (incx == 1) {
        for (int32_t i = 0; i < n; ++i)
            dx[i] *= da;

        return;
    }

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx)
        dx[ix] *= da;
}

/*
 *  x <- a*x with complex a: cscal
 */
void
tinyblas_cscal(int32_t n, float complex ca, float complex *restrict cx, int32_t incx)
{
    if (n <= 0) return;

    assert(cx);

    if (incx < 0) cx += (1 - n) * incx;

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx)
        cx[ix] *= ca;
}

/*
 *  x <- a*x with complex a: zscal
 */
void
tinyblas_zscal(int32_t n, double complex za, double complex *restrict zx, int32_t incx)
{
    if (n <= 0) return;

    assert(zx);

    if (incx < 0) zx += (1 - n) * incx;

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx)
        zx[ix] *= za;
}

/*
 *  x <- a*x with real a on a complex vector: csscal
 */
void
tinyblas_csscal(int32_t n, float sa, float complex *restrict cx, int32_t incx)
{
    if (n <= 0) return;

    assert(cx);

    if (incx < 0) cx += (1 - n) * incx;

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx)
        cx[ix] *= sa;
}

/*
 *  x <- a*x with real a on a complex vector: zdscal
 */
void
tinyblas_zdscal(int32_t n, double da, double complex *restrict zx, int32_t incx)
{
    if (n <= 0) return;

    assert(zx);

    if (incx < 0) zx += (1 - n) * incx;

    for (int32_t i = 0, ix = 0; i < n; ++i, ix += incx)
        zx[ix] *= da;
}

/*
 *  Construct a Givens rotation: srotg
 */
void
tinyblas_srotg(float a, float b, float *restrict c, float *restrict s, float *restrict r)
{
    assert(c && s && r);

    /* hypotf does the overflow-safe scaling the reference implementation
     * hand-rolls. */
    float h = hypotf(a, b);

    if (h == 0.0f) {
        *c = 1.0f;
        *s = 0.0f;
        *r = 0.0f;

        return;
    }

    /* r takes the sign of the larger input, so rotating (a, 0) is the identity. */
    if ((fabsf(a) > fabsf(b) ? a : b) < 0.0f) h = -h;

    *c = a / h;
    *s = b / h;
    *r = h;
}

/*
 *  Construct a Givens rotation: drotg
 */
void
tinyblas_drotg(double a, double b, double *restrict c, double *restrict s, double *restrict r)
{
    assert(c && s && r);

    double h = hypot(a, b);

    if (h == 0.0) {
        *c = 1.0;
        *s = 0.0;
        *r = 0.0;

        return;
    }

    if ((fabs(a) > fabs(b) ? a : b) < 0.0) h = -h;

    *c = a / h;
    *s = b / h;
    *r = h;
}

/*
 *  Construct a complex Givens rotation: crotg (c is real, s is complex)
 */
void
tinyblas_crotg(float complex a, float complex b, float *restrict c, float complex *restrict s, float complex *restrict r)
{
    assert(c && s && r);

    float na = cabsf(a), nb = cabsf(b);

    if (nb == 0.0f) {
        *c = 1.0f;
        *s = 0.0f;
        *r = a;

        return;
    }

    if (na == 0.0f) {
        *c = 0.0f;
        *s = conjf(b) / nb;
        *r = nb;

        return;
    }

    float h = hypotf(na, nb);
    float complex alpha = a / na;      /* unit phase of a */

    *c = na / h;
    *s = alpha * conjf(b) / h;
    *r = alpha * h;
}

/*
 *  Construct a complex Givens rotation: zrotg (c is real, s is complex)
 */
void
tinyblas_zrotg(double complex a, double complex b, double *restrict c, double complex *restrict s, double complex *restrict r)
{
    assert(c && s && r);

    double na = cabs(a), nb = cabs(b);

    if (nb == 0.0) {
        *c = 1.0;
        *s = 0.0;
        *r = a;

        return;
    }

    if (na == 0.0) {
        *c = 0.0;
        *s = conj(b) / nb;
        *r = nb;

        return;
    }

    double h = hypot(na, nb);
    double complex alpha = a / na;

    *c = na / h;
    *s = alpha * conj(b) / h;
    *r = alpha * h;
}

/*
 *  Apply a plane rotation: srot
 */
void
tinyblas_srot(int32_t n, float *restrict sx, int32_t incx, float *restrict sy, int32_t incy, float c, float s)
{
    if (n <= 0) return;

    assert(sx && sy);

    if (incx < 0) sx += (1 - n) * incx;
    if (incy < 0) sy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i) {
            float t = c * sx[i] + s * sy[i];
            sy[i] = c * sy[i] - s * sx[i];
            sx[i] = t;
        }

        return;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy) {
        float t = c * sx[ix] + s * sy[iy];
        sy[iy] = c * sy[iy] - s * sx[ix];
        sx[ix] = t;
    }
}

/*
 *  Apply a plane rotation: drot
 */
void
tinyblas_drot(int32_t n, double *restrict dx, int32_t incx, double *restrict dy, int32_t incy, double c, double s)
{
    if (n <= 0) return;

    assert(dx && dy);

    if (incx < 0) dx += (1 - n) * incx;
    if (incy < 0) dy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        for (int32_t i = 0; i < n; ++i) {
            double t = c * dx[i] + s * dy[i];
            dy[i] = c * dy[i] - s * dx[i];
            dx[i] = t;
        }

        return;
    }

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy) {
        double t = c * dx[ix] + s * dy[iy];
        dy[iy] = c * dy[iy] - s * dx[ix];
        dx[ix] = t;
    }
}

/*
 *  Apply a real plane rotation to complex vectors: csrot
 */
void
tinyblas_csrot(int32_t n, float complex *restrict cx, int32_t incx, float complex *restrict cy, int32_t incy, float c, float s)
{
    if (n <= 0) return;

    assert(cx && cy);

    if (incx < 0) cx += (1 - n) * incx;
    if (incy < 0) cy += (1 - n) * incy;

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy) {
        float complex t = c * cx[ix] + s * cy[iy];
        cy[iy] = c * cy[iy] - s * cx[ix];
        cx[ix] = t;
    }
}

/*
 *  Apply a real plane rotation to complex vectors: zdrot
 */
void
tinyblas_zdrot(int32_t n, double complex *restrict zx, int32_t incx, double complex *restrict zy, int32_t incy, double c, double s)
{
    if (n <= 0) return;

    assert(zx && zy);

    if (incx < 0) zx += (1 - n) * incx;
    if (incy < 0) zy += (1 - n) * incy;

    for (int32_t i = 0, ix = 0, iy = 0; i < n; ++i, ix += incx, iy += incy) {
        double complex t = c * zx[ix] + s * zy[iy];
        zy[iy] = c * zy[iy] - s * zx[ix];
        zx[ix] = t;
    }
}
