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
#include <stddef.h>
#include <assert.h>
#include <complex.h>

#include "tinyblas_level2.h"
#include "tinyblas_common.h"

/* Level 2 moves O(n^2) data to do O(n^2) flops, so all of this is bandwidth
 * bound and none of it wants intrinsics. What it does want is contiguous
 * access, which on a row-major matrix means:
 *
 *   - the untransposed gemv walks rows and reduces, four rows at a time so one
 *     x load feeds four accumulators and the four chains hide FMA latency
 *   - the transposed gemv must NOT walk columns. It runs an axpy per row into
 *     y instead, so A streams contiguously and y takes the repeated traffic.
 *
 * The triangular routines reach A through a stride pair (rs, cs) rather than a
 * transpose flag, exactly as the gemm packing does. Transposing swaps the pair
 * and flips which triangle is referenced, so one loop covers all four
 * combinations of uplo and trans instead of four near-copies.
 *
 * ponytail: the transposed gemv keeps the whole of y live across one pass of
 * A, so it degrades once y stops fitting in L2, around n = 30000 for double.
 * Blocking j would fix that at the cost of re-reading A per block, which is
 * the wrong trade below that size. Revisit only if very wide matrices matter.
 */

/*
 *  Single-precision general matrix-vector product: sgemv
 */
void
tinyblas_sgemv(enum tinyblas_op trans, int32_t m, int32_t n, float alpha,
        const float *restrict a, int32_t lda,
        const float *restrict x, int32_t incx,
        float beta, float *restrict y, int32_t incy)
{
    int32_t lenx = (trans == TINYBLAS_NONE) ? n : m;
    int32_t leny = (trans == TINYBLAS_NONE) ? m : n;

    if (m <= 0 || n <= 0) return;

    assert(y);

    if (incx < 0) x += (ptrdiff_t)(1 - lenx) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - leny) * incy;

    if (beta == 0.0f) {
        for (int32_t i = 0; i < leny; ++i) y[(ptrdiff_t)i * incy] = 0.0f;
    } else if (beta != 1.0f) {
        for (int32_t i = 0; i < leny; ++i) y[(ptrdiff_t)i * incy] *= beta;
    }

    if (alpha == 0.0f) return;

    assert(a && x);

    if (trans != TINYBLAS_NONE) {
        for (int32_t i = 0; i < m; ++i) {
            float t = alpha * x[(ptrdiff_t)i * incx];
            const float *row = a + (ptrdiff_t)i * lda;

            for (int32_t j = 0; j < n; ++j)
                y[(ptrdiff_t)j * incy] += t * row[j];
        }

        return;
    }

    if (incx == 1 && incy == 1) {
        int32_t i = 0;

        for (; i + 4 <= m; i += 4) {
            const float *r0 = a + (ptrdiff_t)i * lda;
            const float *r1 = r0 + lda, *r2 = r1 + lda, *r3 = r2 + lda;
            float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;

            for (int32_t j = 0; j < n; ++j) {
                float xv = x[j];

                t0 += r0[j] * xv;
                t1 += r1[j] * xv;
                t2 += r2[j] * xv;
                t3 += r3[j] * xv;
            }

            y[i]     += alpha * t0;
            y[i + 1] += alpha * t1;
            y[i + 2] += alpha * t2;
            y[i + 3] += alpha * t3;
        }

        for (; i < m; ++i) {
            const float *row = a + (ptrdiff_t)i * lda;
            float t = 0.0f;

            for (int32_t j = 0; j < n; ++j)
                t += row[j] * x[j];

            y[i] += alpha * t;
        }

        return;
    }

    for (int32_t i = 0; i < m; ++i) {
        const float *row = a + (ptrdiff_t)i * lda;
        float t = 0.0f;

        for (int32_t j = 0; j < n; ++j)
            t += row[j] * x[(ptrdiff_t)j * incx];

        y[(ptrdiff_t)i * incy] += alpha * t;
    }
}

/*
 *  Double-precision general matrix-vector product: dgemv
 */
void
tinyblas_dgemv(enum tinyblas_op trans, int32_t m, int32_t n, double alpha,
        const double *restrict a, int32_t lda,
        const double *restrict x, int32_t incx,
        double beta, double *restrict y, int32_t incy)
{
    int32_t lenx = (trans == TINYBLAS_NONE) ? n : m;
    int32_t leny = (trans == TINYBLAS_NONE) ? m : n;

    if (m <= 0 || n <= 0) return;

    assert(y);

    if (incx < 0) x += (ptrdiff_t)(1 - lenx) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - leny) * incy;

    if (beta == 0.0) {
        for (int32_t i = 0; i < leny; ++i) y[(ptrdiff_t)i * incy] = 0.0;
    } else if (beta != 1.0) {
        for (int32_t i = 0; i < leny; ++i) y[(ptrdiff_t)i * incy] *= beta;
    }

    if (alpha == 0.0) return;

    assert(a && x);

    if (trans != TINYBLAS_NONE) {
        for (int32_t i = 0; i < m; ++i) {
            double t = alpha * x[(ptrdiff_t)i * incx];
            const double *row = a + (ptrdiff_t)i * lda;

            for (int32_t j = 0; j < n; ++j)
                y[(ptrdiff_t)j * incy] += t * row[j];
        }

        return;
    }

    if (incx == 1 && incy == 1) {
        int32_t i = 0;

        for (; i + 4 <= m; i += 4) {
            const double *r0 = a + (ptrdiff_t)i * lda;
            const double *r1 = r0 + lda, *r2 = r1 + lda, *r3 = r2 + lda;
            double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;

            for (int32_t j = 0; j < n; ++j) {
                double xv = x[j];

                t0 += r0[j] * xv;
                t1 += r1[j] * xv;
                t2 += r2[j] * xv;
                t3 += r3[j] * xv;
            }

            y[i]     += alpha * t0;
            y[i + 1] += alpha * t1;
            y[i + 2] += alpha * t2;
            y[i + 3] += alpha * t3;
        }

        for (; i < m; ++i) {
            const double *row = a + (ptrdiff_t)i * lda;
            double t = 0.0;

            for (int32_t j = 0; j < n; ++j)
                t += row[j] * x[j];

            y[i] += alpha * t;
        }

        return;
    }

    for (int32_t i = 0; i < m; ++i) {
        const double *row = a + (ptrdiff_t)i * lda;
        double t = 0.0;

        for (int32_t j = 0; j < n; ++j)
            t += row[j] * x[(ptrdiff_t)j * incx];

        y[(ptrdiff_t)i * incy] += alpha * t;
    }
}

/*
 *  Single-precision complex general matrix-vector product: cgemv
 */
void
tinyblas_cgemv(enum tinyblas_op trans, int32_t m, int32_t n,
        float complex alpha,
        const float complex *restrict a, int32_t lda,
        const float complex *restrict x, int32_t incx,
        float complex beta, float complex *restrict y, int32_t incy)
{
    int32_t lenx = (trans == TINYBLAS_NONE) ? n : m;
    int32_t leny = (trans == TINYBLAS_NONE) ? m : n;
    int     cnj = (trans == TINYBLAS_CONJ_TRANS);
    float   alr, ali;

    if (m <= 0 || n <= 0) return;

    assert(y);

    if (incx < 0) x += (ptrdiff_t)(1 - lenx) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - leny) * incy;

    if (beta == 0.0f) {
        for (int32_t i = 0; i < leny; ++i) y[(ptrdiff_t)i * incy] = 0.0f;
    } else if (beta != 1.0f) {
        float br = crealf(beta), bi = cimagf(beta);

        for (int32_t i = 0; i < leny; ++i) {
            float complex z = y[(ptrdiff_t)i * incy];
            float zr = crealf(z), zi = cimagf(z);

            y[(ptrdiff_t)i * incy] = (br * zr - bi * zi)
                                   + (br * zi + bi * zr) * I;
        }
    }

    if (alpha == 0.0f) return;

    assert(a && x);

    alr = crealf(alpha);
    ali = cimagf(alpha);

    if (trans != TINYBLAS_NONE) {
        /* A sign, not a branch. The flag is loop invariant, but a ternary in
         * the inner loop stops the vectorizer cold. */
        float sgn = cnj ? -1.0f : 1.0f;

        if (incx == 1 && incy == 1) {
            for (int32_t i = 0; i < m; ++i) {
                float complex xz = x[i];
                float tr = alr * crealf(xz) - ali * cimagf(xz);
                float ti = alr * cimagf(xz) + ali * crealf(xz);
                const float complex *row = a + (ptrdiff_t)i * lda;

                for (int32_t j = 0; j < n; ++j) {
                    float ar = crealf(row[j]), ai = sgn * cimagf(row[j]);
                    float complex z = y[j];

                    y[j] = (crealf(z) + (tr * ar - ti * ai))
                         + (cimagf(z) + (tr * ai + ti * ar)) * I;
                }
            }

            return;
        }

        for (int32_t i = 0; i < m; ++i) {
            float complex xz = x[(ptrdiff_t)i * incx];
            float tr = alr * crealf(xz) - ali * cimagf(xz);
            float ti = alr * cimagf(xz) + ali * crealf(xz);
            const float complex *row = a + (ptrdiff_t)i * lda;

            for (int32_t j = 0; j < n; ++j) {
                float ar = crealf(row[j]), ai = sgn * cimagf(row[j]);
                float complex z = y[(ptrdiff_t)j * incy];

                y[(ptrdiff_t)j * incy] = (crealf(z) + (tr * ar - ti * ai))
                                       + (cimagf(z) + (tr * ai + ti * ar)) * I;
            }
        }

        return;
    }

    if (incx == 1 && incy == 1) {
        int32_t i = 0;

        /* Four rows at a time. One complex dot has only two accumulator
         * chains, which is not enough in flight to cover FMA latency, and
         * that is what left the complex gemv latency bound rather than
         * bandwidth bound. */
        for (; i + 4 <= m; i += 4) {
            const float complex *r[4];
            float sr[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            float si[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

            for (int u = 0; u < 4; ++u)
                r[u] = a + (ptrdiff_t)(i + u) * lda;

            for (int32_t j = 0; j < n; ++j) {
                float xr = crealf(x[j]), xi = cimagf(x[j]);

                for (int u = 0; u < 4; ++u) {
                    float ar = crealf(r[u][j]), ai = cimagf(r[u][j]);

                    sr[u] += ar * xr - ai * xi;
                    si[u] += ar * xi + ai * xr;
                }
            }

            for (int u = 0; u < 4; ++u) {
                float complex z = y[i + u];

                y[i + u] = (crealf(z) + (alr * sr[u] - ali * si[u]))
                         + (cimagf(z) + (alr * si[u] + ali * sr[u])) * I;
            }
        }

        for (; i < m; ++i) {
            const float complex *row = a + (ptrdiff_t)i * lda;
            float sr = 0.0f, si = 0.0f;
            float complex z = y[i];

            for (int32_t j = 0; j < n; ++j) {
                float ar = crealf(row[j]), ai = cimagf(row[j]);
                float xr = crealf(x[j]), xi = cimagf(x[j]);

                sr += ar * xr - ai * xi;
                si += ar * xi + ai * xr;
            }

            y[i] = (crealf(z) + (alr * sr - ali * si))
                 + (cimagf(z) + (alr * si + ali * sr)) * I;
        }

        return;
    }

    for (int32_t i = 0; i < m; ++i) {
        const float complex *row = a + (ptrdiff_t)i * lda;
        float sr = 0.0f, si = 0.0f;
        float complex z;

        for (int32_t j = 0; j < n; ++j) {
            float complex xz = x[(ptrdiff_t)j * incx];
            float ar = crealf(row[j]), ai = cimagf(row[j]);
            float xr = crealf(xz), xi = cimagf(xz);

            sr += ar * xr - ai * xi;
            si += ar * xi + ai * xr;
        }

        z = y[(ptrdiff_t)i * incy];

        y[(ptrdiff_t)i * incy] = (crealf(z) + (alr * sr - ali * si))
                               + (cimagf(z) + (alr * si + ali * sr)) * I;
    }
}

/*
 *  Double-precision complex general matrix-vector product: zgemv
 */
void
tinyblas_zgemv(enum tinyblas_op trans, int32_t m, int32_t n,
        double complex alpha,
        const double complex *restrict a, int32_t lda,
        const double complex *restrict x, int32_t incx,
        double complex beta, double complex *restrict y, int32_t incy)
{
    int32_t lenx = (trans == TINYBLAS_NONE) ? n : m;
    int32_t leny = (trans == TINYBLAS_NONE) ? m : n;
    int     cnj = (trans == TINYBLAS_CONJ_TRANS);
    double  alr, ali;

    if (m <= 0 || n <= 0) return;

    assert(y);

    if (incx < 0) x += (ptrdiff_t)(1 - lenx) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - leny) * incy;

    if (beta == 0.0) {
        for (int32_t i = 0; i < leny; ++i) y[(ptrdiff_t)i * incy] = 0.0;
    } else if (beta != 1.0) {
        double br = creal(beta), bi = cimag(beta);

        for (int32_t i = 0; i < leny; ++i) {
            double complex z = y[(ptrdiff_t)i * incy];
            double zr = creal(z), zi = cimag(z);

            y[(ptrdiff_t)i * incy] = (br * zr - bi * zi)
                                   + (br * zi + bi * zr) * I;
        }
    }

    if (alpha == 0.0) return;

    assert(a && x);

    alr = creal(alpha);
    ali = cimag(alpha);

    if (trans != TINYBLAS_NONE) {
        /* A sign, not a branch. The flag is loop invariant, but a ternary in
         * the inner loop stops the vectorizer cold. */
        double sgn = cnj ? -1.0 : 1.0;

        if (incx == 1 && incy == 1) {
            for (int32_t i = 0; i < m; ++i) {
                double complex xz = x[i];
                double tr = alr * creal(xz) - ali * cimag(xz);
                double ti = alr * cimag(xz) + ali * creal(xz);
                const double complex *row = a + (ptrdiff_t)i * lda;

                for (int32_t j = 0; j < n; ++j) {
                    double ar = creal(row[j]), ai = sgn * cimag(row[j]);
                    double complex z = y[j];

                    y[j] = (creal(z) + (tr * ar - ti * ai))
                         + (cimag(z) + (tr * ai + ti * ar)) * I;
                }
            }

            return;
        }

        for (int32_t i = 0; i < m; ++i) {
            double complex xz = x[(ptrdiff_t)i * incx];
            double tr = alr * creal(xz) - ali * cimag(xz);
            double ti = alr * cimag(xz) + ali * creal(xz);
            const double complex *row = a + (ptrdiff_t)i * lda;

            for (int32_t j = 0; j < n; ++j) {
                double ar = creal(row[j]), ai = sgn * cimag(row[j]);
                double complex z = y[(ptrdiff_t)j * incy];

                y[(ptrdiff_t)j * incy] = (creal(z) + (tr * ar - ti * ai))
                                       + (cimag(z) + (tr * ai + ti * ar)) * I;
            }
        }

        return;
    }

    if (incx == 1 && incy == 1) {
        int32_t i = 0;

        /* Four rows at a time. One complex dot has only two accumulator
         * chains, which is not enough in flight to cover FMA latency, and
         * that is what left the complex gemv latency bound rather than
         * bandwidth bound. */
        for (; i + 4 <= m; i += 4) {
            const double complex *r[4];
            double sr[4] = { 0.0, 0.0, 0.0, 0.0 };
            double si[4] = { 0.0, 0.0, 0.0, 0.0 };

            for (int u = 0; u < 4; ++u)
                r[u] = a + (ptrdiff_t)(i + u) * lda;

            for (int32_t j = 0; j < n; ++j) {
                double xr = creal(x[j]), xi = cimag(x[j]);

                for (int u = 0; u < 4; ++u) {
                    double ar = creal(r[u][j]), ai = cimag(r[u][j]);

                    sr[u] += ar * xr - ai * xi;
                    si[u] += ar * xi + ai * xr;
                }
            }

            for (int u = 0; u < 4; ++u) {
                double complex z = y[i + u];

                y[i + u] = (creal(z) + (alr * sr[u] - ali * si[u]))
                         + (cimag(z) + (alr * si[u] + ali * sr[u])) * I;
            }
        }

        for (; i < m; ++i) {
            const double complex *row = a + (ptrdiff_t)i * lda;
            double sr = 0.0, si = 0.0;
            double complex z = y[i];

            for (int32_t j = 0; j < n; ++j) {
                double ar = creal(row[j]), ai = cimag(row[j]);
                double xr = creal(x[j]), xi = cimag(x[j]);

                sr += ar * xr - ai * xi;
                si += ar * xi + ai * xr;
            }

            y[i] = (creal(z) + (alr * sr - ali * si))
                 + (cimag(z) + (alr * si + ali * sr)) * I;
        }

        return;
    }

    for (int32_t i = 0; i < m; ++i) {
        const double complex *row = a + (ptrdiff_t)i * lda;
        double sr = 0.0, si = 0.0;
        double complex z;

        for (int32_t j = 0; j < n; ++j) {
            double complex xz = x[(ptrdiff_t)j * incx];
            double ar = creal(row[j]), ai = cimag(row[j]);
            double xr = creal(xz), xi = cimag(xz);

            sr += ar * xr - ai * xi;
            si += ar * xi + ai * xr;
        }

        z = y[(ptrdiff_t)i * incy];

        y[(ptrdiff_t)i * incy] = (creal(z) + (alr * sr - ali * si))
                               + (cimag(z) + (alr * si + ali * sr)) * I;
    }
}

/*
 *  Symmetric matrix-vector product: ssymv, dsymv
 *
 *  Each stored element is used twice, once as A[i][j] and once as A[j][i], so
 *  one pass over the stored triangle does both: it scatters t1 * a into y[j]
 *  and gathers a * x[j] into an accumulator for y[i].
 */
void
tinyblas_ssymv(enum tinyblas_uplo uplo, int32_t n, float alpha,
        const float *restrict a, int32_t lda,
        const float *restrict x, int32_t incx,
        float beta, float *restrict y, int32_t incy)
{
    if (n <= 0) return;

    assert(y);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    if (beta == 0.0f) {
        for (int32_t i = 0; i < n; ++i) y[(ptrdiff_t)i * incy] = 0.0f;
    } else if (beta != 1.0f) {
        for (int32_t i = 0; i < n; ++i) y[(ptrdiff_t)i * incy] *= beta;
    }

    if (alpha == 0.0f) return;

    assert(a && x);

    for (int32_t i = 0; i < n; ++i) {
        const float *row = a + (ptrdiff_t)i * lda;
        float t1 = alpha * x[(ptrdiff_t)i * incx];
        float t2 = 0.0f;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i + 1 : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i;

        for (int32_t j = lo; j < hi; ++j) {
            y[(ptrdiff_t)j * incy] += t1 * row[j];
            t2 += row[j] * x[(ptrdiff_t)j * incx];
        }

        y[(ptrdiff_t)i * incy] += t1 * row[i] + alpha * t2;
    }
}

void
tinyblas_dsymv(enum tinyblas_uplo uplo, int32_t n, double alpha,
        const double *restrict a, int32_t lda,
        const double *restrict x, int32_t incx,
        double beta, double *restrict y, int32_t incy)
{
    if (n <= 0) return;

    assert(y);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    if (beta == 0.0) {
        for (int32_t i = 0; i < n; ++i) y[(ptrdiff_t)i * incy] = 0.0;
    } else if (beta != 1.0) {
        for (int32_t i = 0; i < n; ++i) y[(ptrdiff_t)i * incy] *= beta;
    }

    if (alpha == 0.0) return;

    assert(a && x);

    for (int32_t i = 0; i < n; ++i) {
        const double *row = a + (ptrdiff_t)i * lda;
        double t1 = alpha * x[(ptrdiff_t)i * incx];
        double t2 = 0.0;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i + 1 : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i;

        for (int32_t j = lo; j < hi; ++j) {
            y[(ptrdiff_t)j * incy] += t1 * row[j];
            t2 += row[j] * x[(ptrdiff_t)j * incx];
        }

        y[(ptrdiff_t)i * incy] += t1 * row[i] + alpha * t2;
    }
}

/*
 *  Hermitian matrix-vector product: chemv, zhemv
 *
 *  As symv, except the mirrored use of a stored element is its conjugate and
 *  the diagonal is taken to be real whatever the stored imaginary part says.
 */
void
tinyblas_chemv(enum tinyblas_uplo uplo, int32_t n, float complex alpha,
        const float complex *restrict a, int32_t lda,
        const float complex *restrict x, int32_t incx,
        float complex beta, float complex *restrict y, int32_t incy)
{
    float alr, ali;

    if (n <= 0) return;

    assert(y);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    if (beta == 0.0f) {
        for (int32_t i = 0; i < n; ++i) y[(ptrdiff_t)i * incy] = 0.0f;
    } else if (beta != 1.0f) {
        float br = crealf(beta), bi = cimagf(beta);

        for (int32_t i = 0; i < n; ++i) {
            float complex z = y[(ptrdiff_t)i * incy];

            y[(ptrdiff_t)i * incy] = (br * crealf(z) - bi * cimagf(z))
                                   + (br * cimagf(z) + bi * crealf(z)) * I;
        }
    }

    if (alpha == 0.0f) return;

    assert(a && x);

    alr = crealf(alpha);
    ali = cimagf(alpha);

    for (int32_t i = 0; i < n; ++i) {
        const float complex *row = a + (ptrdiff_t)i * lda;
        float complex xi = x[(ptrdiff_t)i * incx];
        float t1r = alr * crealf(xi) - ali * cimagf(xi);
        float t1i = alr * cimagf(xi) + ali * crealf(xi);
        float sr = 0.0f, si = 0.0f;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i + 1 : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i;
        float complex z;

        for (int32_t j = lo; j < hi; ++j) {
            float ar = crealf(row[j]), ai = cimagf(row[j]);
            float complex xz = x[(ptrdiff_t)j * incx];
            float xr = crealf(xz), xj = cimagf(xz);
            float complex yz = y[(ptrdiff_t)j * incy];

            /* y[j] += t1 * conj(A[i][j]), since A[j][i] is conj(A[i][j]) */
            y[(ptrdiff_t)j * incy] = (crealf(yz) + (t1r * ar + t1i * ai))
                                   + (cimagf(yz) + (t1i * ar - t1r * ai)) * I;

            sr += ar * xr - ai * xj;
            si += ar * xj + ai * xr;
        }

        z = y[(ptrdiff_t)i * incy];

        /* the diagonal is real by definition */
        y[(ptrdiff_t)i * incy] =
                (crealf(z) + t1r * crealf(row[i]) + (alr * sr - ali * si))
              + (cimagf(z) + t1i * crealf(row[i]) + (alr * si + ali * sr)) * I;
    }
}

void
tinyblas_zhemv(enum tinyblas_uplo uplo, int32_t n, double complex alpha,
        const double complex *restrict a, int32_t lda,
        const double complex *restrict x, int32_t incx,
        double complex beta, double complex *restrict y, int32_t incy)
{
    double alr, ali;

    if (n <= 0) return;

    assert(y);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    if (beta == 0.0) {
        for (int32_t i = 0; i < n; ++i) y[(ptrdiff_t)i * incy] = 0.0;
    } else if (beta != 1.0) {
        double br = creal(beta), bi = cimag(beta);

        for (int32_t i = 0; i < n; ++i) {
            double complex z = y[(ptrdiff_t)i * incy];

            y[(ptrdiff_t)i * incy] = (br * creal(z) - bi * cimag(z))
                                   + (br * cimag(z) + bi * creal(z)) * I;
        }
    }

    if (alpha == 0.0) return;

    assert(a && x);

    alr = creal(alpha);
    ali = cimag(alpha);

    for (int32_t i = 0; i < n; ++i) {
        const double complex *row = a + (ptrdiff_t)i * lda;
        double complex xi = x[(ptrdiff_t)i * incx];
        double t1r = alr * creal(xi) - ali * cimag(xi);
        double t1i = alr * cimag(xi) + ali * creal(xi);
        double sr = 0.0, si = 0.0;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i + 1 : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i;
        double complex z;

        for (int32_t j = lo; j < hi; ++j) {
            double ar = creal(row[j]), ai = cimag(row[j]);
            double complex xz = x[(ptrdiff_t)j * incx];
            double xr = creal(xz), xj = cimag(xz);
            double complex yz = y[(ptrdiff_t)j * incy];

            y[(ptrdiff_t)j * incy] = (creal(yz) + (t1r * ar + t1i * ai))
                                   + (cimag(yz) + (t1i * ar - t1r * ai)) * I;

            sr += ar * xr - ai * xj;
            si += ar * xj + ai * xr;
        }

        z = y[(ptrdiff_t)i * incy];

        y[(ptrdiff_t)i * incy] =
                (creal(z) + t1r * creal(row[i]) + (alr * sr - ali * si))
              + (cimag(z) + t1i * creal(row[i]) + (alr * si + ali * sr)) * I;
    }
}

/*
 *  Triangular matrix-vector product: strmv, dtrmv, ctrmv, ztrmv
 *
 *  The stride pair turns op(A) into a plain indexing choice, and transposing
 *  flips which triangle is referenced: the transpose of an upper triangle is a
 *  lower one. After that there are only two loops, distinguished by direction.
 */
void
tinyblas_strmv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const float *restrict a, int32_t lda, float *restrict x, int32_t incx)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t rs = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t cs = notrans ? 1 : (ptrdiff_t)lda;
    int upper = notrans ? (uplo == TINYBLAS_UPPER) : (uplo != TINYBLAS_UPPER);
    int unit = (diag == TINYBLAS_UNIT);

    if (n <= 0) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    if (upper) {
        for (int32_t i = 0; i < n; ++i) {
            float t = unit ? x[(ptrdiff_t)i * incx]
                           : a[(ptrdiff_t)i * rs + (ptrdiff_t)i * cs]
                             * x[(ptrdiff_t)i * incx];

            for (int32_t j = i + 1; j < n; ++j)
                t += a[(ptrdiff_t)i * rs + (ptrdiff_t)j * cs]
                     * x[(ptrdiff_t)j * incx];

            x[(ptrdiff_t)i * incx] = t;
        }

        return;
    }

    for (int32_t i = n - 1; i >= 0; --i) {
        float t = unit ? x[(ptrdiff_t)i * incx]
                       : a[(ptrdiff_t)i * rs + (ptrdiff_t)i * cs]
                         * x[(ptrdiff_t)i * incx];

        for (int32_t j = 0; j < i; ++j)
            t += a[(ptrdiff_t)i * rs + (ptrdiff_t)j * cs]
                 * x[(ptrdiff_t)j * incx];

        x[(ptrdiff_t)i * incx] = t;
    }
}

void
tinyblas_dtrmv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const double *restrict a, int32_t lda, double *restrict x, int32_t incx)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t rs = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t cs = notrans ? 1 : (ptrdiff_t)lda;
    int upper = notrans ? (uplo == TINYBLAS_UPPER) : (uplo != TINYBLAS_UPPER);
    int unit = (diag == TINYBLAS_UNIT);

    if (n <= 0) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    if (upper) {
        for (int32_t i = 0; i < n; ++i) {
            double t = unit ? x[(ptrdiff_t)i * incx]
                            : a[(ptrdiff_t)i * rs + (ptrdiff_t)i * cs]
                              * x[(ptrdiff_t)i * incx];

            for (int32_t j = i + 1; j < n; ++j)
                t += a[(ptrdiff_t)i * rs + (ptrdiff_t)j * cs]
                     * x[(ptrdiff_t)j * incx];

            x[(ptrdiff_t)i * incx] = t;
        }

        return;
    }

    for (int32_t i = n - 1; i >= 0; --i) {
        double t = unit ? x[(ptrdiff_t)i * incx]
                        : a[(ptrdiff_t)i * rs + (ptrdiff_t)i * cs]
                          * x[(ptrdiff_t)i * incx];

        for (int32_t j = 0; j < i; ++j)
            t += a[(ptrdiff_t)i * rs + (ptrdiff_t)j * cs]
                 * x[(ptrdiff_t)j * incx];

        x[(ptrdiff_t)i * incx] = t;
    }
}

void
tinyblas_ctrmv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const float complex *restrict a, int32_t lda,
        float complex *restrict x, int32_t incx)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t rs = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t cs = notrans ? 1 : (ptrdiff_t)lda;
    int upper = notrans ? (uplo == TINYBLAS_UPPER) : (uplo != TINYBLAS_UPPER);
    int unit = (diag == TINYBLAS_UNIT);
    int cnj = (trans == TINYBLAS_CONJ_TRANS);
    int32_t i, step, last;

    if (n <= 0) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    i    = upper ? 0 : n - 1;
    step = upper ? 1 : -1;
    last = upper ? n : -1;

    for (; i != last; i += step) {
        float complex xi = x[(ptrdiff_t)i * incx];
        float tr, ti;
        int32_t lo = upper ? i + 1 : 0;
        int32_t hi = upper ? n : i;

        if (unit) {
            tr = crealf(xi);
            ti = cimagf(xi);
        } else {
            float complex d = a[(ptrdiff_t)i * rs + (ptrdiff_t)i * cs];
            float dr = crealf(d), di = cnj ? -cimagf(d) : cimagf(d);

            tr = dr * crealf(xi) - di * cimagf(xi);
            ti = dr * cimagf(xi) + di * crealf(xi);
        }

        for (int32_t j = lo; j < hi; ++j) {
            float complex az = a[(ptrdiff_t)i * rs + (ptrdiff_t)j * cs];
            float complex xz = x[(ptrdiff_t)j * incx];
            float ar = crealf(az), ai = cnj ? -cimagf(az) : cimagf(az);

            tr += ar * crealf(xz) - ai * cimagf(xz);
            ti += ar * cimagf(xz) + ai * crealf(xz);
        }

        x[(ptrdiff_t)i * incx] = tr + ti * I;
    }
}

void
tinyblas_ztrmv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const double complex *restrict a, int32_t lda,
        double complex *restrict x, int32_t incx)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t rs = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t cs = notrans ? 1 : (ptrdiff_t)lda;
    int upper = notrans ? (uplo == TINYBLAS_UPPER) : (uplo != TINYBLAS_UPPER);
    int unit = (diag == TINYBLAS_UNIT);
    int cnj = (trans == TINYBLAS_CONJ_TRANS);
    int32_t i, step, last;

    if (n <= 0) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    i    = upper ? 0 : n - 1;
    step = upper ? 1 : -1;
    last = upper ? n : -1;

    for (; i != last; i += step) {
        double complex xi = x[(ptrdiff_t)i * incx];
        double tr, ti;
        int32_t lo = upper ? i + 1 : 0;
        int32_t hi = upper ? n : i;

        if (unit) {
            tr = creal(xi);
            ti = cimag(xi);
        } else {
            double complex d = a[(ptrdiff_t)i * rs + (ptrdiff_t)i * cs];
            double dr = creal(d), di = cnj ? -cimag(d) : cimag(d);

            tr = dr * creal(xi) - di * cimag(xi);
            ti = dr * cimag(xi) + di * creal(xi);
        }

        for (int32_t j = lo; j < hi; ++j) {
            double complex az = a[(ptrdiff_t)i * rs + (ptrdiff_t)j * cs];
            double complex xz = x[(ptrdiff_t)j * incx];
            double ar = creal(az), ai = cnj ? -cimag(az) : cimag(az);

            tr += ar * creal(xz) - ai * cimag(xz);
            ti += ar * cimag(xz) + ai * creal(xz);
        }

        x[(ptrdiff_t)i * incx] = tr + ti * I;
    }
}

/*
 *  Triangular solve: strsv, dtrsv, ctrsv, ztrsv
 *
 *  Substitution in the direction the triangle dictates. Division by the
 *  diagonal rather than multiplication by a precomputed reciprocal, so the
 *  result matches what the reference produces bit for bit more often.
 */
void
tinyblas_strsv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const float *restrict a, int32_t lda, float *restrict x, int32_t incx)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t rs = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t cs = notrans ? 1 : (ptrdiff_t)lda;
    int upper = notrans ? (uplo == TINYBLAS_UPPER) : (uplo != TINYBLAS_UPPER);
    int unit = (diag == TINYBLAS_UNIT);
    int32_t i, step, last;

    if (n <= 0) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    i    = upper ? n - 1 : 0;
    step = upper ? -1 : 1;
    last = upper ? -1 : n;

    for (; i != last; i += step) {
        float t = x[(ptrdiff_t)i * incx];
        int32_t lo = upper ? i + 1 : 0;
        int32_t hi = upper ? n : i;

        for (int32_t j = lo; j < hi; ++j)
            t -= a[(ptrdiff_t)i * rs + (ptrdiff_t)j * cs]
                 * x[(ptrdiff_t)j * incx];

        x[(ptrdiff_t)i * incx] =
                unit ? t : t / a[(ptrdiff_t)i * rs + (ptrdiff_t)i * cs];
    }
}

void
tinyblas_dtrsv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const double *restrict a, int32_t lda, double *restrict x, int32_t incx)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t rs = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t cs = notrans ? 1 : (ptrdiff_t)lda;
    int upper = notrans ? (uplo == TINYBLAS_UPPER) : (uplo != TINYBLAS_UPPER);
    int unit = (diag == TINYBLAS_UNIT);
    int32_t i, step, last;

    if (n <= 0) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    i    = upper ? n - 1 : 0;
    step = upper ? -1 : 1;
    last = upper ? -1 : n;

    for (; i != last; i += step) {
        double t = x[(ptrdiff_t)i * incx];
        int32_t lo = upper ? i + 1 : 0;
        int32_t hi = upper ? n : i;

        for (int32_t j = lo; j < hi; ++j)
            t -= a[(ptrdiff_t)i * rs + (ptrdiff_t)j * cs]
                 * x[(ptrdiff_t)j * incx];

        x[(ptrdiff_t)i * incx] =
                unit ? t : t / a[(ptrdiff_t)i * rs + (ptrdiff_t)i * cs];
    }
}

void
tinyblas_ctrsv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const float complex *restrict a, int32_t lda,
        float complex *restrict x, int32_t incx)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t rs = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t cs = notrans ? 1 : (ptrdiff_t)lda;
    int upper = notrans ? (uplo == TINYBLAS_UPPER) : (uplo != TINYBLAS_UPPER);
    int unit = (diag == TINYBLAS_UNIT);
    int cnj = (trans == TINYBLAS_CONJ_TRANS);
    int32_t i, step, last;

    if (n <= 0) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    i    = upper ? n - 1 : 0;
    step = upper ? -1 : 1;
    last = upper ? -1 : n;

    for (; i != last; i += step) {
        float complex xi = x[(ptrdiff_t)i * incx];
        float tr = crealf(xi), ti = cimagf(xi);
        int32_t lo = upper ? i + 1 : 0;
        int32_t hi = upper ? n : i;

        for (int32_t j = lo; j < hi; ++j) {
            float complex az = a[(ptrdiff_t)i * rs + (ptrdiff_t)j * cs];
            float complex xz = x[(ptrdiff_t)j * incx];
            float ar = crealf(az), ai = cnj ? -cimagf(az) : cimagf(az);

            tr -= ar * crealf(xz) - ai * cimagf(xz);
            ti -= ar * cimagf(xz) + ai * crealf(xz);
        }

        if (unit) {
            x[(ptrdiff_t)i * incx] = tr + ti * I;
        } else {
            float complex d = a[(ptrdiff_t)i * rs + (ptrdiff_t)i * cs];

            x[(ptrdiff_t)i * incx] =
                    (tr + ti * I) / (cnj ? conjf(d) : d);
        }
    }
}

void
tinyblas_ztrsv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const double complex *restrict a, int32_t lda,
        double complex *restrict x, int32_t incx)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t rs = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t cs = notrans ? 1 : (ptrdiff_t)lda;
    int upper = notrans ? (uplo == TINYBLAS_UPPER) : (uplo != TINYBLAS_UPPER);
    int unit = (diag == TINYBLAS_UNIT);
    int cnj = (trans == TINYBLAS_CONJ_TRANS);
    int32_t i, step, last;

    if (n <= 0) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    i    = upper ? n - 1 : 0;
    step = upper ? -1 : 1;
    last = upper ? -1 : n;

    for (; i != last; i += step) {
        double complex xi = x[(ptrdiff_t)i * incx];
        double tr = creal(xi), ti = cimag(xi);
        int32_t lo = upper ? i + 1 : 0;
        int32_t hi = upper ? n : i;

        for (int32_t j = lo; j < hi; ++j) {
            double complex az = a[(ptrdiff_t)i * rs + (ptrdiff_t)j * cs];
            double complex xz = x[(ptrdiff_t)j * incx];
            double ar = creal(az), ai = cnj ? -cimag(az) : cimag(az);

            tr -= ar * creal(xz) - ai * cimag(xz);
            ti -= ar * cimag(xz) + ai * creal(xz);
        }

        if (unit) {
            x[(ptrdiff_t)i * incx] = tr + ti * I;
        } else {
            double complex d = a[(ptrdiff_t)i * rs + (ptrdiff_t)i * cs];

            x[(ptrdiff_t)i * incx] = (tr + ti * I) / (cnj ? conj(d) : d);
        }
    }
}

/*
 *  Rank 1 update: sger, dger
 */
void
tinyblas_sger(int32_t m, int32_t n, float alpha,
        const float *restrict x, int32_t incx,
        const float *restrict y, int32_t incy,
        float *restrict a, int32_t lda)
{
    if (m <= 0 || n <= 0 || alpha == 0.0f) return;

    assert(a && x && y);

    if (incx < 0) x += (ptrdiff_t)(1 - m) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    for (int32_t i = 0; i < m; ++i) {
        float t = alpha * x[(ptrdiff_t)i * incx];
        float *row = a + (ptrdiff_t)i * lda;

        for (int32_t j = 0; j < n; ++j)
            row[j] += t * y[(ptrdiff_t)j * incy];
    }
}

void
tinyblas_dger(int32_t m, int32_t n, double alpha,
        const double *restrict x, int32_t incx,
        const double *restrict y, int32_t incy,
        double *restrict a, int32_t lda)
{
    if (m <= 0 || n <= 0 || alpha == 0.0) return;

    assert(a && x && y);

    if (incx < 0) x += (ptrdiff_t)(1 - m) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    for (int32_t i = 0; i < m; ++i) {
        double t = alpha * x[(ptrdiff_t)i * incx];
        double *row = a + (ptrdiff_t)i * lda;

        for (int32_t j = 0; j < n; ++j)
            row[j] += t * y[(ptrdiff_t)j * incy];
    }
}

/*
 *  Complex rank 1 update: cgeru, zgeru, cgerc, zgerc
 *
 *  The conjugated and unconjugated forms differ only in the sign carried on
 *  y's imaginary part, so both go through one body with a flag.
 */
static void
geru_c(int32_t m, int32_t n, float complex alpha,
        const float complex *restrict x, int32_t incx,
        const float complex *restrict y, int32_t incy,
        float complex *restrict a, int32_t lda, int cnj)
{
    float alr, ali;

    if (m <= 0 || n <= 0 || alpha == 0.0f) return;

    assert(a && x && y);

    if (incx < 0) x += (ptrdiff_t)(1 - m) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    alr = crealf(alpha);
    ali = cimagf(alpha);

    for (int32_t i = 0; i < m; ++i) {
        float complex xz = x[(ptrdiff_t)i * incx];
        float tr = alr * crealf(xz) - ali * cimagf(xz);
        float ti = alr * cimagf(xz) + ali * crealf(xz);
        float complex *row = a + (ptrdiff_t)i * lda;

        for (int32_t j = 0; j < n; ++j) {
            float complex yz = y[(ptrdiff_t)j * incy];
            float yr = crealf(yz), yi = cnj ? -cimagf(yz) : cimagf(yz);

            row[j] = (crealf(row[j]) + (tr * yr - ti * yi))
                   + (cimagf(row[j]) + (tr * yi + ti * yr)) * I;
        }
    }
}

static void
geru_z(int32_t m, int32_t n, double complex alpha,
        const double complex *restrict x, int32_t incx,
        const double complex *restrict y, int32_t incy,
        double complex *restrict a, int32_t lda, int cnj)
{
    double alr, ali;

    if (m <= 0 || n <= 0 || alpha == 0.0) return;

    assert(a && x && y);

    if (incx < 0) x += (ptrdiff_t)(1 - m) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    alr = creal(alpha);
    ali = cimag(alpha);

    for (int32_t i = 0; i < m; ++i) {
        double complex xz = x[(ptrdiff_t)i * incx];
        double tr = alr * creal(xz) - ali * cimag(xz);
        double ti = alr * cimag(xz) + ali * creal(xz);
        double complex *row = a + (ptrdiff_t)i * lda;

        for (int32_t j = 0; j < n; ++j) {
            double complex yz = y[(ptrdiff_t)j * incy];
            double yr = creal(yz), yi = cnj ? -cimag(yz) : cimag(yz);

            row[j] = (creal(row[j]) + (tr * yr - ti * yi))
                   + (cimag(row[j]) + (tr * yi + ti * yr)) * I;
        }
    }
}

void
tinyblas_cgeru(int32_t m, int32_t n, float complex alpha,
        const float complex *restrict x, int32_t incx,
        const float complex *restrict y, int32_t incy,
        float complex *restrict a, int32_t lda)
{
    geru_c(m, n, alpha, x, incx, y, incy, a, lda, 0);
}

void
tinyblas_zgeru(int32_t m, int32_t n, double complex alpha,
        const double complex *restrict x, int32_t incx,
        const double complex *restrict y, int32_t incy,
        double complex *restrict a, int32_t lda)
{
    geru_z(m, n, alpha, x, incx, y, incy, a, lda, 0);
}

void
tinyblas_cgerc(int32_t m, int32_t n, float complex alpha,
        const float complex *restrict x, int32_t incx,
        const float complex *restrict y, int32_t incy,
        float complex *restrict a, int32_t lda)
{
    geru_c(m, n, alpha, x, incx, y, incy, a, lda, 1);
}

void
tinyblas_zgerc(int32_t m, int32_t n, double complex alpha,
        const double complex *restrict x, int32_t incx,
        const double complex *restrict y, int32_t incy,
        double complex *restrict a, int32_t lda)
{
    geru_z(m, n, alpha, x, incx, y, incy, a, lda, 1);
}

/*
 *  Symmetric rank 1 update: ssyr, dsyr
 */
void
tinyblas_ssyr(enum tinyblas_uplo uplo, int32_t n, float alpha,
        const float *restrict x, int32_t incx, float *restrict a, int32_t lda)
{
    if (n <= 0 || alpha == 0.0f) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    for (int32_t i = 0; i < n; ++i) {
        float t = alpha * x[(ptrdiff_t)i * incx];
        float *row = a + (ptrdiff_t)i * lda;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i + 1;

        for (int32_t j = lo; j < hi; ++j)
            row[j] += t * x[(ptrdiff_t)j * incx];
    }
}

void
tinyblas_dsyr(enum tinyblas_uplo uplo, int32_t n, double alpha,
        const double *restrict x, int32_t incx, double *restrict a, int32_t lda)
{
    if (n <= 0 || alpha == 0.0) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    for (int32_t i = 0; i < n; ++i) {
        double t = alpha * x[(ptrdiff_t)i * incx];
        double *row = a + (ptrdiff_t)i * lda;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i + 1;

        for (int32_t j = lo; j < hi; ++j)
            row[j] += t * x[(ptrdiff_t)j * incx];
    }
}

/*
 *  Hermitian rank 1 update: cher, zher
 *
 *  alpha is real, which is what keeps A hermitian. The diagonal is written as
 *  a real number rather than accumulated, so rounding cannot drift it off the
 *  real axis over repeated updates.
 */
void
tinyblas_cher(enum tinyblas_uplo uplo, int32_t n, float alpha,
        const float complex *restrict x, int32_t incx,
        float complex *restrict a, int32_t lda)
{
    if (n <= 0 || alpha == 0.0f) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    for (int32_t i = 0; i < n; ++i) {
        float complex xz = x[(ptrdiff_t)i * incx];
        float tr = alpha * crealf(xz), ti = alpha * cimagf(xz);
        float complex *row = a + (ptrdiff_t)i * lda;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i + 1 : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i;

        for (int32_t j = lo; j < hi; ++j) {
            float complex yz = x[(ptrdiff_t)j * incx];
            float yr = crealf(yz), yi = -cimagf(yz);

            row[j] = (crealf(row[j]) + (tr * yr - ti * yi))
                   + (cimagf(row[j]) + (tr * yi + ti * yr)) * I;
        }

        row[i] = (crealf(row[i]) + (tr * crealf(xz) + ti * cimagf(xz))) + 0.0f * I;
    }
}

void
tinyblas_zher(enum tinyblas_uplo uplo, int32_t n, double alpha,
        const double complex *restrict x, int32_t incx,
        double complex *restrict a, int32_t lda)
{
    if (n <= 0 || alpha == 0.0) return;

    assert(a && x);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;

    for (int32_t i = 0; i < n; ++i) {
        double complex xz = x[(ptrdiff_t)i * incx];
        double tr = alpha * creal(xz), ti = alpha * cimag(xz);
        double complex *row = a + (ptrdiff_t)i * lda;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i + 1 : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i;

        for (int32_t j = lo; j < hi; ++j) {
            double complex yz = x[(ptrdiff_t)j * incx];
            double yr = creal(yz), yi = -cimag(yz);

            row[j] = (creal(row[j]) + (tr * yr - ti * yi))
                   + (cimag(row[j]) + (tr * yi + ti * yr)) * I;
        }

        row[i] = (creal(row[i]) + (tr * creal(xz) + ti * cimag(xz))) + 0.0 * I;
    }
}

/*
 *  Symmetric rank 2 update: ssyr2, dsyr2
 */
void
tinyblas_ssyr2(enum tinyblas_uplo uplo, int32_t n, float alpha,
        const float *restrict x, int32_t incx,
        const float *restrict y, int32_t incy,
        float *restrict a, int32_t lda)
{
    if (n <= 0 || alpha == 0.0f) return;

    assert(a && x && y);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    for (int32_t i = 0; i < n; ++i) {
        float t1 = alpha * x[(ptrdiff_t)i * incx];
        float t2 = alpha * y[(ptrdiff_t)i * incy];
        float *row = a + (ptrdiff_t)i * lda;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i + 1;

        for (int32_t j = lo; j < hi; ++j)
            row[j] += t1 * y[(ptrdiff_t)j * incy] + t2 * x[(ptrdiff_t)j * incx];
    }
}

void
tinyblas_dsyr2(enum tinyblas_uplo uplo, int32_t n, double alpha,
        const double *restrict x, int32_t incx,
        const double *restrict y, int32_t incy,
        double *restrict a, int32_t lda)
{
    if (n <= 0 || alpha == 0.0) return;

    assert(a && x && y);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    for (int32_t i = 0; i < n; ++i) {
        double t1 = alpha * x[(ptrdiff_t)i * incx];
        double t2 = alpha * y[(ptrdiff_t)i * incy];
        double *row = a + (ptrdiff_t)i * lda;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i + 1;

        for (int32_t j = lo; j < hi; ++j)
            row[j] += t1 * y[(ptrdiff_t)j * incy] + t2 * x[(ptrdiff_t)j * incx];
    }
}

/*
 *  Hermitian rank 2 update: cher2, zher2
 *
 *      A <- alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
 *
 *  The conjugate on the second term is what keeps the result hermitian for a
 *  complex alpha. As in her, the diagonal is stored real.
 */
void
tinyblas_cher2(enum tinyblas_uplo uplo, int32_t n, float complex alpha,
        const float complex *restrict x, int32_t incx,
        const float complex *restrict y, int32_t incy,
        float complex *restrict a, int32_t lda)
{
    float alr, ali;

    if (n <= 0 || alpha == 0.0f) return;

    assert(a && x && y);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    alr = crealf(alpha);
    ali = cimagf(alpha);

    for (int32_t i = 0; i < n; ++i) {
        float complex xi = x[(ptrdiff_t)i * incx];
        float complex yi = y[(ptrdiff_t)i * incy];

        /* t1 = alpha * x[i], t2 = conj(alpha) * y[i] */
        float t1r = alr * crealf(xi) - ali * cimagf(xi);
        float t1i = alr * cimagf(xi) + ali * crealf(xi);
        float t2r = alr * crealf(yi) + ali * cimagf(yi);
        float t2i = alr * cimagf(yi) - ali * crealf(yi);

        float complex *row = a + (ptrdiff_t)i * lda;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i + 1 : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i;

        for (int32_t j = lo; j < hi; ++j) {
            float complex yz = y[(ptrdiff_t)j * incy];
            float complex xz = x[(ptrdiff_t)j * incx];
            float yr = crealf(yz), yj = -cimagf(yz);
            float xr = crealf(xz), xj = -cimagf(xz);

            row[j] = (crealf(row[j]) + (t1r * yr - t1i * yj)
                                     + (t2r * xr - t2i * xj))
                   + (cimagf(row[j]) + (t1r * yj + t1i * yr)
                                     + (t2r * xj + t2i * xr)) * I;
        }

        row[i] = (crealf(row[i]) + 2.0f * (t1r * crealf(yi) + t1i * cimagf(yi)))
               + 0.0f * I;
    }
}

void
tinyblas_zher2(enum tinyblas_uplo uplo, int32_t n, double complex alpha,
        const double complex *restrict x, int32_t incx,
        const double complex *restrict y, int32_t incy,
        double complex *restrict a, int32_t lda)
{
    double alr, ali;

    if (n <= 0 || alpha == 0.0) return;

    assert(a && x && y);

    if (incx < 0) x += (ptrdiff_t)(1 - n) * incx;
    if (incy < 0) y += (ptrdiff_t)(1 - n) * incy;

    alr = creal(alpha);
    ali = cimag(alpha);

    for (int32_t i = 0; i < n; ++i) {
        double complex xi = x[(ptrdiff_t)i * incx];
        double complex yi = y[(ptrdiff_t)i * incy];

        double t1r = alr * creal(xi) - ali * cimag(xi);
        double t1i = alr * cimag(xi) + ali * creal(xi);
        double t2r = alr * creal(yi) + ali * cimag(yi);
        double t2i = alr * cimag(yi) - ali * creal(yi);

        double complex *row = a + (ptrdiff_t)i * lda;
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i + 1 : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i;

        for (int32_t j = lo; j < hi; ++j) {
            double complex yz = y[(ptrdiff_t)j * incy];
            double complex xz = x[(ptrdiff_t)j * incx];
            double yr = creal(yz), yj = -cimag(yz);
            double xr = creal(xz), xj = -cimag(xz);

            row[j] = (creal(row[j]) + (t1r * yr - t1i * yj)
                                    + (t2r * xr - t2i * xj))
                   + (cimag(row[j]) + (t1r * yj + t1i * yr)
                                    + (t2r * xj + t2i * xr)) * I;
        }

        row[i] = (creal(row[i]) + 2.0 * (t1r * creal(yi) + t1i * cimag(yi)))
               + 0.0 * I;
    }
}
