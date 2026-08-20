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
#include <stdlib.h>
#include <assert.h>
#include <complex.h>

#include "tinyblas_level2.h"
#include "tinyblas_level3.h"
#include "tinyblas_common.h"

/* Everything here is gemm wearing a different hat, so none of it reimplements
 * a kernel. The three shapes:
 *
 *   symm, hemm, trmm  expand the structured operand into a dense square and
 *                     hand it to gemm. Expanding costs O(n^2) against O(n^2 k)
 *                     of arithmetic, so it disappears into the noise.
 *
 *   syrk and friends  block C into column panels. Everything strictly off the
 *                     diagonal is a plain rectangular gemm straight into C;
 *                     only the small diagonal block needs a scratch buffer and
 *                     a triangle merge, so the flop count stays at n^2 k / 2
 *                     rather than the n^2 k a full product would cost.
 *
 *   trsm              solves one column, or one row on the right, at a time
 *                     with the level 2 trsv.
 *
 * If the expansion cannot be allocated, the routine falls back to a path that
 * needs no memory at all: symv or trmv for the real types, and for complex,
 * which has no symv to fall back on, a naive triple loop that doubles as the
 * small-size path. Slower, but the library never fails.
 *
 * ponytail: trmm expands to a dense square and so does twice the arithmetic a
 * triangular multiply needs, and trsm is unblocked, so it runs at trsv speed
 * rather than gemm speed. Both are correct and both have the right asymptotic
 * cost. make bench at n = 1024 now prices the shortcut:
 *
 *     dtrmm  47% of gemm      ztrmm  49% of gemm
 *     dtrsm 2.5% of gemm      ztrsm 7.3% of gemm
 *
 * trmm is close to the 50% its doubled flop count predicts, so the expansion
 * itself is nearly free and there is little left short of a real triangular
 * kernel.
 *
 * trsm was the genuinely slow one, at 2.5%: trsv walks a column of B, and in
 * row-major that is a strided access that never vectorises. side LEFT is now
 * blocked (see trsm_left_? below) and runs at 62% of gemm for double, 71% for
 * double complex -- 26x and 11x what the trsv sweep managed. side RIGHT is
 * still the trsv sweep, because there it is B's columns that are strided and
 * the same blocking wants a transposed scratch copy to pay off.
 */

/* Column-panel width for the rank-k updates. The scratch diagonal block is
 * RK_NB squared, which at 64 is 64 KB for double complex, comfortably on the
 * stack and comfortably inside L2. */
#define RK_NB 64

/* Below this order the dense expansion and its malloc cost more than the
 * arithmetic they save, so complex symm and hemm take the same scratch-free
 * path they take when the malloc fails.
 * ponytail: 16 is a guess. No bench row owns this number yet; if one ever
 * does, sweep it the way the gemm blocking constants were swept. */
#define SYMM_SMALL 16

/* Row-block height for the blocked trsm. The dense diagonal block is TRSM_MB
 * squared, which at 64 is 64 KB for double complex: the same stack budget
 * rk_blocked already spends, and it stays inside L2.
 *
 * Swept with ./build/bench trsm at n = 1024, best of three (GFLOP/s):
 *
 *     TRSM_MB     32      64     128
 *     dtrsm    21.36   25.12   22.64
 *     ztrsm    18.09   23.40   24.05
 *
 * Flat from 64 up and clearly worse below it, which is what the cost model
 * predicts: the unblocked fraction of the work is TRSM_MB/m, but shrinking
 * the block also shrinks the m of every gemm update. 64 it is. */
#define TRSM_MB 64

/*
 *  C <- beta * C over one triangle only
 *
 *  beta == 0 stores rather than multiplies, so an uninitialised or NaN-filled
 *  C is legal input here exactly as it is for gemm.
 */
static void
scale_tri_s(enum tinyblas_uplo uplo, int32_t n, float beta, float *restrict c, int32_t ldc)
{
    if (beta == 1.0f) return;

    for (int32_t i = 0; i < n; ++i) {
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i + 1;

        for (int32_t j = lo; j < hi; ++j)
            if (beta == 0.0f) c[(ptrdiff_t)i * ldc + j] = 0.0f;
            else              c[(ptrdiff_t)i * ldc + j] *= beta;
    }
}

static void
scale_tri_d(enum tinyblas_uplo uplo, int32_t n, double beta, double *restrict c, int32_t ldc)
{
    if (beta == 1.0) return;

    for (int32_t i = 0; i < n; ++i) {
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i + 1;

        for (int32_t j = lo; j < hi; ++j)
            if (beta == 0.0) c[(ptrdiff_t)i * ldc + j] = 0.0;
            else             c[(ptrdiff_t)i * ldc + j] *= beta;
    }
}

/*
 *  Mirror a stored triangle into a full dense square
 */
static void
expand_sym_s(enum tinyblas_uplo uplo, int32_t n, const float *restrict a, int32_t lda, float *restrict out)
{
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            int stored = (uplo == TINYBLAS_UPPER) ? (j >= i) : (j <= i);

            out[(ptrdiff_t)i * n + j] = stored
                    ? a[(ptrdiff_t)i * lda + j]
                    : a[(ptrdiff_t)j * lda + i];
        }
    }
}

static void
expand_sym_d(enum tinyblas_uplo uplo, int32_t n, const double *restrict a, int32_t lda, double *restrict out)
{
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            int stored = (uplo == TINYBLAS_UPPER) ? (j >= i) : (j <= i);

            out[(ptrdiff_t)i * n + j] = stored
                    ? a[(ptrdiff_t)i * lda + j]
                    : a[(ptrdiff_t)j * lda + i];
        }
    }
}

/*
 *  Write op(A) out as a dense square, zeros outside the triangle
 *
 *  Transposing is a swap of the two indices, and a unit diagonal overrides
 *  whatever happens to be stored on the diagonal.
 */
static void
expand_tri_s(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const float *restrict a, int32_t lda, float *restrict out)
{
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            int32_t r = (trans == TINYBLAS_NONE) ? i : j;
            int32_t c = (trans == TINYBLAS_NONE) ? j : i;
            int in = (uplo == TINYBLAS_UPPER) ? (c >= r) : (c <= r);
            float v = 0.0f;

            if (in) v = (r == c && diag == TINYBLAS_UNIT)
                      ? 1.0f : a[(ptrdiff_t)r * lda + c];

            out[(ptrdiff_t)i * n + j] = v;
        }
    }
}

static void
expand_tri_d(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const double *restrict a, int32_t lda, double *restrict out)
{
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            int32_t r = (trans == TINYBLAS_NONE) ? i : j;
            int32_t c = (trans == TINYBLAS_NONE) ? j : i;
            int in = (uplo == TINYBLAS_UPPER) ? (c >= r) : (c <= r);
            double v = 0.0;

            if (in) v = (r == c && diag == TINYBLAS_UNIT)
                      ? 1.0 : a[(ptrdiff_t)r * lda + c];

            out[(ptrdiff_t)i * n + j] = v;
        }
    }
}

/*
 *  C_triangle += alpha * opA(A) * opB(B), blocked over column panels of C
 *
 *  astep and bstep say how to reach conceptual row i of each operand: a row
 *  stride when the operand is untransposed, one element when it is not. The
 *  caller picks ta and tb, which is what lets syrk, herk, syr2k and her2k all
 *  come through here.
 */
static void
rk_blocked_d(enum tinyblas_uplo uplo, int32_t n, int32_t k, double alpha,
        const double *restrict a, int32_t lda, ptrdiff_t astep,
        const double *restrict b, int32_t ldb, ptrdiff_t bstep,
        enum tinyblas_op ta, enum tinyblas_op tb,
        double *restrict c, int32_t ldc)
{
    double tmp[RK_NB * RK_NB];

    for (int32_t jb = 0; jb < n; jb += RK_NB) {
        int32_t nb = (n - jb < RK_NB) ? n - jb : RK_NB;

        /* the part of this panel that lies wholly off the diagonal block */
        if (uplo == TINYBLAS_UPPER && jb > 0)
            tinyblas_dgemm(ta, tb, jb, nb, k, alpha,
                           a, lda, b + (ptrdiff_t)jb * bstep, ldb,
                           1.0, c + jb, ldc);

        if (uplo == TINYBLAS_LOWER && jb + nb < n)
            tinyblas_dgemm(ta, tb, n - (jb + nb), nb, k, alpha,
                           a + (ptrdiff_t)(jb + nb) * astep, lda,
                           b + (ptrdiff_t)jb * bstep, ldb,
                           1.0, c + (ptrdiff_t)(jb + nb) * ldc + jb, ldc);

        /* the diagonal block straddles the triangle, so it goes via scratch */
        tinyblas_dgemm(ta, tb, nb, nb, k, alpha,
                       a + (ptrdiff_t)jb * astep, lda,
                       b + (ptrdiff_t)jb * bstep, ldb,
                       0.0, tmp, nb);

        for (int32_t i = 0; i < nb; ++i) {
            int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
            int32_t hi = (uplo == TINYBLAS_UPPER) ? nb : i + 1;

            for (int32_t j = lo; j < hi; ++j)
                c[(ptrdiff_t)(jb + i) * ldc + jb + j] += tmp[i * nb + j];
        }
    }
}

static void
rk_blocked_s(enum tinyblas_uplo uplo, int32_t n, int32_t k, float alpha,
        const float *restrict a, int32_t lda, ptrdiff_t astep,
        const float *restrict b, int32_t ldb, ptrdiff_t bstep,
        enum tinyblas_op ta, enum tinyblas_op tb,
        float *restrict c, int32_t ldc)
{
    float tmp[RK_NB * RK_NB];

    for (int32_t jb = 0; jb < n; jb += RK_NB) {
        int32_t nb = (n - jb < RK_NB) ? n - jb : RK_NB;

        if (uplo == TINYBLAS_UPPER && jb > 0)
            tinyblas_sgemm(ta, tb, jb, nb, k, alpha,
                           a, lda, b + (ptrdiff_t)jb * bstep, ldb,
                           1.0f, c + jb, ldc);

        if (uplo == TINYBLAS_LOWER && jb + nb < n)
            tinyblas_sgemm(ta, tb, n - (jb + nb), nb, k, alpha,
                           a + (ptrdiff_t)(jb + nb) * astep, lda,
                           b + (ptrdiff_t)jb * bstep, ldb,
                           1.0f, c + (ptrdiff_t)(jb + nb) * ldc + jb, ldc);

        tinyblas_sgemm(ta, tb, nb, nb, k, alpha,
                       a + (ptrdiff_t)jb * astep, lda,
                       b + (ptrdiff_t)jb * bstep, ldb,
                       0.0f, tmp, nb);

        for (int32_t i = 0; i < nb; ++i) {
            int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
            int32_t hi = (uplo == TINYBLAS_UPPER) ? nb : i + 1;

            for (int32_t j = lo; j < hi; ++j)
                c[(ptrdiff_t)(jb + i) * ldc + jb + j] += tmp[i * nb + j];
        }
    }
}

static void
scale_tri_c(enum tinyblas_uplo uplo, int32_t n, float complex beta,
        float complex *restrict c, int32_t ldc)
{
    float br = crealf(beta), bi = cimagf(beta);

    if (beta == 1.0f) return;

    for (int32_t i = 0; i < n; ++i) {
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i + 1;

        for (int32_t j = lo; j < hi; ++j) {
            float complex z = c[(ptrdiff_t)i * ldc + j];

            c[(ptrdiff_t)i * ldc + j] = (beta == 0.0f)
                    ? 0.0f
                    : (br * crealf(z) - bi * cimagf(z))
                      + (br * cimagf(z) + bi * crealf(z)) * I;
        }
    }
}

static void
scale_tri_z(enum tinyblas_uplo uplo, int32_t n, double complex beta,
        double complex *restrict c, int32_t ldc)
{
    double br = creal(beta), bi = cimag(beta);

    if (beta == 1.0) return;

    for (int32_t i = 0; i < n; ++i) {
        int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
        int32_t hi = (uplo == TINYBLAS_UPPER) ? n : i + 1;

        for (int32_t j = lo; j < hi; ++j) {
            double complex z = c[(ptrdiff_t)i * ldc + j];

            c[(ptrdiff_t)i * ldc + j] = (beta == 0.0)
                    ? 0.0
                    : (br * creal(z) - bi * cimag(z))
                      + (br * cimag(z) + bi * creal(z)) * I;
        }
    }
}

/* force the diagonal of a hermitian result back onto the real axis */
static void
real_diag_c(int32_t n, float complex *restrict c, int32_t ldc)
{
    for (int32_t i = 0; i < n; ++i)
        c[(ptrdiff_t)i * ldc + i] = crealf(c[(ptrdiff_t)i * ldc + i]) + 0.0f * I;
}

static void
real_diag_z(int32_t n, double complex *restrict c, int32_t ldc)
{
    for (int32_t i = 0; i < n; ++i)
        c[(ptrdiff_t)i * ldc + i] = creal(c[(ptrdiff_t)i * ldc + i]) + 0.0 * I;
}

/*
 *  One element of a symmetric or hermitian operand, mirrored on demand
 *
 *  The single definition of what the unstored triangle means; the dense
 *  expansion and the no-memory fallback both read through here.
 */
static float complex
sym_at_c(enum tinyblas_uplo uplo, int herm,
        const float complex *restrict a, int32_t lda, int32_t r, int32_t c)
{
    int stored = (uplo == TINYBLAS_UPPER) ? (c >= r) : (c <= r);
    float complex v = stored ? a[(ptrdiff_t)r * lda + c]
                             : a[(ptrdiff_t)c * lda + r];

    if (herm && !stored) v = conjf(v);
    if (herm && r == c)  v = crealf(v) + 0.0f * I;

    return v;
}

static double complex
sym_at_z(enum tinyblas_uplo uplo, int herm,
        const double complex *restrict a, int32_t lda, int32_t r, int32_t c)
{
    int stored = (uplo == TINYBLAS_UPPER) ? (c >= r) : (c <= r);
    double complex v = stored ? a[(ptrdiff_t)r * lda + c]
                              : a[(ptrdiff_t)c * lda + r];

    if (herm && !stored) v = conj(v);
    if (herm && r == c)  v = creal(v) + 0.0 * I;

    return v;
}

static void
expand_sym_c(enum tinyblas_uplo uplo, int32_t n,
        const float complex *restrict a, int32_t lda,
        float complex *restrict out, int herm)
{
    for (int32_t i = 0; i < n; ++i)
        for (int32_t j = 0; j < n; ++j)
            out[(ptrdiff_t)i * n + j] = sym_at_c(uplo, herm, a, lda, i, j);
}

static void
expand_sym_z(enum tinyblas_uplo uplo, int32_t n,
        const double complex *restrict a, int32_t lda,
        double complex *restrict out, int herm)
{
    for (int32_t i = 0; i < n; ++i)
        for (int32_t j = 0; j < n; ++j)
            out[(ptrdiff_t)i * n + j] = sym_at_z(uplo, herm, a, lda, i, j);
}

static void
expand_tri_c(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const float complex *restrict a, int32_t lda, float complex *restrict out)
{
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            int32_t r = (trans == TINYBLAS_NONE) ? i : j;
            int32_t c = (trans == TINYBLAS_NONE) ? j : i;
            int in = (uplo == TINYBLAS_UPPER) ? (c >= r) : (c <= r);
            float complex v = 0.0f;

            if (in) {
                if (r == c && diag == TINYBLAS_UNIT) {
                    v = 1.0f;
                } else {
                    v = a[(ptrdiff_t)r * lda + c];

                    if (trans == TINYBLAS_CONJ_TRANS) v = conjf(v);
                }
            }

            out[(ptrdiff_t)i * n + j] = v;
        }
    }
}

static void
expand_tri_z(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n,
        const double complex *restrict a, int32_t lda,
        double complex *restrict out)
{
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            int32_t r = (trans == TINYBLAS_NONE) ? i : j;
            int32_t c = (trans == TINYBLAS_NONE) ? j : i;
            int in = (uplo == TINYBLAS_UPPER) ? (c >= r) : (c <= r);
            double complex v = 0.0;

            if (in) {
                if (r == c && diag == TINYBLAS_UNIT) {
                    v = 1.0;
                } else {
                    v = a[(ptrdiff_t)r * lda + c];

                    if (trans == TINYBLAS_CONJ_TRANS) v = conj(v);
                }
            }

            out[(ptrdiff_t)i * n + j] = v;
        }
    }
}

/*
 *  B <- alpha * B over a dense m by n block
 *
 *  alpha == 0 stores zeros rather than multiplying, which is what lets trmm
 *  take a NaN-filled B and what lets symm skip reading C.
 */
static void
scale_mat_c(int32_t m, int32_t n, float complex alpha,
        float complex *restrict b, int32_t ldb)
{
    float ar = crealf(alpha), ai = cimagf(alpha);

    if (alpha == 1.0f) return;

    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            float complex z = b[(ptrdiff_t)i * ldb + j];

            b[(ptrdiff_t)i * ldb + j] = (alpha == 0.0f)
                    ? 0.0f
                    : (ar * crealf(z) - ai * cimagf(z))
                      + (ar * cimagf(z) + ai * crealf(z)) * I;
        }
    }
}

static void
scale_mat_z(int32_t m, int32_t n, double complex alpha,
        double complex *restrict b, int32_t ldb)
{
    double ar = creal(alpha), ai = cimag(alpha);

    if (alpha == 1.0) return;

    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            double complex z = b[(ptrdiff_t)i * ldb + j];

            b[(ptrdiff_t)i * ldb + j] = (alpha == 0.0)
                    ? 0.0
                    : (ar * creal(z) - ai * cimag(z))
                      + (ar * cimag(z) + ai * creal(z)) * I;
        }
    }
}

/* x <- conj(x) over a contiguous row */
static void
conj_vec_c(int32_t n, float complex *restrict x)
{
    for (int32_t i = 0; i < n; ++i) x[i] = conjf(x[i]);
}

static void
conj_vec_z(int32_t n, double complex *restrict x)
{
    for (int32_t i = 0; i < n; ++i) x[i] = conj(x[i]);
}

/*
 *  symm and hemm with no scratch memory at all
 *
 *  The real path falls back on symv when the dense expansion cannot be
 *  allocated. There is no complex symv to fall back on, so this triple loop
 *  is the floor that keeps the promise that the library never fails.
 */
static void
symm_naive_c(enum tinyblas_side side, enum tinyblas_uplo uplo, int herm,
        int32_t m, int32_t n, float complex alpha,
        const float complex *restrict a, int32_t lda,
        const float complex *restrict b, int32_t ldb,
        float complex beta, float complex *restrict c, int32_t ldc)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    float alr = crealf(alpha), ali = cimagf(alpha);
    float ber = crealf(beta),  bei = cimagf(beta);

    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            float sr = 0.0f, si = 0.0f, cr, ci;

            for (int32_t p = 0; p < na; ++p) {
                float complex av = (side == TINYBLAS_LEFT)
                        ? sym_at_c(uplo, herm, a, lda, i, p)
                        : sym_at_c(uplo, herm, a, lda, p, j);
                float complex bv = (side == TINYBLAS_LEFT)
                        ? b[(ptrdiff_t)p * ldb + j]
                        : b[(ptrdiff_t)i * ldb + p];
                float vr = crealf(av), vi = cimagf(av);
                float wr = crealf(bv), wi = cimagf(bv);

                sr += vr * wr - vi * wi;
                si += vr * wi + vi * wr;
            }

            cr = alr * sr - ali * si;
            ci = alr * si + ali * sr;

            if (beta != 0.0f) {
                float complex z = c[(ptrdiff_t)i * ldc + j];

                cr += ber * crealf(z) - bei * cimagf(z);
                ci += ber * cimagf(z) + bei * crealf(z);
            }

            c[(ptrdiff_t)i * ldc + j] = cr + ci * I;
        }
    }
}

static void
symm_naive_z(enum tinyblas_side side, enum tinyblas_uplo uplo, int herm,
        int32_t m, int32_t n, double complex alpha,
        const double complex *restrict a, int32_t lda,
        const double complex *restrict b, int32_t ldb,
        double complex beta, double complex *restrict c, int32_t ldc)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    double alr = creal(alpha), ali = cimag(alpha);
    double ber = creal(beta),  bei = cimag(beta);

    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            double sr = 0.0, si = 0.0, cr, ci;

            for (int32_t p = 0; p < na; ++p) {
                double complex av = (side == TINYBLAS_LEFT)
                        ? sym_at_z(uplo, herm, a, lda, i, p)
                        : sym_at_z(uplo, herm, a, lda, p, j);
                double complex bv = (side == TINYBLAS_LEFT)
                        ? b[(ptrdiff_t)p * ldb + j]
                        : b[(ptrdiff_t)i * ldb + p];
                double vr = creal(av), vi = cimag(av);
                double wr = creal(bv), wi = cimag(bv);

                sr += vr * wr - vi * wi;
                si += vr * wi + vi * wr;
            }

            cr = alr * sr - ali * si;
            ci = alr * si + ali * sr;

            if (beta != 0.0) {
                double complex z = c[(ptrdiff_t)i * ldc + j];

                cr += ber * creal(z) - bei * cimag(z);
                ci += ber * cimag(z) + bei * creal(z);
            }

            c[(ptrdiff_t)i * ldc + j] = cr + ci * I;
        }
    }
}

static void
rk_blocked_c(enum tinyblas_uplo uplo, int32_t n, int32_t k, float complex alpha,
        const float complex *restrict a, int32_t lda, ptrdiff_t astep,
        const float complex *restrict b, int32_t ldb, ptrdiff_t bstep,
        enum tinyblas_op ta, enum tinyblas_op tb,
        float complex *restrict c, int32_t ldc)
{
    float complex tmp[RK_NB * RK_NB];

    for (int32_t jb = 0; jb < n; jb += RK_NB) {
        int32_t nb = (n - jb < RK_NB) ? n - jb : RK_NB;

        if (uplo == TINYBLAS_UPPER && jb > 0)
            tinyblas_cgemm(ta, tb, jb, nb, k, alpha,
                           a, lda, b + (ptrdiff_t)jb * bstep, ldb,
                           1.0f, c + jb, ldc);

        if (uplo == TINYBLAS_LOWER && jb + nb < n)
            tinyblas_cgemm(ta, tb, n - (jb + nb), nb, k, alpha,
                           a + (ptrdiff_t)(jb + nb) * astep, lda,
                           b + (ptrdiff_t)jb * bstep, ldb,
                           1.0f, c + (ptrdiff_t)(jb + nb) * ldc + jb, ldc);

        tinyblas_cgemm(ta, tb, nb, nb, k, alpha,
                       a + (ptrdiff_t)jb * astep, lda,
                       b + (ptrdiff_t)jb * bstep, ldb,
                       0.0f, tmp, nb);

        for (int32_t i = 0; i < nb; ++i) {
            int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
            int32_t hi = (uplo == TINYBLAS_UPPER) ? nb : i + 1;

            for (int32_t j = lo; j < hi; ++j)
                c[(ptrdiff_t)(jb + i) * ldc + jb + j] += tmp[i * nb + j];
        }
    }
}

static void
rk_blocked_z(enum tinyblas_uplo uplo, int32_t n, int32_t k, double complex alpha,
        const double complex *restrict a, int32_t lda, ptrdiff_t astep,
        const double complex *restrict b, int32_t ldb, ptrdiff_t bstep,
        enum tinyblas_op ta, enum tinyblas_op tb,
        double complex *restrict c, int32_t ldc)
{
    double complex tmp[RK_NB * RK_NB];

    for (int32_t jb = 0; jb < n; jb += RK_NB) {
        int32_t nb = (n - jb < RK_NB) ? n - jb : RK_NB;

        if (uplo == TINYBLAS_UPPER && jb > 0)
            tinyblas_zgemm(ta, tb, jb, nb, k, alpha,
                           a, lda, b + (ptrdiff_t)jb * bstep, ldb,
                           1.0, c + jb, ldc);

        if (uplo == TINYBLAS_LOWER && jb + nb < n)
            tinyblas_zgemm(ta, tb, n - (jb + nb), nb, k, alpha,
                           a + (ptrdiff_t)(jb + nb) * astep, lda,
                           b + (ptrdiff_t)jb * bstep, ldb,
                           1.0, c + (ptrdiff_t)(jb + nb) * ldc + jb, ldc);

        tinyblas_zgemm(ta, tb, nb, nb, k, alpha,
                       a + (ptrdiff_t)jb * astep, lda,
                       b + (ptrdiff_t)jb * bstep, ldb,
                       0.0, tmp, nb);

        for (int32_t i = 0; i < nb; ++i) {
            int32_t lo = (uplo == TINYBLAS_UPPER) ? i : 0;
            int32_t hi = (uplo == TINYBLAS_UPPER) ? nb : i + 1;

            for (int32_t j = lo; j < hi; ++j)
                c[(ptrdiff_t)(jb + i) * ldc + jb + j] += tmp[i * nb + j];
        }
    }
}

/*
 *  One trmv or trsv per column on the left, per row on the right
 *
 *  A row meets op(A) from the right, so the vector routine sees op(A)
 *  transposed: A^T, A, or conj(A). Only the last has no flag of its own, and
 *  it is reached by conjugating the row on the way in and back out.
 *
 *  This is trsm's whole implementation and trmm's no-memory fallback, so the
 *  conjugate trick above is written once and exercised on every trsm call.
 */
static void
tri_sweep_c(int solve, enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, const float complex *restrict a, int32_t lda,
        float complex *restrict b, int32_t ldb)
{
    if (side == TINYBLAS_LEFT) {
        for (int32_t j = 0; j < n; ++j)
            if (solve) tinyblas_ctrsv(uplo, trans, diag, m, a, lda, b + j, ldb);
            else       tinyblas_ctrmv(uplo, trans, diag, m, a, lda, b + j, ldb);

        return;
    }

    {
        enum tinyblas_op fl = (trans == TINYBLAS_NONE)
                               ? TINYBLAS_TRANS : TINYBLAS_NONE;
        int cj = (trans == TINYBLAS_CONJ_TRANS);

        for (int32_t i = 0; i < m; ++i) {
            float complex *row = b + (ptrdiff_t)i * ldb;

            if (cj) conj_vec_c(n, row);

            if (solve) tinyblas_ctrsv(uplo, fl, diag, n, a, lda, row, 1);
            else       tinyblas_ctrmv(uplo, fl, diag, n, a, lda, row, 1);

            if (cj) conj_vec_c(n, row);
        }
    }
}

static void
tri_sweep_z(int solve, enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, const double complex *restrict a, int32_t lda,
        double complex *restrict b, int32_t ldb)
{
    if (side == TINYBLAS_LEFT) {
        for (int32_t j = 0; j < n; ++j)
            if (solve) tinyblas_ztrsv(uplo, trans, diag, m, a, lda, b + j, ldb);
            else       tinyblas_ztrmv(uplo, trans, diag, m, a, lda, b + j, ldb);

        return;
    }

    {
        enum tinyblas_op fl = (trans == TINYBLAS_NONE)
                               ? TINYBLAS_TRANS : TINYBLAS_NONE;
        int cj = (trans == TINYBLAS_CONJ_TRANS);

        for (int32_t i = 0; i < m; ++i) {
            double complex *row = b + (ptrdiff_t)i * ldb;

            if (cj) conj_vec_z(n, row);

            if (solve) tinyblas_ztrsv(uplo, fl, diag, n, a, lda, row, 1);
            else       tinyblas_ztrmv(uplo, fl, diag, n, a, lda, row, 1);

            if (cj) conj_vec_z(n, row);
        }
    }
}

/*
 *  The shared body behind csymm/chemm and zsymm/zhemm
 *
 *  herm is the only difference between the two: it decides whether the
 *  unstored triangle is a mirror or a conjugate mirror.
 */
static void
symm_impl_c(enum tinyblas_side side, enum tinyblas_uplo uplo, int herm,
        int32_t m, int32_t n, float complex alpha,
        const float complex *restrict a, int32_t lda,
        const float complex *restrict b, int32_t ldb,
        float complex beta, float complex *restrict c, int32_t ldc)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    float complex *dense;

    if (m <= 0 || n <= 0) return;

    assert(c);

    if (alpha == 0.0f) {
        scale_mat_c(m, n, beta, c, ldc);

        return;
    }

    assert(a && b);

    dense = (na < SYMM_SMALL)
          ? NULL : malloc((size_t)na * (size_t)na * sizeof(float complex));

    /* small, or out of memory: the naive loop needs no scratch at all */
    if (dense == NULL) {
        symm_naive_c(side, uplo, herm, m, n, alpha, a, lda, b, ldb,
                     beta, c, ldc);

        return;
    }

    expand_sym_c(uplo, na, a, lda, dense, herm);

    if (side == TINYBLAS_LEFT)
        tinyblas_cgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, m,
                       alpha, dense, na, b, ldb, beta, c, ldc);
    else
        tinyblas_cgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, n,
                       alpha, b, ldb, dense, na, beta, c, ldc);

    free(dense);
}

static void
symm_impl_z(enum tinyblas_side side, enum tinyblas_uplo uplo, int herm,
        int32_t m, int32_t n, double complex alpha,
        const double complex *restrict a, int32_t lda,
        const double complex *restrict b, int32_t ldb,
        double complex beta, double complex *restrict c, int32_t ldc)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    double complex *dense;

    if (m <= 0 || n <= 0) return;

    assert(c);

    if (alpha == 0.0) {
        scale_mat_z(m, n, beta, c, ldc);

        return;
    }

    assert(a && b);

    dense = (na < SYMM_SMALL)
          ? NULL : malloc((size_t)na * (size_t)na * sizeof(double complex));

    /* small, or out of memory: the naive loop needs no scratch at all */
    if (dense == NULL) {
        symm_naive_z(side, uplo, herm, m, n, alpha, a, lda, b, ldb,
                     beta, c, ldc);

        return;
    }

    expand_sym_z(uplo, na, a, lda, dense, herm);

    if (side == TINYBLAS_LEFT)
        tinyblas_zgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, m,
                       alpha, dense, na, b, ldb, beta, c, ldc);
    else
        tinyblas_zgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, n,
                       alpha, b, ldb, dense, na, beta, c, ldc);

    free(dense);
}

/*
 *  The shared body behind all four complex rank-k updates
 *
 *  herm swaps the transpose for a conjugate transpose and pins the diagonal
 *  to the real axis; rank2 adds the second pass. syrk passes a for b, exactly
 *  as the real path does.
 */
static void
rk_impl_c(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int herm, int rank2, int32_t n, int32_t k, float complex alpha,
        const float complex *restrict a, int32_t lda,
        const float complex *restrict b, int32_t ldb,
        float complex beta, float complex *restrict c, int32_t ldc)
{
    int notrans = (trans == TINYBLAS_NONE);
    enum tinyblas_op tc = herm ? TINYBLAS_CONJ_TRANS : TINYBLAS_TRANS;
    enum tinyblas_op t1 = notrans ? TINYBLAS_NONE : tc;
    enum tinyblas_op t2 = notrans ? tc : TINYBLAS_NONE;
    ptrdiff_t as = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t bs = notrans ? (ptrdiff_t)ldb : 1;

    if (n <= 0) return;

    assert(c);

    scale_tri_c(uplo, n, beta, c, ldc);

    if (k > 0 && alpha != 0.0f) {
        assert(a && b);

        rk_blocked_c(uplo, n, k, alpha, a, lda, as, b, ldb, bs, t1, t2, c, ldc);

        if (rank2)
            rk_blocked_c(uplo, n, k, herm ? conjf(alpha) : alpha,
                         b, ldb, bs, a, lda, as, t1, t2, c, ldc);
    }

    /* rounding leaves a little imaginary dust on the diagonal, and a
     * hermitian result has none by definition */
    if (herm) real_diag_c(n, c, ldc);
}

static void
rk_impl_z(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int herm, int rank2, int32_t n, int32_t k, double complex alpha,
        const double complex *restrict a, int32_t lda,
        const double complex *restrict b, int32_t ldb,
        double complex beta, double complex *restrict c, int32_t ldc)
{
    int notrans = (trans == TINYBLAS_NONE);
    enum tinyblas_op tc = herm ? TINYBLAS_CONJ_TRANS : TINYBLAS_TRANS;
    enum tinyblas_op t1 = notrans ? TINYBLAS_NONE : tc;
    enum tinyblas_op t2 = notrans ? tc : TINYBLAS_NONE;
    ptrdiff_t as = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t bs = notrans ? (ptrdiff_t)ldb : 1;

    if (n <= 0) return;

    assert(c);

    scale_tri_z(uplo, n, beta, c, ldc);

    if (k > 0 && alpha != 0.0) {
        assert(a && b);

        rk_blocked_z(uplo, n, k, alpha, a, lda, as, b, ldb, bs, t1, t2, c, ldc);

        if (rank2)
            rk_blocked_z(uplo, n, k, herm ? conj(alpha) : alpha,
                         b, ldb, bs, a, lda, as, t1, t2, c, ldc);
    }

    if (herm) real_diag_z(n, c, ldc);
}

/*
 *  Symmetric matrix-matrix product: ssymm, dsymm
 */
void
tinyblas_ssymm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        int32_t m, int32_t n, float alpha,
        const float *restrict a, int32_t lda,
        const float *restrict b, int32_t ldb,
        float beta, float *restrict c, int32_t ldc)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    float *dense;

    if (m <= 0 || n <= 0) return;

    assert(c);

    if (alpha == 0.0f) {
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                if (beta == 0.0f) c[(ptrdiff_t)i * ldc + j] = 0.0f;
                else              c[(ptrdiff_t)i * ldc + j] *= beta;

        return;
    }

    assert(a && b);

    dense = malloc((size_t)na * (size_t)na * sizeof(float));

    if (dense == NULL) {
        /* no memory: one symv per column on the left, per row on the right */
        if (side == TINYBLAS_LEFT)
            for (int32_t j = 0; j < n; ++j)
                tinyblas_ssymv(uplo, m, alpha, a, lda, b + j, ldb,
                               beta, c + j, ldc);
        else
            for (int32_t i = 0; i < m; ++i)
                tinyblas_ssymv(uplo, n, alpha, a, lda,
                               b + (ptrdiff_t)i * ldb, 1,
                               beta, c + (ptrdiff_t)i * ldc, 1);

        return;
    }

    expand_sym_s(uplo, na, a, lda, dense);

    if (side == TINYBLAS_LEFT)
        tinyblas_sgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, m,
                       alpha, dense, na, b, ldb, beta, c, ldc);
    else
        tinyblas_sgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, n,
                       alpha, b, ldb, dense, na, beta, c, ldc);

    free(dense);
}

void
tinyblas_dsymm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        int32_t m, int32_t n, double alpha,
        const double *restrict a, int32_t lda,
        const double *restrict b, int32_t ldb,
        double beta, double *restrict c, int32_t ldc)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    double *dense;

    if (m <= 0 || n <= 0) return;

    assert(c);

    if (alpha == 0.0) {
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                if (beta == 0.0) c[(ptrdiff_t)i * ldc + j] = 0.0;
                else             c[(ptrdiff_t)i * ldc + j] *= beta;

        return;
    }

    assert(a && b);

    dense = malloc((size_t)na * (size_t)na * sizeof(double));

    if (dense == NULL) {
        if (side == TINYBLAS_LEFT)
            for (int32_t j = 0; j < n; ++j)
                tinyblas_dsymv(uplo, m, alpha, a, lda, b + j, ldb,
                               beta, c + j, ldc);
        else
            for (int32_t i = 0; i < m; ++i)
                tinyblas_dsymv(uplo, n, alpha, a, lda,
                               b + (ptrdiff_t)i * ldb, 1,
                               beta, c + (ptrdiff_t)i * ldc, 1);

        return;
    }

    expand_sym_d(uplo, na, a, lda, dense);

    if (side == TINYBLAS_LEFT)
        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, m,
                       alpha, dense, na, b, ldb, beta, c, ldc);
    else
        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, n,
                       alpha, b, ldb, dense, na, beta, c, ldc);

    free(dense);
}

/*
 *  Symmetric and hermitian matrix-matrix product: csymm, zsymm, chemm, zhemm
 */
void
tinyblas_csymm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        int32_t m, int32_t n, float complex alpha,
        const float complex *restrict a, int32_t lda,
        const float complex *restrict b, int32_t ldb,
        float complex beta, float complex *restrict c, int32_t ldc)
{
    symm_impl_c(side, uplo, 0, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tinyblas_zsymm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        int32_t m, int32_t n, double complex alpha,
        const double complex *restrict a, int32_t lda,
        const double complex *restrict b, int32_t ldb,
        double complex beta, double complex *restrict c, int32_t ldc)
{
    symm_impl_z(side, uplo, 0, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tinyblas_chemm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        int32_t m, int32_t n, float complex alpha,
        const float complex *restrict a, int32_t lda,
        const float complex *restrict b, int32_t ldb,
        float complex beta, float complex *restrict c, int32_t ldc)
{
    symm_impl_c(side, uplo, 1, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tinyblas_zhemm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        int32_t m, int32_t n, double complex alpha,
        const double complex *restrict a, int32_t lda,
        const double complex *restrict b, int32_t ldb,
        double complex beta, double complex *restrict c, int32_t ldc)
{
    symm_impl_z(side, uplo, 1, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

/*
 *  Symmetric rank-k update: ssyrk, dsyrk
 */
void
tinyblas_ssyrk(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, float alpha,
        const float *restrict a, int32_t lda,
        float beta, float *restrict c, int32_t ldc)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t step = notrans ? (ptrdiff_t)lda : 1;

    if (n <= 0) return;

    assert(c);

    scale_tri_s(uplo, n, beta, c, ldc);

    if (k <= 0 || alpha == 0.0f) return;

    assert(a);

    rk_blocked_s(uplo, n, k, alpha, a, lda, step, a, lda, step,
                 notrans ? TINYBLAS_NONE : TINYBLAS_TRANS,
                 notrans ? TINYBLAS_TRANS : TINYBLAS_NONE,
                 c, ldc);
}

void
tinyblas_dsyrk(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, double alpha,
        const double *restrict a, int32_t lda,
        double beta, double *restrict c, int32_t ldc)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t step = notrans ? (ptrdiff_t)lda : 1;

    if (n <= 0) return;

    assert(c);

    scale_tri_d(uplo, n, beta, c, ldc);

    if (k <= 0 || alpha == 0.0) return;

    assert(a);

    rk_blocked_d(uplo, n, k, alpha, a, lda, step, a, lda, step,
                 notrans ? TINYBLAS_NONE : TINYBLAS_TRANS,
                 notrans ? TINYBLAS_TRANS : TINYBLAS_NONE,
                 c, ldc);
}

/*
 *  Complex rank-k updates: csyrk, zsyrk, cherk, zherk
 *
 *  herk takes real alpha and beta, which is exactly what keeps C hermitian.
 */
void
tinyblas_csyrk(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, float complex alpha,
        const float complex *restrict a, int32_t lda,
        float complex beta, float complex *restrict c, int32_t ldc)
{
    rk_impl_c(uplo, trans, 0, 0, n, k, alpha, a, lda, a, lda, beta, c, ldc);
}

void
tinyblas_zsyrk(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, double complex alpha,
        const double complex *restrict a, int32_t lda,
        double complex beta, double complex *restrict c, int32_t ldc)
{
    rk_impl_z(uplo, trans, 0, 0, n, k, alpha, a, lda, a, lda, beta, c, ldc);
}

void
tinyblas_cherk(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, float alpha,
        const float complex *restrict a, int32_t lda,
        float beta, float complex *restrict c, int32_t ldc)
{
    rk_impl_c(uplo, trans, 1, 0, n, k, alpha, a, lda, a, lda, beta, c, ldc);
}

void
tinyblas_zherk(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, double alpha,
        const double complex *restrict a, int32_t lda,
        double beta, double complex *restrict c, int32_t ldc)
{
    rk_impl_z(uplo, trans, 1, 0, n, k, alpha, a, lda, a, lda, beta, c, ldc);
}

/*
 *  Symmetric rank-2k update: ssyr2k, dsyr2k
 *
 *  Two rank-k passes, the second accumulating on top of the first.
 */
void
tinyblas_ssyr2k(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, float alpha,
        const float *restrict a, int32_t lda,
        const float *restrict b, int32_t ldb,
        float beta, float *restrict c, int32_t ldc)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t as = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t bs = notrans ? (ptrdiff_t)ldb : 1;
    enum tinyblas_op t1 = notrans ? TINYBLAS_NONE : TINYBLAS_TRANS;
    enum tinyblas_op t2 = notrans ? TINYBLAS_TRANS : TINYBLAS_NONE;

    if (n <= 0) return;

    assert(c);

    scale_tri_s(uplo, n, beta, c, ldc);

    if (k <= 0 || alpha == 0.0f) return;

    assert(a && b);

    rk_blocked_s(uplo, n, k, alpha, a, lda, as, b, ldb, bs, t1, t2, c, ldc);
    rk_blocked_s(uplo, n, k, alpha, b, ldb, bs, a, lda, as, t1, t2, c, ldc);
}

void
tinyblas_dsyr2k(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, double alpha,
        const double *restrict a, int32_t lda,
        const double *restrict b, int32_t ldb,
        double beta, double *restrict c, int32_t ldc)
{
    int notrans = (trans == TINYBLAS_NONE);
    ptrdiff_t as = notrans ? (ptrdiff_t)lda : 1;
    ptrdiff_t bs = notrans ? (ptrdiff_t)ldb : 1;
    enum tinyblas_op t1 = notrans ? TINYBLAS_NONE : TINYBLAS_TRANS;
    enum tinyblas_op t2 = notrans ? TINYBLAS_TRANS : TINYBLAS_NONE;

    if (n <= 0) return;

    assert(c);

    scale_tri_d(uplo, n, beta, c, ldc);

    if (k <= 0 || alpha == 0.0) return;

    assert(a && b);

    rk_blocked_d(uplo, n, k, alpha, a, lda, as, b, ldb, bs, t1, t2, c, ldc);
    rk_blocked_d(uplo, n, k, alpha, b, ldb, bs, a, lda, as, t1, t2, c, ldc);
}

/*
 *  Complex rank-2k updates: csyr2k, zsyr2k, cher2k, zher2k
 *
 *  her2k conjugates alpha on the second pass, which is what makes the two
 *  passes add up to a hermitian result.
 */
void
tinyblas_csyr2k(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, float complex alpha,
        const float complex *restrict a, int32_t lda,
        const float complex *restrict b, int32_t ldb,
        float complex beta, float complex *restrict c, int32_t ldc)
{
    rk_impl_c(uplo, trans, 0, 1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tinyblas_zsyr2k(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, double complex alpha,
        const double complex *restrict a, int32_t lda,
        const double complex *restrict b, int32_t ldb,
        double complex beta, double complex *restrict c, int32_t ldc)
{
    rk_impl_z(uplo, trans, 0, 1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tinyblas_cher2k(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, float complex alpha,
        const float complex *restrict a, int32_t lda,
        const float complex *restrict b, int32_t ldb,
        float beta, float complex *restrict c, int32_t ldc)
{
    rk_impl_c(uplo, trans, 1, 1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
tinyblas_zher2k(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        int32_t n, int32_t k, double complex alpha,
        const double complex *restrict a, int32_t lda,
        const double complex *restrict b, int32_t ldb,
        double beta, double complex *restrict c, int32_t ldc)
{
    rk_impl_z(uplo, trans, 1, 1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/*
 *  Triangular matrix-matrix product: strmm, dtrmm
 */
void
tinyblas_strmm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, float alpha,
        const float *restrict a, int32_t lda, float *restrict b, int32_t ldb)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    float *dense, *tmp;

    if (m <= 0 || n <= 0) return;

    assert(b);

    if (alpha == 0.0f) {
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                b[(ptrdiff_t)i * ldb + j] = 0.0f;

        return;
    }

    assert(a);

    dense = malloc((size_t)na * (size_t)na * sizeof(float));
    tmp   = malloc((size_t)m * (size_t)n * sizeof(float));

    if (dense == NULL || tmp == NULL) {
        free(dense);
        free(tmp);

        /* no memory: one trmv per column on the left, per row on the right.
         * The right side applies the transpose of op(A) to each row. */
        if (side == TINYBLAS_LEFT) {
            for (int32_t j = 0; j < n; ++j)
                tinyblas_strmv(uplo, trans, diag, m, a, lda, b + j, ldb);
        } else {
            enum tinyblas_op fl = (trans == TINYBLAS_NONE)
                                   ? TINYBLAS_TRANS : TINYBLAS_NONE;

            for (int32_t i = 0; i < m; ++i)
                tinyblas_strmv(uplo, fl, diag, n, a, lda,
                               b + (ptrdiff_t)i * ldb, 1);
        }

        if (alpha != 1.0f)
            for (int32_t i = 0; i < m; ++i)
                for (int32_t j = 0; j < n; ++j)
                    b[(ptrdiff_t)i * ldb + j] *= alpha;

        return;
    }

    expand_tri_s(uplo, trans, diag, na, a, lda, dense);

    if (side == TINYBLAS_LEFT)
        tinyblas_sgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, m,
                       alpha, dense, na, b, ldb, 0.0f, tmp, n);
    else
        tinyblas_sgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, n,
                       alpha, b, ldb, dense, na, 0.0f, tmp, n);

    for (int32_t i = 0; i < m; ++i)
        for (int32_t j = 0; j < n; ++j)
            b[(ptrdiff_t)i * ldb + j] = tmp[(ptrdiff_t)i * n + j];

    free(dense);
    free(tmp);
}

void
tinyblas_dtrmm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, double alpha,
        const double *restrict a, int32_t lda, double *restrict b, int32_t ldb)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    double *dense, *tmp;

    if (m <= 0 || n <= 0) return;

    assert(b);

    if (alpha == 0.0) {
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                b[(ptrdiff_t)i * ldb + j] = 0.0;

        return;
    }

    assert(a);

    dense = malloc((size_t)na * (size_t)na * sizeof(double));
    tmp   = malloc((size_t)m * (size_t)n * sizeof(double));

    if (dense == NULL || tmp == NULL) {
        free(dense);
        free(tmp);

        if (side == TINYBLAS_LEFT) {
            for (int32_t j = 0; j < n; ++j)
                tinyblas_dtrmv(uplo, trans, diag, m, a, lda, b + j, ldb);
        } else {
            enum tinyblas_op fl = (trans == TINYBLAS_NONE)
                                   ? TINYBLAS_TRANS : TINYBLAS_NONE;

            for (int32_t i = 0; i < m; ++i)
                tinyblas_dtrmv(uplo, fl, diag, n, a, lda,
                               b + (ptrdiff_t)i * ldb, 1);
        }

        if (alpha != 1.0)
            for (int32_t i = 0; i < m; ++i)
                for (int32_t j = 0; j < n; ++j)
                    b[(ptrdiff_t)i * ldb + j] *= alpha;

        return;
    }

    expand_tri_d(uplo, trans, diag, na, a, lda, dense);

    if (side == TINYBLAS_LEFT)
        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, m,
                       alpha, dense, na, b, ldb, 0.0, tmp, n);
    else
        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, n,
                       alpha, b, ldb, dense, na, 0.0, tmp, n);

    for (int32_t i = 0; i < m; ++i)
        for (int32_t j = 0; j < n; ++j)
            b[(ptrdiff_t)i * ldb + j] = tmp[(ptrdiff_t)i * n + j];

    free(dense);
    free(tmp);
}

/*
 *  Triangular matrix-matrix product: ctrmm, ztrmm
 */
void
tinyblas_ctrmm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, float complex alpha,
        const float complex *restrict a, int32_t lda,
        float complex *restrict b, int32_t ldb)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    float complex *dense, *tmp;

    if (m <= 0 || n <= 0) return;

    assert(b);

    if (alpha == 0.0f) {
        scale_mat_c(m, n, 0.0f, b, ldb);

        return;
    }

    assert(a);

    dense = malloc((size_t)na * (size_t)na * sizeof(float complex));
    tmp   = malloc((size_t)m * (size_t)n * sizeof(float complex));

    if (dense == NULL || tmp == NULL) {
        free(dense);
        free(tmp);

        tri_sweep_c(0, side, uplo, trans, diag, m, n, a, lda, b, ldb);
        scale_mat_c(m, n, alpha, b, ldb);

        return;
    }

    expand_tri_c(uplo, trans, diag, na, a, lda, dense);

    if (side == TINYBLAS_LEFT)
        tinyblas_cgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, m,
                       alpha, dense, na, b, ldb, 0.0f, tmp, n);
    else
        tinyblas_cgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, n,
                       alpha, b, ldb, dense, na, 0.0f, tmp, n);

    for (int32_t i = 0; i < m; ++i)
        for (int32_t j = 0; j < n; ++j)
            b[(ptrdiff_t)i * ldb + j] = tmp[(ptrdiff_t)i * n + j];

    free(dense);
    free(tmp);
}

void
tinyblas_ztrmm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, double complex alpha,
        const double complex *restrict a, int32_t lda,
        double complex *restrict b, int32_t ldb)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    double complex *dense, *tmp;

    if (m <= 0 || n <= 0) return;

    assert(b);

    if (alpha == 0.0) {
        scale_mat_z(m, n, 0.0, b, ldb);

        return;
    }

    assert(a);

    dense = malloc((size_t)na * (size_t)na * sizeof(double complex));
    tmp   = malloc((size_t)m * (size_t)n * sizeof(double complex));

    if (dense == NULL || tmp == NULL) {
        free(dense);
        free(tmp);

        tri_sweep_z(0, side, uplo, trans, diag, m, n, a, lda, b, ldb);
        scale_mat_z(m, n, alpha, b, ldb);

        return;
    }

    expand_tri_z(uplo, trans, diag, na, a, lda, dense);

    if (side == TINYBLAS_LEFT)
        tinyblas_zgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, m,
                       alpha, dense, na, b, ldb, 0.0, tmp, n);
    else
        tinyblas_zgemm(TINYBLAS_NONE, TINYBLAS_NONE, m, n, n,
                       alpha, b, ldb, dense, na, 0.0, tmp, n);

    for (int32_t i = 0; i < m; ++i)
        for (int32_t j = 0; j < n; ++j)
            b[(ptrdiff_t)i * ldb + j] = tmp[(ptrdiff_t)i * n + j];

    free(dense);
    free(tmp);
}

/*
 *  X <- op(A)^-1 * B for side LEFT, blocked
 *
 *  The bulk of the work is one gemm per row block against the rows already
 *  solved, which runs at gemm speed. What is left is a TRSM_MB square
 *  diagonal block, expanded dense and solved one row at a time: in row-major
 *  B an elimination step is a contiguous axpy across every right-hand side at
 *  once. That is the whole reason this beats a trsv per column -- trsv walks a
 *  column of B, which in row-major is a strided access that never vectorises.
 *
 *  op(A) is lower triangular when uplo and trans agree, and the block sweep
 *  runs forwards then; otherwise it is upper and the sweep runs backwards.
 *
 *  ponytail: side RIGHT still goes through the trsv sweep. The same blocking
 *  works there, but B's columns are the strided direction in row-major, so it
 *  wants a transposed scratch copy to be worth writing. Only if a profile asks.
 */
static void
trsm_left_s(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t m, int32_t n,
        const float *restrict a, int32_t lda,
        float *restrict b, int32_t ldb)
{
    int lower = (uplo == TINYBLAS_LOWER) == (trans == TINYBLAS_NONE);
    int32_t nblk = (m + TRSM_MB - 1) / TRSM_MB;
    float dblk[TRSM_MB * TRSM_MB];

    for (int32_t t = 0; t < nblk; ++t) {
        int32_t ib = (lower ? t : nblk - 1 - t) * TRSM_MB;
        int32_t mb = (m - ib < TRSM_MB) ? m - ib : TRSM_MB;
        int32_t js = lower ? 0 : ib + mb;           /* first solved row */
        int32_t kk = lower ? ib : m - (ib + mb);    /* how many are solved */

        /* every row already solved, subtracted in one gemm */
        if (kk > 0) {
            const float *ao = (trans == TINYBLAS_NONE)
                           ? a + (ptrdiff_t)ib * lda + js
                           : a + (ptrdiff_t)js * lda + ib;

            tinyblas_sgemm(trans, TINYBLAS_NONE, mb, n, kk, -1.0f,
                           ao, lda, b + (ptrdiff_t)js * ldb, ldb,
                           1.0f, b + (ptrdiff_t)ib * ldb, ldb);
        }

        /* what is left is one small dense triangle */
        expand_tri_s(uplo, trans, diag, mb,
                a + (ptrdiff_t)ib * lda + ib, lda, dblk);

        for (int32_t s = 0; s < mb; ++s) {
            int32_t i = lower ? s : mb - 1 - s;
            float *row = b + (ptrdiff_t)(ib + i) * ldb;
            float dii = dblk[(ptrdiff_t)i * mb + i];

            for (int32_t u = 0; u < s; ++u) {
                int32_t pp = lower ? u : mb - 1 - u;
                float d = dblk[(ptrdiff_t)i * mb + pp];
                const float *rp = b + (ptrdiff_t)(ib + pp) * ldb;

                for (int32_t j = 0; j < n; ++j) row[j] -= d * rp[j];
            }

            /* expand_tri already substituted 1 for a unit diagonal */
            if (dii != 1.0f)
                for (int32_t j = 0; j < n; ++j) row[j] /= dii;
        }
    }
}

static void
trsm_left_d(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t m, int32_t n,
        const double *restrict a, int32_t lda,
        double *restrict b, int32_t ldb)
{
    int lower = (uplo == TINYBLAS_LOWER) == (trans == TINYBLAS_NONE);
    int32_t nblk = (m + TRSM_MB - 1) / TRSM_MB;
    double dblk[TRSM_MB * TRSM_MB];

    for (int32_t t = 0; t < nblk; ++t) {
        int32_t ib = (lower ? t : nblk - 1 - t) * TRSM_MB;
        int32_t mb = (m - ib < TRSM_MB) ? m - ib : TRSM_MB;
        int32_t js = lower ? 0 : ib + mb;           /* first solved row */
        int32_t kk = lower ? ib : m - (ib + mb);    /* how many are solved */

        /* every row already solved, subtracted in one gemm */
        if (kk > 0) {
            const double *ao = (trans == TINYBLAS_NONE)
                           ? a + (ptrdiff_t)ib * lda + js
                           : a + (ptrdiff_t)js * lda + ib;

            tinyblas_dgemm(trans, TINYBLAS_NONE, mb, n, kk, -1.0,
                           ao, lda, b + (ptrdiff_t)js * ldb, ldb,
                           1.0, b + (ptrdiff_t)ib * ldb, ldb);
        }

        /* what is left is one small dense triangle */
        expand_tri_d(uplo, trans, diag, mb,
                a + (ptrdiff_t)ib * lda + ib, lda, dblk);

        for (int32_t s = 0; s < mb; ++s) {
            int32_t i = lower ? s : mb - 1 - s;
            double *row = b + (ptrdiff_t)(ib + i) * ldb;
            double dii = dblk[(ptrdiff_t)i * mb + i];

            for (int32_t u = 0; u < s; ++u) {
                int32_t pp = lower ? u : mb - 1 - u;
                double d = dblk[(ptrdiff_t)i * mb + pp];
                const double *rp = b + (ptrdiff_t)(ib + pp) * ldb;

                for (int32_t j = 0; j < n; ++j) row[j] -= d * rp[j];
            }

            /* expand_tri already substituted 1 for a unit diagonal */
            if (dii != 1.0)
                for (int32_t j = 0; j < n; ++j) row[j] /= dii;
        }
    }
}

static void
trsm_left_c(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t m, int32_t n,
        const float complex *restrict a, int32_t lda,
        float complex *restrict b, int32_t ldb)
{
    int lower = (uplo == TINYBLAS_LOWER) == (trans == TINYBLAS_NONE);
    int32_t nblk = (m + TRSM_MB - 1) / TRSM_MB;
    float complex dblk[TRSM_MB * TRSM_MB];

    for (int32_t t = 0; t < nblk; ++t) {
        int32_t ib = (lower ? t : nblk - 1 - t) * TRSM_MB;
        int32_t mb = (m - ib < TRSM_MB) ? m - ib : TRSM_MB;
        int32_t js = lower ? 0 : ib + mb;           /* first solved row */
        int32_t kk = lower ? ib : m - (ib + mb);    /* how many are solved */

        if (kk > 0) {
            const float complex *ao = (trans == TINYBLAS_NONE)
                           ? a + (ptrdiff_t)ib * lda + js
                           : a + (ptrdiff_t)js * lda + ib;

            tinyblas_cgemm(trans, TINYBLAS_NONE, mb, n, kk, -1.0f,
                           ao, lda, b + (ptrdiff_t)js * ldb, ldb,
                           1.0f, b + (ptrdiff_t)ib * ldb, ldb);
        }

        expand_tri_c(uplo, trans, diag, mb,
                a + (ptrdiff_t)ib * lda + ib, lda, dblk);

        for (int32_t s = 0; s < mb; ++s) {
            int32_t i = lower ? s : mb - 1 - s;
            float complex *row = b + (ptrdiff_t)(ib + i) * ldb;
            float complex dii = dblk[(ptrdiff_t)i * mb + i];

            for (int32_t u = 0; u < s; ++u) {
                int32_t pp = lower ? u : mb - 1 - u;
                float complex d = dblk[(ptrdiff_t)i * mb + pp];
                const float complex *rp = b + (ptrdiff_t)(ib + pp) * ldb;
                float dr = crealf(d), di = cimagf(d);

                for (int32_t j = 0; j < n; ++j) {
                    float zr = crealf(rp[j]), zi = cimagf(rp[j]);
                    float wr = crealf(row[j]), wi = cimagf(row[j]);

                    row[j] = (wr - (dr * zr - di * zi))
                           + (wi - (dr * zi + di * zr)) * I;
                }
            }

            /* expand_tri already substituted 1 for a unit diagonal */
            if (dii != 1.0f) {
                float dr = crealf(dii), di = cimagf(dii);
                float den = dr * dr + di * di;
                float ir = dr / den, ii = -di / den;

                /* one reciprocal per row rather than a __divsc3 call per
                 * element; A is assumed nonsingular either way */
                for (int32_t j = 0; j < n; ++j) {
                    float wr = crealf(row[j]), wi = cimagf(row[j]);

                    row[j] = (wr * ir - wi * ii) + (wr * ii + wi * ir) * I;
                }
            }
        }
    }
}

static void
trsm_left_z(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t m, int32_t n,
        const double complex *restrict a, int32_t lda,
        double complex *restrict b, int32_t ldb)
{
    int lower = (uplo == TINYBLAS_LOWER) == (trans == TINYBLAS_NONE);
    int32_t nblk = (m + TRSM_MB - 1) / TRSM_MB;
    double complex dblk[TRSM_MB * TRSM_MB];

    for (int32_t t = 0; t < nblk; ++t) {
        int32_t ib = (lower ? t : nblk - 1 - t) * TRSM_MB;
        int32_t mb = (m - ib < TRSM_MB) ? m - ib : TRSM_MB;
        int32_t js = lower ? 0 : ib + mb;           /* first solved row */
        int32_t kk = lower ? ib : m - (ib + mb);    /* how many are solved */

        if (kk > 0) {
            const double complex *ao = (trans == TINYBLAS_NONE)
                           ? a + (ptrdiff_t)ib * lda + js
                           : a + (ptrdiff_t)js * lda + ib;

            tinyblas_zgemm(trans, TINYBLAS_NONE, mb, n, kk, -1.0,
                           ao, lda, b + (ptrdiff_t)js * ldb, ldb,
                           1.0, b + (ptrdiff_t)ib * ldb, ldb);
        }

        expand_tri_z(uplo, trans, diag, mb,
                a + (ptrdiff_t)ib * lda + ib, lda, dblk);

        for (int32_t s = 0; s < mb; ++s) {
            int32_t i = lower ? s : mb - 1 - s;
            double complex *row = b + (ptrdiff_t)(ib + i) * ldb;
            double complex dii = dblk[(ptrdiff_t)i * mb + i];

            for (int32_t u = 0; u < s; ++u) {
                int32_t pp = lower ? u : mb - 1 - u;
                double complex d = dblk[(ptrdiff_t)i * mb + pp];
                const double complex *rp = b + (ptrdiff_t)(ib + pp) * ldb;
                double dr = creal(d), di = cimag(d);

                for (int32_t j = 0; j < n; ++j) {
                    double zr = creal(rp[j]), zi = cimag(rp[j]);
                    double wr = creal(row[j]), wi = cimag(row[j]);

                    row[j] = (wr - (dr * zr - di * zi))
                           + (wi - (dr * zi + di * zr)) * I;
                }
            }

            /* expand_tri already substituted 1 for a unit diagonal */
            if (dii != 1.0) {
                double dr = creal(dii), di = cimag(dii);
                double den = dr * dr + di * di;
                double ir = dr / den, ii = -di / den;

                /* one reciprocal per row rather than a __divdc3 call per
                 * element; A is assumed nonsingular either way */
                for (int32_t j = 0; j < n; ++j) {
                    double wr = creal(row[j]), wi = cimag(row[j]);

                    row[j] = (wr * ir - wi * ii) + (wr * ii + wi * ir) * I;
                }
            }
        }
    }
}

/*
 *  Triangular solve with multiple right-hand sides: strsm, dtrsm
 *
 *  side LEFT is the blocked solver above. side RIGHT is one trsv per row:
 *  solving X * op(A) = alpha * B from the right is the same as
 *  op(A)^T x = alpha * b for each row of B, which is why the trans flag flips.
 */
void
tinyblas_strsm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, float alpha,
        const float *restrict a, int32_t lda, float *restrict b, int32_t ldb)
{
    if (m <= 0 || n <= 0) return;

    assert(b);

    if (alpha == 0.0f) {
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                b[(ptrdiff_t)i * ldb + j] = 0.0f;

        return;
    }

    assert(a);

    if (alpha != 1.0f)
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                b[(ptrdiff_t)i * ldb + j] *= alpha;

    if (side == TINYBLAS_LEFT) {
        trsm_left_s(uplo, trans, diag, m, n, a, lda, b, ldb);

        return;
    }

    {
        enum tinyblas_op fl = (trans == TINYBLAS_NONE)
                               ? TINYBLAS_TRANS : TINYBLAS_NONE;

        for (int32_t i = 0; i < m; ++i)
            tinyblas_strsv(uplo, fl, diag, n, a, lda,
                           b + (ptrdiff_t)i * ldb, 1);
    }
}

void
tinyblas_dtrsm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, double alpha,
        const double *restrict a, int32_t lda, double *restrict b, int32_t ldb)
{
    if (m <= 0 || n <= 0) return;

    assert(b);

    if (alpha == 0.0) {
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                b[(ptrdiff_t)i * ldb + j] = 0.0;

        return;
    }

    assert(a);

    if (alpha != 1.0)
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                b[(ptrdiff_t)i * ldb + j] *= alpha;

    if (side == TINYBLAS_LEFT) {
        trsm_left_d(uplo, trans, diag, m, n, a, lda, b, ldb);

        return;
    }

    {
        enum tinyblas_op fl = (trans == TINYBLAS_NONE)
                               ? TINYBLAS_TRANS : TINYBLAS_NONE;

        for (int32_t i = 0; i < m; ++i)
            tinyblas_dtrsv(uplo, fl, diag, n, a, lda,
                           b + (ptrdiff_t)i * ldb, 1);
    }
}

/*
 *  Triangular solve with multiple right-hand sides: ctrsm, ztrsm
 *
 *  side LEFT goes through the blocked solver above; side RIGHT stays on the
 *  trsv sweep, which carries the transpose bookkeeping.
 */
void
tinyblas_ctrsm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, float complex alpha,
        const float complex *restrict a, int32_t lda,
        float complex *restrict b, int32_t ldb)
{
    if (m <= 0 || n <= 0) return;

    assert(b);

    if (alpha == 0.0f) {
        scale_mat_c(m, n, 0.0f, b, ldb);

        return;
    }

    assert(a);

    scale_mat_c(m, n, alpha, b, ldb);

    if (side == TINYBLAS_LEFT)
        trsm_left_c(uplo, trans, diag, m, n, a, lda, b, ldb);
    else
        tri_sweep_c(1, side, uplo, trans, diag, m, n, a, lda, b, ldb);
}

void
tinyblas_ztrsm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, double complex alpha,
        const double complex *restrict a, int32_t lda,
        double complex *restrict b, int32_t ldb)
{
    if (m <= 0 || n <= 0) return;

    assert(b);

    if (alpha == 0.0) {
        scale_mat_z(m, n, 0.0, b, ldb);

        return;
    }

    assert(a);

    scale_mat_z(m, n, alpha, b, ldb);

    if (side == TINYBLAS_LEFT)
        trsm_left_z(uplo, trans, diag, m, n, a, lda, b, ldb);
    else
        tri_sweep_z(1, side, uplo, trans, diag, m, n, a, lda, b, ldb);
}
