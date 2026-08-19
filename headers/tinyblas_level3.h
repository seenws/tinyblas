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

#ifndef TINYBLAS_LEVEL_3_H_
#define TINYBLAS_LEVEL_3_H_

#include <complex.h>
#include <stdint.h>

#include "tinyblas_common.h"

/* C is always m by n. op(A) is m by k and op(B) is k by n, so the *stored*
 * shapes depend on the trans flags:
 *   transa == NO_TRANS -> A is m by k, lda >= k;  else A is k by m, lda >= m
 *   transb == NO_TRANS -> B is k by n, ldb >= n;  else B is n by k, ldb >= k
 * ldc >= n always.
 */

/* C <- alpha * op(A) * op(B) + beta * C */
void tinyblas_sgemm(enum tinyblas_trans transa, enum tinyblas_trans transb,
                    int32_t m, int32_t n, int32_t k,
                    float alpha,
                    const float *restrict a, int32_t lda,
                    const float *restrict b, int32_t ldb,
                    float beta,
                    float *restrict c, int32_t ldc);

void tinyblas_dgemm(enum tinyblas_trans transa, enum tinyblas_trans transb,
                    int32_t m, int32_t n, int32_t k,
                    double alpha,
                    const double *restrict a, int32_t lda,
                    const double *restrict b, int32_t ldb,
                    double beta,
                    double *restrict c, int32_t ldc);

void tinyblas_cgemm(enum tinyblas_trans transa, enum tinyblas_trans transb,
                    int32_t m, int32_t n, int32_t k,
                    float complex alpha,
                    const float complex *restrict a, int32_t lda,
                    const float complex *restrict b, int32_t ldb,
                    float complex beta,
                    float complex *restrict c, int32_t ldc);

void tinyblas_zgemm(enum tinyblas_trans transa, enum tinyblas_trans transb,
                    int32_t m, int32_t n, int32_t k,
                    double complex alpha,
                    const double complex *restrict a, int32_t lda,
                    const double complex *restrict b, int32_t ldb,
                    double complex beta,
                    double complex *restrict c, int32_t ldc);

/* C <- alpha * A * B + beta * C with A symmetric, or the mirror image with
 * side = RIGHT. A is m by m on the left and n by n on the right. */
void tinyblas_ssymm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    int32_t m, int32_t n, float alpha,
                    const float *restrict a, int32_t lda,
                    const float *restrict b, int32_t ldb,
                    float beta, float *restrict c, int32_t ldc);

void tinyblas_dsymm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    int32_t m, int32_t n, double alpha,
                    const double *restrict a, int32_t lda,
                    const double *restrict b, int32_t ldb,
                    double beta, double *restrict c, int32_t ldc);

void tinyblas_csymm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    int32_t m, int32_t n, float complex alpha,
                    const float complex *restrict a, int32_t lda,
                    const float complex *restrict b, int32_t ldb,
                    float complex beta, float complex *restrict c, int32_t ldc);

void tinyblas_zsymm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    int32_t m, int32_t n, double complex alpha,
                    const double complex *restrict a, int32_t lda,
                    const double complex *restrict b, int32_t ldb,
                    double complex beta, double complex *restrict c, int32_t ldc);

/* as symm, with A hermitian */
void tinyblas_chemm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    int32_t m, int32_t n, float complex alpha,
                    const float complex *restrict a, int32_t lda,
                    const float complex *restrict b, int32_t ldb,
                    float complex beta, float complex *restrict c, int32_t ldc);

void tinyblas_zhemm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    int32_t m, int32_t n, double complex alpha,
                    const double complex *restrict a, int32_t lda,
                    const double complex *restrict b, int32_t ldb,
                    double complex beta, double complex *restrict c, int32_t ldc);

/* C <- alpha * A * A^T + beta * C, only the uplo triangle of C is referenced.
 * trans = NO_TRANS makes A n by k; otherwise A is k by n and it is A^T * A. */
void tinyblas_ssyrk(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                    int32_t n, int32_t k, float alpha,
                    const float *restrict a, int32_t lda,
                    float beta, float *restrict c, int32_t ldc);

void tinyblas_dsyrk(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                    int32_t n, int32_t k, double alpha,
                    const double *restrict a, int32_t lda,
                    double beta, double *restrict c, int32_t ldc);

void tinyblas_csyrk(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                    int32_t n, int32_t k, float complex alpha,
                    const float complex *restrict a, int32_t lda,
                    float complex beta, float complex *restrict c, int32_t ldc);

void tinyblas_zsyrk(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                    int32_t n, int32_t k, double complex alpha,
                    const double complex *restrict a, int32_t lda,
                    double complex beta, double complex *restrict c, int32_t ldc);

/* C <- alpha * A * A^H + beta * C, hermitian. alpha and beta are real, which
 * is what keeps C hermitian. */
void tinyblas_cherk(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                    int32_t n, int32_t k, float alpha,
                    const float complex *restrict a, int32_t lda,
                    float beta, float complex *restrict c, int32_t ldc);

void tinyblas_zherk(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                    int32_t n, int32_t k, double alpha,
                    const double complex *restrict a, int32_t lda,
                    double beta, double complex *restrict c, int32_t ldc);

/* C <- alpha * A * B^T + alpha * B * A^T + beta * C */
void tinyblas_ssyr2k(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                     int32_t n, int32_t k, float alpha,
                     const float *restrict a, int32_t lda,
                     const float *restrict b, int32_t ldb,
                     float beta, float *restrict c, int32_t ldc);

void tinyblas_dsyr2k(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                     int32_t n, int32_t k, double alpha,
                     const double *restrict a, int32_t lda,
                     const double *restrict b, int32_t ldb,
                     double beta, double *restrict c, int32_t ldc);

void tinyblas_csyr2k(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                     int32_t n, int32_t k, float complex alpha,
                     const float complex *restrict a, int32_t lda,
                     const float complex *restrict b, int32_t ldb,
                     float complex beta, float complex *restrict c, int32_t ldc);

void tinyblas_zsyr2k(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                     int32_t n, int32_t k, double complex alpha,
                     const double complex *restrict a, int32_t lda,
                     const double complex *restrict b, int32_t ldb,
                     double complex beta, double complex *restrict c, int32_t ldc);

/* C <- alpha * A * B^H + conj(alpha) * B * A^H + beta * C, beta real */
void tinyblas_cher2k(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                     int32_t n, int32_t k, float complex alpha,
                     const float complex *restrict a, int32_t lda,
                     const float complex *restrict b, int32_t ldb,
                     float beta, float complex *restrict c, int32_t ldc);

void tinyblas_zher2k(enum tinyblas_uplo uplo, enum tinyblas_trans trans,
                     int32_t n, int32_t k, double complex alpha,
                     const double complex *restrict a, int32_t lda,
                     const double complex *restrict b, int32_t ldb,
                     double beta, double complex *restrict c, int32_t ldc);

/* B <- alpha * op(A) * B, or B * op(A) with side = RIGHT, A triangular */
void tinyblas_strmm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    enum tinyblas_trans trans, enum tinyblas_diag diag,
                    int32_t m, int32_t n, float alpha,
                    const float *restrict a, int32_t lda,
                    float *restrict b, int32_t ldb);

void tinyblas_dtrmm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    enum tinyblas_trans trans, enum tinyblas_diag diag,
                    int32_t m, int32_t n, double alpha,
                    const double *restrict a, int32_t lda,
                    double *restrict b, int32_t ldb);

void tinyblas_ctrmm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    enum tinyblas_trans trans, enum tinyblas_diag diag,
                    int32_t m, int32_t n, float complex alpha,
                    const float complex *restrict a, int32_t lda,
                    float complex *restrict b, int32_t ldb);

void tinyblas_ztrmm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    enum tinyblas_trans trans, enum tinyblas_diag diag,
                    int32_t m, int32_t n, double complex alpha,
                    const double complex *restrict a, int32_t lda,
                    double complex *restrict b, int32_t ldb);

/* solve op(A) * X = alpha * B, or X * op(A) = alpha * B with side = RIGHT,
 * writing X over B. A is triangular and assumed nonsingular. */
void tinyblas_strsm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    enum tinyblas_trans trans, enum tinyblas_diag diag,
                    int32_t m, int32_t n, float alpha,
                    const float *restrict a, int32_t lda,
                    float *restrict b, int32_t ldb);

void tinyblas_dtrsm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    enum tinyblas_trans trans, enum tinyblas_diag diag,
                    int32_t m, int32_t n, double alpha,
                    const double *restrict a, int32_t lda,
                    double *restrict b, int32_t ldb);

void tinyblas_ctrsm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    enum tinyblas_trans trans, enum tinyblas_diag diag,
                    int32_t m, int32_t n, float complex alpha,
                    const float complex *restrict a, int32_t lda,
                    float complex *restrict b, int32_t ldb);

void tinyblas_ztrsm(enum tinyblas_side side, enum tinyblas_uplo uplo,
                    enum tinyblas_trans trans, enum tinyblas_diag diag,
                    int32_t m, int32_t n, double complex alpha,
                    const double complex *restrict a, int32_t lda,
                    double complex *restrict b, int32_t ldb);

#endif
