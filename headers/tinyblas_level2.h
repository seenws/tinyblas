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

#ifndef TINYBLAS_LEVEL_2_H_
#define TINYBLAS_LEVEL_2_H_

#include <complex.h>
#include <stdint.h>

#include "tinyblas_common.h"

/* Matrices are row-major with lda >= their column count. Vectors carry the
 * level 1 increment convention: any increment is allowed and a negative one
 * walks backwards.
 *
 * The symmetric, hermitian and triangular forms read only the triangle named
 * by uplo; the other half is never touched and may hold anything. The
 * hermitian forms take the diagonal to be real and ignore its stored
 * imaginary part rather than trusting it.
 */

/* y <- alpha * op(A) * x + beta * y, A is m by n */
void tinyblas_sgemv(enum tinyblas_op trans, int32_t m, int32_t n, float alpha,
                    const float *restrict a, int32_t lda,
                    const float *restrict x, int32_t incx,
                    float beta, float *restrict y, int32_t incy);

void tinyblas_dgemv(enum tinyblas_op trans, int32_t m, int32_t n, double alpha,
                    const double *restrict a, int32_t lda,
                    const double *restrict x, int32_t incx,
                    double beta, double *restrict y, int32_t incy);

void tinyblas_cgemv(enum tinyblas_op trans, int32_t m, int32_t n,
                    float complex alpha,
                    const float complex *restrict a, int32_t lda,
                    const float complex *restrict x, int32_t incx,
                    float complex beta, float complex *restrict y, int32_t incy);

void tinyblas_zgemv(enum tinyblas_op trans, int32_t m, int32_t n,
                    double complex alpha,
                    const double complex *restrict a, int32_t lda,
                    const double complex *restrict x, int32_t incx,
                    double complex beta, double complex *restrict y, int32_t incy);

/* y <- alpha * A * x + beta * y, A symmetric */
void tinyblas_ssymv(enum tinyblas_uplo uplo, int32_t n, float alpha,
                    const float *restrict a, int32_t lda,
                    const float *restrict x, int32_t incx,
                    float beta, float *restrict y, int32_t incy);

void tinyblas_dsymv(enum tinyblas_uplo uplo, int32_t n, double alpha,
                    const double *restrict a, int32_t lda,
                    const double *restrict x, int32_t incx,
                    double beta, double *restrict y, int32_t incy);

/* y <- alpha * A * x + beta * y, A hermitian */
void tinyblas_chemv(enum tinyblas_uplo uplo, int32_t n, float complex alpha,
                    const float complex *restrict a, int32_t lda,
                    const float complex *restrict x, int32_t incx,
                    float complex beta, float complex *restrict y, int32_t incy);

void tinyblas_zhemv(enum tinyblas_uplo uplo, int32_t n, double complex alpha,
                    const double complex *restrict a, int32_t lda,
                    const double complex *restrict x, int32_t incx,
                    double complex beta, double complex *restrict y, int32_t incy);

/* x <- op(A) * x, A triangular */
void tinyblas_strmv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
                    enum tinyblas_diag diag, int32_t n,
                    const float *restrict a, int32_t lda,
                    float *restrict x, int32_t incx);

void tinyblas_dtrmv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
                    enum tinyblas_diag diag, int32_t n,
                    const double *restrict a, int32_t lda,
                    double *restrict x, int32_t incx);

void tinyblas_ctrmv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
                    enum tinyblas_diag diag, int32_t n,
                    const float complex *restrict a, int32_t lda,
                    float complex *restrict x, int32_t incx);

void tinyblas_ztrmv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
                    enum tinyblas_diag diag, int32_t n,
                    const double complex *restrict a, int32_t lda,
                    double complex *restrict x, int32_t incx);

/* solve op(A) * x = b in place, A triangular and assumed nonsingular */
void tinyblas_strsv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
                    enum tinyblas_diag diag, int32_t n,
                    const float *restrict a, int32_t lda,
                    float *restrict x, int32_t incx);

void tinyblas_dtrsv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
                    enum tinyblas_diag diag, int32_t n,
                    const double *restrict a, int32_t lda,
                    double *restrict x, int32_t incx);

void tinyblas_ctrsv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
                    enum tinyblas_diag diag, int32_t n,
                    const float complex *restrict a, int32_t lda,
                    float complex *restrict x, int32_t incx);

void tinyblas_ztrsv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
                    enum tinyblas_diag diag, int32_t n,
                    const double complex *restrict a, int32_t lda,
                    double complex *restrict x, int32_t incx);

/* A <- alpha * x * y^T + A */
void tinyblas_sger(int32_t m, int32_t n, float alpha,
                   const float *restrict x, int32_t incx,
                   const float *restrict y, int32_t incy,
                   float *restrict a, int32_t lda);

void tinyblas_dger(int32_t m, int32_t n, double alpha,
                   const double *restrict x, int32_t incx,
                   const double *restrict y, int32_t incy,
                   double *restrict a, int32_t lda);

/* A <- alpha * x * y^T + A, unconjugated */
void tinyblas_cgeru(int32_t m, int32_t n, float complex alpha,
                    const float complex *restrict x, int32_t incx,
                    const float complex *restrict y, int32_t incy,
                    float complex *restrict a, int32_t lda);

void tinyblas_zgeru(int32_t m, int32_t n, double complex alpha,
                    const double complex *restrict x, int32_t incx,
                    const double complex *restrict y, int32_t incy,
                    double complex *restrict a, int32_t lda);

/* A <- alpha * x * conj(y)^T + A */
void tinyblas_cgerc(int32_t m, int32_t n, float complex alpha,
                    const float complex *restrict x, int32_t incx,
                    const float complex *restrict y, int32_t incy,
                    float complex *restrict a, int32_t lda);

void tinyblas_zgerc(int32_t m, int32_t n, double complex alpha,
                    const double complex *restrict x, int32_t incx,
                    const double complex *restrict y, int32_t incy,
                    double complex *restrict a, int32_t lda);

/* A <- alpha * x * x^T + A, A symmetric */
void tinyblas_ssyr(enum tinyblas_uplo uplo, int32_t n, float alpha,
                   const float *restrict x, int32_t incx,
                   float *restrict a, int32_t lda);

void tinyblas_dsyr(enum tinyblas_uplo uplo, int32_t n, double alpha,
                   const double *restrict x, int32_t incx,
                   double *restrict a, int32_t lda);

/* A <- alpha * x * conj(x)^T + A, A hermitian, alpha real */
void tinyblas_cher(enum tinyblas_uplo uplo, int32_t n, float alpha,
                   const float complex *restrict x, int32_t incx,
                   float complex *restrict a, int32_t lda);

void tinyblas_zher(enum tinyblas_uplo uplo, int32_t n, double alpha,
                   const double complex *restrict x, int32_t incx,
                   double complex *restrict a, int32_t lda);

/* A <- alpha * x * y^T + alpha * y * x^T + A, A symmetric */
void tinyblas_ssyr2(enum tinyblas_uplo uplo, int32_t n, float alpha,
                    const float *restrict x, int32_t incx,
                    const float *restrict y, int32_t incy,
                    float *restrict a, int32_t lda);

void tinyblas_dsyr2(enum tinyblas_uplo uplo, int32_t n, double alpha,
                    const double *restrict x, int32_t incx,
                    const double *restrict y, int32_t incy,
                    double *restrict a, int32_t lda);

/* A <- alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A, A hermitian */
void tinyblas_cher2(enum tinyblas_uplo uplo, int32_t n, float complex alpha,
                    const float complex *restrict x, int32_t incx,
                    const float complex *restrict y, int32_t incy,
                    float complex *restrict a, int32_t lda);

void tinyblas_zher2(enum tinyblas_uplo uplo, int32_t n, double complex alpha,
                    const double complex *restrict x, int32_t incx,
                    const double complex *restrict y, int32_t incy,
                    double complex *restrict a, int32_t lda);

#endif
