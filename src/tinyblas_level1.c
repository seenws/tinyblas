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

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "../headers/tinyblas_level1.h"
#include "../headers/tinyblas_common.h"

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif

/* Helper macros for debug mode */
#ifdef TINYBLAS_DEBUG
#define TINYBLAS_ASSERT(cond) assert(cond)
#else
#define TINYBLAS_ASSERT(cond) ((void)0)
#endif

/*
 *  Double-precision dot product: ddot
 */
double
tinyblas_ddot(int32_t n,
        const double *__restrict dx, int32_t incx,
        const double *__restrict dy, int32_t incy)
{
    if (n <= 0) return 0.0;

    TINYBLAS_ASSERT(dx != NULL && dy != NULL);

    if (incx < 0) dx += (1 - n) * incx;
    if (incy < 0) dy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        double dot = 0.0;

#if defined(__AVX2__) || defined(__AVX__)
        int32_t n_vec = n & -4;

        __m256d sum = _mm256_setzero_pd();

        for (int32_t i = 0; i < n_vec; i += 4) {
            __m256d vx = _mm256_loadu_pd(dx + i); // unaligned load is fine

            __m256d vy = _mm256_loadu_pd(dy + i);
            sum = _mm256_fmadd_pd(vx, vy, sum); // FMA if AVX2, else mul+add
        }

        // Horizontal sum
        __m256d hi = _mm256_permute2f128_pd(sum, sum, 1);
        __m256d sum2 = _mm256_add_pd(sum, hi);

        hi = _mm256_permute_pd(sum2, 1);
        sum2 = _mm256_add_pd(sum2, hi);
        dot = _mm256_cvtsd_f64(sum2);

        // Tail
        for (int32_t i = n_vec; i < n; ++i) {
            dot += dx[i] * dy[i];
        }

#else
        const int32_t unroll = 8;
        int32_t n_unroll = n - (n % unroll);

        for (int32_t i = 0; i < n_unroll; i += unroll) {
            dot += dx[i+0] * dy[i+0] + dx[i+1] * dy[i+1] +
                   dx[i+2] * dy[i+2] + dx[i+3] * dy[i+3] +
                   dx[i+4] * dy[i+4] + dx[i+5] * dy[i+5] +
                   dx[i+6] * dy[i+6] + dx[i+7] * dy[i+7];
        }

        /* Tail */
        for (int32_t i = n_unroll; i < n; ++i) {
            dot += dx[i] * dy[i];
        }
#endif

        return dot;
    }

    /* General strided case */
    double dot = 0.0;
    int32_t ix = 0;
    int32_t iy = 0;

    for (int32_t i = 0; i < n; ++i) {
        dot += dx[ix] * dy[iy];
        ix += incx;
        iy += incy;
    }

    return dot;
}

/*
 *  Single-precision dot product: sdot
 */
float
tinyblas_sdot(int32_t n,
        const float *__restrict sx, int32_t incx,
        const float *__restrict sy, int32_t incy)
{
    if (n <= 0) return 0.0f;

    TINYBLAS_ASSERT(sx != NULL && sy != NULL);

    if (incx < 0) sx += (1 - n) * incx;
    if (incy < 0) sy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        float dot = 0.0f;

#if defined(__AVX2__) || defined(__AVX__)
        int32_t n_vec = n & -8;

        __m256 sum = _mm256_setzero_ps();

        for (int32_t i = 0; i < n_vec; i += 8) {
            __m256 vx   = _mm256_loadu_ps(sx + i);
            __m256 vy   = _mm256_loadu_ps(sy + i);
            sum         = _mm256_fmadd_ps(vx, vy, sum);
        }

        /* Horizontal sum */
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 hi = _mm256_extractf128_ps(sum, 1);

        lo = _mm_add_ps(lo, hi);
        hi = _mm_movehl_ps(hi, lo);
        lo = _mm_add_ps(lo, hi);
        hi = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1,1,1,1));
        lo = _mm_add_ps(lo, hi);
        dot = _mm_cvtss_f32(lo);

        for (int32_t i = n_vec; i < n; ++i) {
            dot += sx[i] * sy[i];
        }

#else
        const int32_t unroll = 16;
        int32_t n_unroll = n - (n % unroll);

        for (int32_t i = 0; i < n_unroll; i += unroll) {
            dot += sx[i+0]  * sy[i+0]  + sx[i+1]  * sy[i+1]  +
                   sx[i+2]  * sy[i+2]  + sx[i+3]  * sy[i+3]  +
                   sx[i+4]  * sy[i+4]  + sx[i+5]  * sy[i+5]  +
                   sx[i+6]  * sy[i+6]  + sx[i+7]  * sy[i+7]  +
                   sx[i+8]  * sy[i+8]  + sx[i+9]  * sy[i+9]  +
                   sx[i+10] * sy[i+10] + sx[i+11] * sy[i+11] +
                   sx[i+12] * sy[i+12] + sx[i+13] * sy[i+13] +
                   sx[i+14] * sy[i+14] + sx[i+15] * sy[i+15];
        }

        for (int32_t i = n_unroll; i < n; ++i) {
            dot += sx[i] * sy[i];
        }
#endif

        return dot;
    }

    /* General strided case */
    float dot = 0.0f;
    int32_t ix = 0;
    int32_t iy = 0;

    for (int32_t i = 0; i < n; ++i) {
        dot += sx[ix] * sy[iy];
        ix += incx;
        iy += incy;
    }

    return dot;
}

/*
 *  Mixed-precision dot product: single-precision inputs, double-precision accumulator
 */
double
tinyblas_dsdot(int32_t n,
                      const float *__restrict dx, int32_t incx,
                      const float *__restrict dy, int32_t incy)
{
    if (n <= 0) return 0.0;

    TINYBLAS_ASSERT(dx != NULL && dy != NULL);

    if (incx < 0) dx += (1 - n) * incx;
    if (incy < 0) dy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        double dot = 0.0;

#if defined(__AVX2__) || defined(__AVX__)
        int32_t n_vec = n & -8;

        __m256d accum1 = _mm256_setzero_pd();
        __m256d accum2 = _mm256_setzero_pd();

        for (int32_t i = 0; i < n_vec; i += 8) {
            __m256 vx = _mm256_loadu_ps(dx + i);
            __m256 vy = _mm256_loadu_ps(dy + i);

            /* Convert to double and multiply in two halves */
            __m256d vx_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vx));
            __m256d vx_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vx, 1));
            __m256d vy_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vy));
            __m256d vy_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vy, 1));

            accum1 = _mm256_fmadd_pd(vx_lo, vy_lo, accum1);
            accum2 = _mm256_fmadd_pd(vx_hi, vy_hi, accum2);
        }

        /* Reduce accum1 and accum2 */
        accum1 = _mm256_add_pd(accum1, accum2);
        __m128d hi = _mm256_extractf128_pd(accum1, 1);
        __m128d lo = _mm256_castpd256_pd128(accum1);

        lo = _mm_add_pd(lo, hi);
        hi = _mm_shuffle_pd(lo, lo, 1);
        lo = _mm_add_pd(lo, hi);

        dot = _mm_cvtsd_f64(lo);

        /* Tail */
        for (int32_t i = n_vec; i < n; ++i) {
            dot += (double)dx[i] * (double)dy[i];
        }

#else
        const int32_t unroll = 16;
        int32_t n_unroll = n - (n % unroll);

        for (int32_t i = 0; i < n_unroll; i += unroll) {
            dot += (double)dx[i+0]  * dy[i+0]  + (double)dx[i+1]  * dy[i+1]  +
                   (double)dx[i+2]  * dy[i+2]  + (double)dx[i+3]  * dy[i+3]  +
                   (double)dx[i+4]  * dy[i+4]  + (double)dx[i+5]  * dy[i+5]  +
                   (double)dx[i+6]  * dy[i+6]  + (double)dx[i+7]  * dy[i+7]  +
                   (double)dx[i+8]  * dy[i+8]  + (double)dx[i+9]  * dy[i+9]  +
                   (double)dx[i+10] * dy[i+10] + (double)dx[i+11] * dy[i+11] +
                   (double)dx[i+12] * dy[i+12] + (double)dx[i+13] * dy[i+13] +
                   (double)dx[i+14] * dy[i+14] + (double)dx[i+15] * dy[i+15];
        }

        for (int32_t i = n_unroll; i < n; ++i) {
            dot += (double)dx[i] * dy[i];
        }
#endif

        return dot;
    }

    /* General strided case */
    double dot = 0.0;
    int32_t ix = 0;
    int32_t iy = 0;

    for (int32_t i = 0; i < n; ++i) {
        dot += (double)dx[ix] * dy[iy];
        ix += incx;
        iy += incy;
    }

    return dot;
}

/*
 *  Complex single-precision unconjugated dot product: cdotu
 */
float complex
tinyblas_cdotu(int32_t n,
        const float complex *__restrict zx, int32_t incx,
        const float complex *__restrict zy, int32_t incy)
{
    if (n <= 0) return 0.0f + 0.0f * I;

    TINYBLAS_ASSERT(zx != NULL && zy != NULL);

    if (incx < 0) zx += (1 - n) * incx;
    if (incy < 0) zy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        float complex dot = 0.0f + 0.0f * I;

#if defined(__AVX2__) || defined(__AVX__)
        int32_t n_vec = n & -4;
        __m256 sum_re = _mm256_setzero_ps();
        __m256 sum_im = _mm256_setzero_ps();

        for (int32_t i = 0; i < n_vec; n += 4) {
            __m256 x = _mm256_loadu_ps((const float*)(zx + i));
            __m256 y = _mm256_loadu_ps((const float*)(zy + i));

            /* x_re = [re0, re1, re2, re3, re0, re1, re2, re3] */
            __m256 x_re = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(2,0,2,0));
            __m256 x_im = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(3,1,3,1));
            __m256 y_re = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(2,0,2,0));
            __m256 y_im = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(3,1,3,1));

            /* Real part: x_re * y_re - x_im * y_im */
            __m256 re = _mm256_fmsub_ps(x_re, y_re, _mm256_mul_ps(x_im, y_im));
            /* Imag part: x_re * y_im + x_im * y_re */
            __m256 im = _mm256_fmadd_ps(x_re, y_im, _mm256_mul_ps(x_im, y_re));

            sum_re = _mm256_add_ps(sum_re, re);
            sum_im = _mm256_add_ps(sum_im, im);
        }

        /* Horizontal sum */
        float re = 0.0f, im = 0.0f;

        __m128 lo_re = _mm256_castps256_ps128(sum_re);
        __m128 hi_re = _mm256_extractf128_ps(sum_re, 1);

        lo_re = _mm_add_ps(lo_re, hi_re);
        lo_re = _mm_hadd_ps(lo_re, lo_re);
        lo_re = _mm_hadd_ps(lo_re, lo_re);

        re = _mm_cvtss_f32(lo_re);

        __m128 lo_im = _mm256_castps256_ps128(sum_im);
        __m128 hi_im = _mm256_extractf128_ps(sum_im, 1);

        lo_im = _mm_add_ps(lo_im, hi_im);
        lo_im = _mm_hadd_ps(lo_im, lo_im);
        lo_im = _mm_hadd_ps(lo_im, lo_im);

        im = _mm_cvtss_f32(lo_im);

        dot = re + im * I;

        /* Tail */
        for (int32_t i = n_vec; i < n; ++i) {
            dot += zx[i] * zy[i];
        }

#else
        const int32_t unroll = 8;
        int32_t n_unroll = n - (n % unroll);

        for (int32_t i = 0; i < n_unroll; i += unroll) {
            dot += zx[i+0] * zy[i+0] + zx[i+1] * zy[i+1] +
                   zx[i+2] * zy[i+2] + zx[i+3] * zy[i+3] +
                   zx[i+4] * zy[i+4] + zx[i+5] * zy[i+5] +
                   zx[i+6] * zy[i+6] + zx[i+7] * zy[i+7];
        }

        for (int32_t i = n_unroll; i < n; ++i) {
            dot += zx[i] * zy[i];
        }
#endif

        return dot;
    }

    /* General strided case */
    float complex dot = 0.0f + 0.0f * I;
    int32_t ix = 0;
    int32_t iy = 0;

    for (int32_t i = 0; i < n; ++i) {
        dot += zx[ix] * zy[iy];
        ix += incx;
        iy += incy;
    }

    return dot;
}
