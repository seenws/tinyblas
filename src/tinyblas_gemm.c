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

#include "tinyblas_level3.h"
#include "tinyblas_common.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/* Every routine here reaches its operands through a stride pair (rs, cs) rather
 * than a transpose flag: NoTrans passes (ld, 1) and Trans passes (1, ld). One
 * loop then covers both, and the same trick later collapses the side/uplo/trans
 * explosion in trmm and trsm.
 *
 * Index arithmetic is ptrdiff_t on purpose. int32_t * int32_t overflows at
 * roughly 46000 squared, which is a plausible matrix, not a hypothetical one.
 *
 * Complex values are built as `re + im * I` rather than multiplied with `*`.
 * The operator would call __muldc3 for its inf/nan bookkeeping; the explicit
 * form compiles to the same handful of instructions with no call.
 */

/*
 *  C <- beta * C, the once-per-call scaling every gemm starts with
 *
 *  beta == 0 stores zeros instead of multiplying, so a caller may hand us an
 *  uninitialised or NaN-filled C and still get a finite result.
 */
static void
scale_block_s(int32_t m, int32_t n, float beta, float *restrict c, int32_t ldc)
{
    if (beta == 1.0f) return;

    if (beta == 0.0f) {
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                c[(ptrdiff_t)i * ldc + j] = 0.0f;

        return;
    }

    for (int32_t i = 0; i < m; ++i)
        for (int32_t j = 0; j < n; ++j)
            c[(ptrdiff_t)i * ldc + j] *= beta;
}

static void
scale_block_d(int32_t m, int32_t n, double beta, double *restrict c, int32_t ldc)
{
    if (beta == 1.0) return;

    if (beta == 0.0) {
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                c[(ptrdiff_t)i * ldc + j] = 0.0;

        return;
    }

    for (int32_t i = 0; i < m; ++i)
        for (int32_t j = 0; j < n; ++j)
            c[(ptrdiff_t)i * ldc + j] *= beta;
}

static void
scale_block_c(int32_t m, int32_t n, float complex beta,
        float complex *restrict c, int32_t ldc)
{
    if (beta == 1.0f) return;

    if (beta == 0.0f) {
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                c[(ptrdiff_t)i * ldc + j] = 0.0f;

        return;
    }

    {
        float br = crealf(beta), bi = cimagf(beta);

        for (int32_t i = 0; i < m; ++i) {
            for (int32_t j = 0; j < n; ++j) {
                float complex z = c[(ptrdiff_t)i * ldc + j];
                float zr = crealf(z), zi = cimagf(z);

                c[(ptrdiff_t)i * ldc + j] = (br * zr - bi * zi)
                                          + (br * zi + bi * zr) * I;
            }
        }
    }
}

static void
scale_block_z(int32_t m, int32_t n, double complex beta,
        double complex *restrict c, int32_t ldc)
{
    if (beta == 1.0) return;

    if (beta == 0.0) {
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                c[(ptrdiff_t)i * ldc + j] = 0.0;

        return;
    }

    {
        double br = creal(beta), bi = cimag(beta);

        for (int32_t i = 0; i < m; ++i) {
            for (int32_t j = 0; j < n; ++j) {
                double complex z = c[(ptrdiff_t)i * ldc + j];
                double zr = creal(z), zi = cimag(z);

                c[(ptrdiff_t)i * ldc + j] = (br * zr - bi * zi)
                                          + (br * zi + bi * zr) * I;
            }
        }
    }
}

/*
 *  C += alpha * A * B, the unblocked reference
 *
 *  Loop order is (i, p, j) so both B and C run contiguously in the inner loop
 *  whenever the operands are untransposed. alpha is folded into the A element
 *  once per (i, p), which is where the packed path will fold it too.
 */
static void
gemm_naive_s(int32_t m, int32_t n, int32_t k, float alpha,
        const float *restrict a, ptrdiff_t ars, ptrdiff_t acs,
        const float *restrict b, ptrdiff_t brs, ptrdiff_t bcs,
        float *restrict c, int32_t ldc)
{
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t p = 0; p < k; ++p) {
            float av = alpha * a[(ptrdiff_t)i * ars + (ptrdiff_t)p * acs];

            for (int32_t j = 0; j < n; ++j)
                c[(ptrdiff_t)i * ldc + j] +=
                        av * b[(ptrdiff_t)p * brs + (ptrdiff_t)j * bcs];
        }
    }
}

static void
gemm_naive_d(int32_t m, int32_t n, int32_t k, double alpha,
        const double *restrict a, ptrdiff_t ars, ptrdiff_t acs,
        const double *restrict b, ptrdiff_t brs, ptrdiff_t bcs,
        double *restrict c, int32_t ldc)
{
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t p = 0; p < k; ++p) {
            double av = alpha * a[(ptrdiff_t)i * ars + (ptrdiff_t)p * acs];

            for (int32_t j = 0; j < n; ++j)
                c[(ptrdiff_t)i * ldc + j] +=
                        av * b[(ptrdiff_t)p * brs + (ptrdiff_t)j * bcs];
        }
    }
}

static void
gemm_naive_c(int32_t m, int32_t n, int32_t k, float complex alpha,
        const float complex *restrict a, ptrdiff_t ars, ptrdiff_t acs, int conja,
        const float complex *restrict b, ptrdiff_t brs, ptrdiff_t bcs, int conjb,
        float complex *restrict c, int32_t ldc)
{
    float alr = crealf(alpha), ali = cimagf(alpha);

    for (int32_t i = 0; i < m; ++i) {
        for (int32_t p = 0; p < k; ++p) {
            float complex az = a[(ptrdiff_t)i * ars + (ptrdiff_t)p * acs];
            float ar = crealf(az);
            float ai = conja ? -cimagf(az) : cimagf(az);

            float vr = alr * ar - ali * ai;
            float vi = alr * ai + ali * ar;

            for (int32_t j = 0; j < n; ++j) {
                float complex bz = b[(ptrdiff_t)p * brs + (ptrdiff_t)j * bcs];
                float br = crealf(bz);
                float bi = conjb ? -cimagf(bz) : cimagf(bz);

                float complex z = c[(ptrdiff_t)i * ldc + j];

                c[(ptrdiff_t)i * ldc + j] = (crealf(z) + (vr * br - vi * bi))
                                          + (cimagf(z) + (vr * bi + vi * br)) * I;
            }
        }
    }
}

static void
gemm_naive_z(int32_t m, int32_t n, int32_t k, double complex alpha,
        const double complex *restrict a, ptrdiff_t ars, ptrdiff_t acs, int conja,
        const double complex *restrict b, ptrdiff_t brs, ptrdiff_t bcs, int conjb,
        double complex *restrict c, int32_t ldc)
{
    double alr = creal(alpha), ali = cimag(alpha);

    for (int32_t i = 0; i < m; ++i) {
        for (int32_t p = 0; p < k; ++p) {
            double complex az = a[(ptrdiff_t)i * ars + (ptrdiff_t)p * acs];
            double ar = creal(az);
            double ai = conja ? -cimag(az) : cimag(az);

            double vr = alr * ar - ali * ai;
            double vi = alr * ai + ali * ar;

            for (int32_t j = 0; j < n; ++j) {
                double complex bz = b[(ptrdiff_t)p * brs + (ptrdiff_t)j * bcs];
                double br = creal(bz);
                double bi = conjb ? -cimag(bz) : cimag(bz);

                double complex z = c[(ptrdiff_t)i * ldc + j];

                c[(ptrdiff_t)i * ldc + j] = (creal(z) + (vr * br - vi * bi))
                                          + (cimag(z) + (vr * bi + vi * br)) * I;
            }
        }
    }
}

/* Micro-kernel shape and cache blocking for double.
 *
 * C is row-major, so a ymm register holds four consecutive columns of one C
 * row: NR is the vectorised dimension and MR the broadcast one. 6x8 uses the
 * register file exactly:
 *
 *     accumulators  6*8/4 = 12 ymm
 *     B micro-row   8/4   =  2 ymm   (8 doubles, one 64-byte cache line)
 *     A broadcasts           2 ymm
 *                           ------
 *                            16 ymm
 *
 * Per k step that is 12 FMAs against 8 loads. Skylake retires two FMAs and two
 * loads per cycle, so the kernel is FMA-bound with two load slots to spare.
 * FMA latency is 4 cycles on 2 ports, needing 8 independent chains to hide;
 * there are 12, and that slack is why 6x8 beats 4x8.
 *
 * KC keeps the B micro-panel (256*8*8 = 16 KB) and the A micro-panel
 * (6*256*8 = 12 KB) inside the 32 KB L1d. MC keeps the A block in the 256 KB
 * L2, taking only ~56% of it because this L2 is 4-way, not 8-way. NC keeps the
 * B block (256*3072*8 = 6 MB) inside the 8 MB L3.
 *
 * They are overridable (-DD_MC=... and friends) so the bench can sweep them.
 *
 * ponytail: the defaults are derived from the cache sizes, not measured. A
 * sweep over MC in {72..192} and KC in {192..384} on this box was inconclusive:
 * sustained AVX2 throttles the chip enough that the run-to-run spread swamped
 * the parameter effect, and OpenBLAS itself, as the control, varied by 2x
 * across the same grid. Re-sweep on a machine that holds its clock, and
 * compare the ratio-to-OpenBLAS column rather than raw GFLOP/s, since that
 * ratio turns out to be thermally invariant (82.1% cold, 81.4% throttled).
 */
#ifndef D_MR
#define D_MR 6
#endif
#ifndef D_NR
#define D_NR 8
#endif
#ifndef D_MC
#define D_MC 72
#endif
#ifndef D_KC
#define D_KC 256
#endif
#ifndef D_NC
#define D_NC 3072
#endif

/* Float reuses MR = 6 with a doubled NR, so the accumulator count is again 12
 * and a B micro-row is again exactly one 64-byte cache line. MC doubles
 * because a float A block of the same byte size holds twice the rows. */
#ifndef S_MR
#define S_MR 6
#endif
#ifndef S_NR
#define S_NR 16
#endif
#ifndef S_MC
#define S_MC 144
#endif
#ifndef S_KC
#define S_KC 256
#endif
#ifndef S_NC
#define S_NC 3072
#endif

/* Complex micro-kernel shape. One vector holds two complex doubles (four
 * complex floats), so NR = 4 (8) complex is two vectors, and MR = 4 gives
 * eight accumulators: the minimum in flight to cover a four-cycle FMA latency
 * at two FMAs per cycle. MR = 6 would want 19 live registers and spill, which
 * is why the real kernel's 6 by 2 shape is not reused here. A B micro-row is
 * 64 bytes either way, one cache line, as in the real packers.
 *
 * The cache blocks hold the same byte budget as their real counterparts, so
 * the double-complex MC halves against D_MC and the rest carry over.
 *
 * ponytail: this lands at roughly 87% of the real kernel's own rate. The gap
 * is the swap-and-negate pair per k step plus eight broadcasts against the
 * real kernel's six, and closing it wants a wider register file than AVX2
 * has, not a better arrangement of this one. KC was swept against 128 on a
 * paired interleaved run; 256 won, so the L1 arithmetic that suggests 128 is
 * not what binds here. */
#ifndef Z_MR
#define Z_MR 4
#endif
#ifndef Z_NR
#define Z_NR 4
#endif
#ifndef Z_MC
#define Z_MC 36
#endif
#ifndef Z_KC
#define Z_KC 256
#endif
#ifndef Z_NC
#define Z_NC 1536
#endif

#ifndef C_MR
#define C_MR 4
#endif
#ifndef C_NR
#define C_NR 8
#endif
#ifndef C_MC
#define C_MC 72
#endif
#ifndef C_KC
#define C_KC 256
#endif
#ifndef C_NC
#define C_NC 3072
#endif

/* Below this the packing and the malloc cost more than the triple loop. */
#ifndef D_SMALL
#define D_SMALL 8000
#endif

/*
 *  The double micro-kernel: C += Ap * Bp over a full MR by NR tile
 *
 *  Both panels arrive packed, so the k loop walks them contiguously and needs
 *  no strides. alpha is already folded into Ap, and beta was applied to C
 *  before any of this ran, which is why the kernel only ever accumulates.
 */
static void
dgemm_ukernel(int32_t kc, const double *restrict ap, const double *restrict bp,
        double *restrict c, int32_t ldc)
{
#if defined(__AVX2__) && defined(__FMA__)
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

    for (int32_t p = 0; p < kc; ++p) {
        __m256d b0 = _mm256_loadu_pd(bp);
        __m256d b1 = _mm256_loadu_pd(bp + 4);
        __m256d a;

        a = _mm256_broadcast_sd(ap + 0);
        c00 = _mm256_fmadd_pd(a, b0, c00);
        c01 = _mm256_fmadd_pd(a, b1, c01);

        a = _mm256_broadcast_sd(ap + 1);
        c10 = _mm256_fmadd_pd(a, b0, c10);
        c11 = _mm256_fmadd_pd(a, b1, c11);

        a = _mm256_broadcast_sd(ap + 2);
        c20 = _mm256_fmadd_pd(a, b0, c20);
        c21 = _mm256_fmadd_pd(a, b1, c21);

        a = _mm256_broadcast_sd(ap + 3);
        c30 = _mm256_fmadd_pd(a, b0, c30);
        c31 = _mm256_fmadd_pd(a, b1, c31);

        a = _mm256_broadcast_sd(ap + 4);
        c40 = _mm256_fmadd_pd(a, b0, c40);
        c41 = _mm256_fmadd_pd(a, b1, c41);

        a = _mm256_broadcast_sd(ap + 5);
        c50 = _mm256_fmadd_pd(a, b0, c50);
        c51 = _mm256_fmadd_pd(a, b1, c51);

        ap += D_MR;
        bp += D_NR;
    }

#define STORE_ROW(r, v0, v1) do { \
    double *cr = c + (ptrdiff_t)(r) * ldc; \
    _mm256_storeu_pd(cr,     _mm256_add_pd(_mm256_loadu_pd(cr),     (v0))); \
    _mm256_storeu_pd(cr + 4, _mm256_add_pd(_mm256_loadu_pd(cr + 4), (v1))); \
} while (0)

    STORE_ROW(0, c00, c01);
    STORE_ROW(1, c10, c11);
    STORE_ROW(2, c20, c21);
    STORE_ROW(3, c30, c31);
    STORE_ROW(4, c40, c41);
    STORE_ROW(5, c50, c51);

#undef STORE_ROW
#else
    /* Portable path: plain C99 over the same packed layout. It auto-vectorises
     * to roughly 40% of the intrinsic version, which is what a fallback is for. */
    double acc[D_MR][D_NR];

    for (int32_t i = 0; i < D_MR; ++i)
        for (int32_t j = 0; j < D_NR; ++j)
            acc[i][j] = 0.0;

    for (int32_t p = 0; p < kc; ++p)
        for (int32_t i = 0; i < D_MR; ++i)
            for (int32_t j = 0; j < D_NR; ++j)
                acc[i][j] += ap[p * D_MR + i] * bp[p * D_NR + j];

    for (int32_t i = 0; i < D_MR; ++i)
        for (int32_t j = 0; j < D_NR; ++j)
            c[(ptrdiff_t)i * ldc + j] += acc[i][j];
#endif
}

/*
 *  Pack an MC by KC block of A into MR by KC micro-panels
 *
 *  Column-major inside each micro-panel, so one k step is MR consecutive
 *  doubles and the kernel broadcasts straight off the stream. alpha is folded
 *  in here, which is why the kernel never sees it. Short edge panels are
 *  zero-filled: the kernel then computes zeros there and the write-back throws
 *  them away, which is cheaper than carrying a second set of edge kernels.
 */
static void
pack_a_d(int32_t mc, int32_t kc, double alpha,
        const double *restrict a, ptrdiff_t rs, ptrdiff_t cs,
        double *restrict ap)
{
    for (int32_t i = 0; i < mc; i += D_MR) {
        int32_t mr = (mc - i < D_MR) ? mc - i : D_MR;

        for (int32_t p = 0; p < kc; ++p) {
            int32_t ii;

            for (ii = 0; ii < mr; ++ii)
                *ap++ = alpha * a[(ptrdiff_t)(i + ii) * rs + (ptrdiff_t)p * cs];

            for (; ii < D_MR; ++ii)
                *ap++ = 0.0;
        }
    }
}

/*
 *  Pack a KC by NC block of B into KC by NR micro-panels
 *
 *  Row-major inside each micro-panel, so one k step is NR consecutive doubles,
 *  which is exactly one cache line and one pair of vector loads.
 */
static void
pack_b_d(int32_t kc, int32_t nc,
        const double *restrict b, ptrdiff_t rs, ptrdiff_t cs,
        double *restrict bp)
{
    for (int32_t j = 0; j < nc; j += D_NR) {
        int32_t nr = (nc - j < D_NR) ? nc - j : D_NR;

        if (cs == 1 && nr == D_NR) {
            /* untransposed B: the micro-panel row is already contiguous */
            for (int32_t p = 0; p < kc; ++p) {
                const double *src = b + (ptrdiff_t)p * rs + j;

                for (int32_t jj = 0; jj < D_NR; ++jj)
                    *bp++ = src[jj];
            }

            continue;
        }

        for (int32_t p = 0; p < kc; ++p) {
            int32_t jj;

            for (jj = 0; jj < nr; ++jj)
                *bp++ = b[(ptrdiff_t)p * rs + (ptrdiff_t)(j + jj) * cs];

            for (; jj < D_NR; ++jj)
                *bp++ = 0.0;
        }
    }
}

/*
 *  Sweep the packed panels over one MC by NC block of C
 *
 *  jr outer and ir inner, so the 16 KB B micro-panel stays in L1 for the whole
 *  ir sweep while the A micro-panels stream past it out of L2.
 */
static void
macro_kernel_d(int32_t mc, int32_t nc, int32_t kc,
        const double *restrict ap, const double *restrict bp,
        double *restrict c, int32_t ldc)
{
    for (int32_t j = 0; j < nc; j += D_NR) {
        int32_t nr = (nc - j < D_NR) ? nc - j : D_NR;
        const double *bpj = bp + (ptrdiff_t)(j / D_NR) * kc * D_NR;

        for (int32_t i = 0; i < mc; i += D_MR) {
            int32_t mr = (mc - i < D_MR) ? mc - i : D_MR;
            const double *api = ap + (ptrdiff_t)(i / D_MR) * kc * D_MR;
            double *cij = c + (ptrdiff_t)i * ldc + j;

            if (mr == D_MR && nr == D_NR) {
                dgemm_ukernel(kc, api, bpj, cij, ldc);

                continue;
            }

            {
                double ct[D_MR * D_NR];

                for (int32_t t = 0; t < D_MR * D_NR; ++t)
                    ct[t] = 0.0;

                dgemm_ukernel(kc, api, bpj, ct, D_NR);

                for (int32_t ii = 0; ii < mr; ++ii)
                    for (int32_t jj = 0; jj < nr; ++jj)
                        cij[(ptrdiff_t)ii * ldc + jj] += ct[ii * D_NR + jj];
            }
        }
    }
}

/*
 *  The five-loop nest around the micro-kernel
 *
 *  The A block is reused across every jr panel (L2 residency) and the B block
 *  across every ic step (L3 residency), which is what MC and NC were sized for.
 *  C has already been scaled by beta, so this only ever accumulates.
 */
static void
gemm_packed_d(int32_t m, int32_t n, int32_t k, double alpha,
        const double *restrict a, ptrdiff_t ars, ptrdiff_t acs,
        const double *restrict b, ptrdiff_t brs, ptrdiff_t bcs,
        double *restrict c, int32_t ldc,
        double *restrict ap, double *restrict bp)
{
    for (int32_t jc = 0; jc < n; jc += D_NC) {
        int32_t nc = (n - jc < D_NC) ? n - jc : D_NC;

        for (int32_t pc = 0; pc < k; pc += D_KC) {
            int32_t kc = (k - pc < D_KC) ? k - pc : D_KC;

            pack_b_d(kc, nc,
                     b + (ptrdiff_t)pc * brs + (ptrdiff_t)jc * bcs, brs, bcs, bp);

            for (int32_t ic = 0; ic < m; ic += D_MC) {
                int32_t mc = (m - ic < D_MC) ? m - ic : D_MC;

                pack_a_d(mc, kc, alpha,
                         a + (ptrdiff_t)ic * ars + (ptrdiff_t)pc * acs,
                         ars, acs, ap);

                macro_kernel_d(mc, nc, kc, ap, bp,
                               c + (ptrdiff_t)ic * ldc + jc, ldc);
            }
        }
    }
}

/*
 *  The float micro-kernel: C += Ap * Bp over a full MR by NR tile
 *
 *  Same 6 by 2-register shape as the double kernel; each ymm now holds eight
 *  floats instead of four doubles, so NR is 16 rather than 8.
 */
static void
sgemm_ukernel(int32_t kc, const float *restrict ap, const float *restrict bp,
        float *restrict c, int32_t ldc)
{
#if defined(__AVX2__) && defined(__FMA__)
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int32_t p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(bp);
        __m256 b1 = _mm256_loadu_ps(bp + 8);
        __m256 a;

        a = _mm256_broadcast_ss(ap + 0);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);

        a = _mm256_broadcast_ss(ap + 1);
        c10 = _mm256_fmadd_ps(a, b0, c10);
        c11 = _mm256_fmadd_ps(a, b1, c11);

        a = _mm256_broadcast_ss(ap + 2);
        c20 = _mm256_fmadd_ps(a, b0, c20);
        c21 = _mm256_fmadd_ps(a, b1, c21);

        a = _mm256_broadcast_ss(ap + 3);
        c30 = _mm256_fmadd_ps(a, b0, c30);
        c31 = _mm256_fmadd_ps(a, b1, c31);

        a = _mm256_broadcast_ss(ap + 4);
        c40 = _mm256_fmadd_ps(a, b0, c40);
        c41 = _mm256_fmadd_ps(a, b1, c41);

        a = _mm256_broadcast_ss(ap + 5);
        c50 = _mm256_fmadd_ps(a, b0, c50);
        c51 = _mm256_fmadd_ps(a, b1, c51);

        ap += S_MR;
        bp += S_NR;
    }

#define STORE_ROW(r, v0, v1) do { \
    float *cr = c + (ptrdiff_t)(r) * ldc; \
    _mm256_storeu_ps(cr,     _mm256_add_ps(_mm256_loadu_ps(cr),     (v0))); \
    _mm256_storeu_ps(cr + 8, _mm256_add_ps(_mm256_loadu_ps(cr + 8), (v1))); \
} while (0)

    STORE_ROW(0, c00, c01);
    STORE_ROW(1, c10, c11);
    STORE_ROW(2, c20, c21);
    STORE_ROW(3, c30, c31);
    STORE_ROW(4, c40, c41);
    STORE_ROW(5, c50, c51);

#undef STORE_ROW
#else
    float acc[S_MR][S_NR];

    for (int32_t i = 0; i < S_MR; ++i)
        for (int32_t j = 0; j < S_NR; ++j)
            acc[i][j] = 0.0f;

    for (int32_t p = 0; p < kc; ++p)
        for (int32_t i = 0; i < S_MR; ++i)
            for (int32_t j = 0; j < S_NR; ++j)
                acc[i][j] += ap[p * S_MR + i] * bp[p * S_NR + j];

    for (int32_t i = 0; i < S_MR; ++i)
        for (int32_t j = 0; j < S_NR; ++j)
            c[(ptrdiff_t)i * ldc + j] += acc[i][j];
#endif
}

static void
pack_a_s(int32_t mc, int32_t kc, float alpha,
        const float *restrict a, ptrdiff_t rs, ptrdiff_t cs,
        float *restrict ap)
{
    for (int32_t i = 0; i < mc; i += S_MR) {
        int32_t mr = (mc - i < S_MR) ? mc - i : S_MR;

        for (int32_t p = 0; p < kc; ++p) {
            int32_t ii;

            for (ii = 0; ii < mr; ++ii)
                *ap++ = alpha * a[(ptrdiff_t)(i + ii) * rs + (ptrdiff_t)p * cs];

            for (; ii < S_MR; ++ii)
                *ap++ = 0.0f;
        }
    }
}

static void
pack_b_s(int32_t kc, int32_t nc,
        const float *restrict b, ptrdiff_t rs, ptrdiff_t cs,
        float *restrict bp)
{
    for (int32_t j = 0; j < nc; j += S_NR) {
        int32_t nr = (nc - j < S_NR) ? nc - j : S_NR;

        if (cs == 1 && nr == S_NR) {
            for (int32_t p = 0; p < kc; ++p) {
                const float *src = b + (ptrdiff_t)p * rs + j;

                for (int32_t jj = 0; jj < S_NR; ++jj)
                    *bp++ = src[jj];
            }

            continue;
        }

        for (int32_t p = 0; p < kc; ++p) {
            int32_t jj;

            for (jj = 0; jj < nr; ++jj)
                *bp++ = b[(ptrdiff_t)p * rs + (ptrdiff_t)(j + jj) * cs];

            for (; jj < S_NR; ++jj)
                *bp++ = 0.0f;
        }
    }
}

static void
macro_kernel_s(int32_t mc, int32_t nc, int32_t kc,
        const float *restrict ap, const float *restrict bp,
        float *restrict c, int32_t ldc)
{
    for (int32_t j = 0; j < nc; j += S_NR) {
        int32_t nr = (nc - j < S_NR) ? nc - j : S_NR;
        const float *bpj = bp + (ptrdiff_t)(j / S_NR) * kc * S_NR;

        for (int32_t i = 0; i < mc; i += S_MR) {
            int32_t mr = (mc - i < S_MR) ? mc - i : S_MR;
            const float *api = ap + (ptrdiff_t)(i / S_MR) * kc * S_MR;
            float *cij = c + (ptrdiff_t)i * ldc + j;

            if (mr == S_MR && nr == S_NR) {
                sgemm_ukernel(kc, api, bpj, cij, ldc);

                continue;
            }

            {
                float ct[S_MR * S_NR];

                for (int32_t t = 0; t < S_MR * S_NR; ++t)
                    ct[t] = 0.0f;

                sgemm_ukernel(kc, api, bpj, ct, S_NR);

                for (int32_t ii = 0; ii < mr; ++ii)
                    for (int32_t jj = 0; jj < nr; ++jj)
                        cij[(ptrdiff_t)ii * ldc + jj] += ct[ii * S_NR + jj];
            }
        }
    }
}

static void
gemm_packed_s(int32_t m, int32_t n, int32_t k, float alpha,
        const float *restrict a, ptrdiff_t ars, ptrdiff_t acs,
        const float *restrict b, ptrdiff_t brs, ptrdiff_t bcs,
        float *restrict c, int32_t ldc,
        float *restrict ap, float *restrict bp)
{
    for (int32_t jc = 0; jc < n; jc += S_NC) {
        int32_t nc = (n - jc < S_NC) ? n - jc : S_NC;

        for (int32_t pc = 0; pc < k; pc += S_KC) {
            int32_t kc = (k - pc < S_KC) ? k - pc : S_KC;

            pack_b_s(kc, nc,
                     b + (ptrdiff_t)pc * brs + (ptrdiff_t)jc * bcs, brs, bcs, bp);

            for (int32_t ic = 0; ic < m; ic += S_MC) {
                int32_t mc = (m - ic < S_MC) ? m - ic : S_MC;

                pack_a_s(mc, kc, alpha,
                         a + (ptrdiff_t)ic * ars + (ptrdiff_t)pc * acs,
                         ars, acs, ap);

                macro_kernel_s(mc, nc, kc, ap, bp,
                               c + (ptrdiff_t)ic * ldc + jc, ldc);
            }
        }
    }
}

/*
 *  Single-precision general matrix multiply: sgemm
 */
void
tinyblas_sgemm(enum tinyblas_op transa, enum tinyblas_op transb,
        int32_t m, int32_t n, int32_t k,
        float alpha,
        const float *restrict a, int32_t lda,
        const float *restrict b, int32_t ldb,
        float beta,
        float *restrict c, int32_t ldc)
{
    if (m <= 0 || n <= 0) return;

    assert(c);
    assert(ldc >= n);

    scale_block_s(m, n, beta, c, ldc);

    if (k <= 0 || alpha == 0.0f) return;

    assert(a && b);

    {
        ptrdiff_t ars = transa == TINYBLAS_NONE ? (ptrdiff_t)lda : 1;
        ptrdiff_t acs = transa == TINYBLAS_NONE ? 1 : (ptrdiff_t)lda;
        ptrdiff_t brs = transb == TINYBLAS_NONE ? (ptrdiff_t)ldb : 1;
        ptrdiff_t bcs = transb == TINYBLAS_NONE ? 1 : (ptrdiff_t)ldb;

        int32_t mcl = m < S_MC ? m : S_MC;
        int32_t kcl = k < S_KC ? k : S_KC;
        int32_t ncl = n < S_NC ? n : S_NC;

        char  *raw;
        float *ap, *bp;

        mcl = ((mcl + S_MR - 1) / S_MR) * S_MR;
        ncl = ((ncl + S_NR - 1) / S_NR) * S_NR;

        raw = malloc(((size_t)mcl * (size_t)kcl + (size_t)kcl * (size_t)ncl)
                     * sizeof(float) + 64u);

        if (raw == NULL || (int64_t)m * (int64_t)n * (int64_t)k < D_SMALL) {
            free(raw);
            gemm_naive_s(m, n, k, alpha, a, ars, acs, b, brs, bcs, c, ldc);

            return;
        }

        ap = (float *)(void *)(((uintptr_t)raw + 63u) & ~(uintptr_t)63);
        bp = ap + (size_t)mcl * (size_t)kcl;

        gemm_packed_s(m, n, k, alpha, a, ars, acs, b, brs, bcs, c, ldc, ap, bp);

        free(raw);
    }
}

/*
 *  Double-precision general matrix multiply: dgemm
 */
void
tinyblas_dgemm(enum tinyblas_op transa, enum tinyblas_op transb,
        int32_t m, int32_t n, int32_t k,
        double alpha,
        const double *restrict a, int32_t lda,
        const double *restrict b, int32_t ldb,
        double beta,
        double *restrict c, int32_t ldc)
{
    if (m <= 0 || n <= 0) return;

    assert(c);
    assert(ldc >= n);

    scale_block_d(m, n, beta, c, ldc);

    if (k <= 0 || alpha == 0.0) return;

    assert(a && b);

    {
        ptrdiff_t ars = transa == TINYBLAS_NONE ? (ptrdiff_t)lda : 1;
        ptrdiff_t acs = transa == TINYBLAS_NONE ? 1 : (ptrdiff_t)lda;
        ptrdiff_t brs = transb == TINYBLAS_NONE ? (ptrdiff_t)ldb : 1;
        ptrdiff_t bcs = transb == TINYBLAS_NONE ? 1 : (ptrdiff_t)ldb;

        /* Clipped to the problem, so a 64-cubed gemm allocates 64 KB rather
         * than the 6 MB the full blocking would ask for. Edge micro-panels are
         * zero-padded to MR and NR, so round up rather than down. */
        int32_t mcl = m < D_MC ? m : D_MC;
        int32_t kcl = k < D_KC ? k : D_KC;
        int32_t ncl = n < D_NC ? n : D_NC;

        char   *raw;
        double *ap, *bp;

        mcl = ((mcl + D_MR - 1) / D_MR) * D_MR;
        ncl = ((ncl + D_NR - 1) / D_NR) * D_NR;

        raw = malloc(((size_t)mcl * (size_t)kcl + (size_t)kcl * (size_t)ncl)
                     * sizeof(double) + 64u);

        /* ponytail: 8000 is a guess at where packing stops paying for itself.
         * The bench owns this number; re-measure it if the kernel changes. */
        if (raw == NULL || (int64_t)m * (int64_t)n * (int64_t)k < D_SMALL) {
            free(raw);
            gemm_naive_d(m, n, k, alpha, a, ars, acs, b, brs, bcs, c, ldc);

            return;
        }

        ap = (double *)(void *)(((uintptr_t)raw + 63u) & ~(uintptr_t)63);
        bp = ap + (size_t)mcl * (size_t)kcl;

        gemm_packed_d(m, n, k, alpha, a, ars, acs, b, brs, bcs, c, ldc, ap, bp);

        free(raw);
    }
}

/*
 *  The double-complex micro-kernel: C += Ap * Bp over a full 4 by 4 tile
 *
 *  Both panels are packed interleaved (re, im), so one vector holds two
 *  complex numbers. A complex multiply-add is two FMAs and nothing else:
 *
 *      c += ar * (br, bi)  +  ai * (-bi, br)
 *
 *  The second operand is the first with the halves swapped and the real lane
 *  negated, and it depends only on B, so it is built once per k step and
 *  shared across all 4 rows. Two FMAs of 4 lanes do two complex MACs,
 *  which is the conventional 8 flops each with none wasted — the same peak the
 *  real kernel reaches, which is the entire point of not splitting into four
 *  real gemms.
 *
 *  c points at the tile as interleaved scalars and ldc is its row stride in
 *  scalars, not in complex elements.
 */
static void
zgemm_ukernel(int32_t kc, const double *restrict ap, const double *restrict bp,
        double *restrict c, ptrdiff_t ldc)
{
#if defined(__AVX2__) && defined(__FMA__)
    const __m256d neg = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);

    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();

    for (int32_t p = 0; p < kc; ++p) {
        __m256d b0 = _mm256_loadu_pd(bp);
        __m256d b1 = _mm256_loadu_pd(bp + 4);
        __m256d b0s = _mm256_xor_pd(_mm256_permute_pd(b0, 0x5), neg);
        __m256d b1s = _mm256_xor_pd(_mm256_permute_pd(b1, 0x5), neg);
        __m256d ar, ai;

        ar = _mm256_broadcast_sd(ap + 0);
        ai = _mm256_broadcast_sd(ap + 1);
        c00 = _mm256_fmadd_pd(ar, b0, c00);
        c00 = _mm256_fmadd_pd(ai, b0s, c00);
        c01 = _mm256_fmadd_pd(ar, b1, c01);
        c01 = _mm256_fmadd_pd(ai, b1s, c01);

        ar = _mm256_broadcast_sd(ap + 2);
        ai = _mm256_broadcast_sd(ap + 3);
        c10 = _mm256_fmadd_pd(ar, b0, c10);
        c10 = _mm256_fmadd_pd(ai, b0s, c10);
        c11 = _mm256_fmadd_pd(ar, b1, c11);
        c11 = _mm256_fmadd_pd(ai, b1s, c11);

        ar = _mm256_broadcast_sd(ap + 4);
        ai = _mm256_broadcast_sd(ap + 5);
        c20 = _mm256_fmadd_pd(ar, b0, c20);
        c20 = _mm256_fmadd_pd(ai, b0s, c20);
        c21 = _mm256_fmadd_pd(ar, b1, c21);
        c21 = _mm256_fmadd_pd(ai, b1s, c21);

        ar = _mm256_broadcast_sd(ap + 6);
        ai = _mm256_broadcast_sd(ap + 7);
        c30 = _mm256_fmadd_pd(ar, b0, c30);
        c30 = _mm256_fmadd_pd(ai, b0s, c30);
        c31 = _mm256_fmadd_pd(ar, b1, c31);
        c31 = _mm256_fmadd_pd(ai, b1s, c31);

        ap += 2 * Z_MR;
        bp += 2 * Z_NR;
    }

#define STORE_ROW(r, v0, v1) do { \
    double *cr = c + (ptrdiff_t)(r) * ldc; \
    _mm256_storeu_pd(cr,     _mm256_add_pd(_mm256_loadu_pd(cr),     (v0))); \
    _mm256_storeu_pd(cr + 4, _mm256_add_pd(_mm256_loadu_pd(cr + 4), (v1))); \
} while (0)

    STORE_ROW(0, c00, c01);
    STORE_ROW(1, c10, c11);
    STORE_ROW(2, c20, c21);
    STORE_ROW(3, c30, c31);

#undef STORE_ROW
#else
    /* Portable path: plain C99 over the same packed layout. */
    double acc[Z_MR][2 * Z_NR];

    for (int32_t i = 0; i < Z_MR; ++i)
        for (int32_t j = 0; j < 2 * Z_NR; ++j)
            acc[i][j] = 0.0;

    for (int32_t p = 0; p < kc; ++p) {
        for (int32_t i = 0; i < Z_MR; ++i) {
            double ar = ap[p * 2 * Z_MR + 2 * i];
            double ai = ap[p * 2 * Z_MR + 2 * i + 1];

            for (int32_t j = 0; j < Z_NR; ++j) {
                double br = bp[p * 2 * Z_NR + 2 * j];
                double bi = bp[p * 2 * Z_NR + 2 * j + 1];

                acc[i][2 * j]     += ar * br - ai * bi;
                acc[i][2 * j + 1] += ar * bi + ai * br;
            }
        }
    }

    for (int32_t i = 0; i < Z_MR; ++i)
        for (int32_t j = 0; j < 2 * Z_NR; ++j)
            c[(ptrdiff_t)i * ldc + j] += acc[i][j];
#endif
}

/*
 *  Pack an MC by KC block of A into 4 by KC micro-panels
 *
 *  alpha and the conjugate are both folded in here, which is why the kernel
 *  sees neither. Short edge panels are zero-filled, as in the real packers.
 */
static void
pack_a_z(int32_t mc, int32_t kc, double complex alpha,
        const double complex *restrict a, ptrdiff_t rs, ptrdiff_t cs, int conj,
        double *restrict ap)
{
    double alr = creal(alpha), ali = cimag(alpha);

    for (int32_t i = 0; i < mc; i += Z_MR) {
        int32_t mr = (mc - i < Z_MR) ? mc - i : Z_MR;

        for (int32_t p = 0; p < kc; ++p) {
            int32_t ii;

            for (ii = 0; ii < mr; ++ii) {
                double complex z = a[(ptrdiff_t)(i + ii) * rs + (ptrdiff_t)p * cs];
                double zr = creal(z), zi = conj ? -cimag(z) : cimag(z);

                *ap++ = alr * zr - ali * zi;
                *ap++ = alr * zi + ali * zr;
            }

            for (; ii < Z_MR; ++ii) {
                *ap++ = 0.0;
                *ap++ = 0.0;
            }
        }
    }
}

/*
 *  Pack a KC by NC block of B into KC by 4 micro-panels
 *
 *  One k step is 4 consecutive complex numbers, which is 64 bytes: one
 *  cache line and one pair of vector loads, exactly as in the real packers.
 */
static void
pack_b_z(int32_t kc, int32_t nc,
        const double complex *restrict b, ptrdiff_t rs, ptrdiff_t cs, int conj,
        double *restrict bp)
{
    for (int32_t j = 0; j < nc; j += Z_NR) {
        int32_t nr = (nc - j < Z_NR) ? nc - j : Z_NR;

        for (int32_t p = 0; p < kc; ++p) {
            int32_t jj;

            for (jj = 0; jj < nr; ++jj) {
                double complex z = b[(ptrdiff_t)p * rs + (ptrdiff_t)(j + jj) * cs];

                *bp++ = creal(z);
                *bp++ = conj ? -cimag(z) : cimag(z);
            }

            for (; jj < Z_NR; ++jj) {
                *bp++ = 0.0;
                *bp++ = 0.0;
            }
        }
    }
}

/*
 *  Sweep the packed panels over one MC by NC block of C
 */
static void
macro_kernel_z(int32_t mc, int32_t nc, int32_t kc,
        const double *restrict ap, const double *restrict bp,
        double complex *restrict c, int32_t ldc)
{
    for (int32_t j = 0; j < nc; j += Z_NR) {
        int32_t nr = (nc - j < Z_NR) ? nc - j : Z_NR;
        const double *bpj = bp + (ptrdiff_t)(j / Z_NR) * kc * 2 * Z_NR;

        for (int32_t i = 0; i < mc; i += Z_MR) {
            int32_t mr = (mc - i < Z_MR) ? mc - i : Z_MR;
            const double *api = ap + (ptrdiff_t)(i / Z_MR) * kc * 2 * Z_MR;
            double complex *cij = c + (ptrdiff_t)i * ldc + j;

            if (mr == Z_MR && nr == Z_NR) {
                zgemm_ukernel(kc, api, bpj, (double *)(void *)cij,
                                2 * (ptrdiff_t)ldc);

                continue;
            }

            {
                double ct[Z_MR * 2 * Z_NR];

                for (int32_t t = 0; t < Z_MR * 2 * Z_NR; ++t)
                    ct[t] = 0.0;

                zgemm_ukernel(kc, api, bpj, ct, 2 * Z_NR);

                for (int32_t ii = 0; ii < mr; ++ii) {
                    for (int32_t jj = 0; jj < nr; ++jj) {
                        double pr = ct[ii * 2 * Z_NR + 2 * jj];
                        double pi = ct[ii * 2 * Z_NR + 2 * jj + 1];
                        double complex z = cij[(ptrdiff_t)ii * ldc + jj];

                        cij[(ptrdiff_t)ii * ldc + jj] = (creal(z) + pr)
                                                      + (cimag(z) + pi) * I;
                    }
                }
            }
        }
    }
}

/*
 *  The five-loop nest around the complex micro-kernel
 */
static void
gemm_packed_z(int32_t m, int32_t n, int32_t k, double complex alpha,
        const double complex *restrict a, ptrdiff_t ars, ptrdiff_t acs, int conja,
        const double complex *restrict b, ptrdiff_t brs, ptrdiff_t bcs, int conjb,
        double complex *restrict c, int32_t ldc,
        double *restrict ap, double *restrict bp)
{
    for (int32_t jc = 0; jc < n; jc += Z_NC) {
        int32_t nc = (n - jc < Z_NC) ? n - jc : Z_NC;

        for (int32_t pc = 0; pc < k; pc += Z_KC) {
            int32_t kc = (k - pc < Z_KC) ? k - pc : Z_KC;

            pack_b_z(kc, nc,
                       b + (ptrdiff_t)pc * brs + (ptrdiff_t)jc * bcs,
                       brs, bcs, conjb, bp);

            for (int32_t ic = 0; ic < m; ic += Z_MC) {
                int32_t mc = (m - ic < Z_MC) ? m - ic : Z_MC;

                pack_a_z(mc, kc, alpha,
                           a + (ptrdiff_t)ic * ars + (ptrdiff_t)pc * acs,
                           ars, acs, conja, ap);

                macro_kernel_z(mc, nc, kc, ap, bp,
                                 c + (ptrdiff_t)ic * ldc + jc, ldc);
            }
        }
    }
}

/*
 *  The single-complex micro-kernel: C += Ap * Bp over a full 4 by 8 tile
 *
 *  Both panels are packed interleaved (re, im), so one vector holds four
 *  complex numbers. A complex multiply-add is two FMAs and nothing else:
 *
 *      c += ar * (br, bi)  +  ai * (-bi, br)
 *
 *  The second operand is the first with the halves swapped and the real lane
 *  negated, and it depends only on B, so it is built once per k step and
 *  shared across all 4 rows. Two FMAs of 8 lanes do four complex MACs,
 *  which is the conventional 8 flops each with none wasted — the same peak the
 *  real kernel reaches, which is the entire point of not splitting into four
 *  real gemms.
 *
 *  c points at the tile as interleaved scalars and ldc is its row stride in
 *  scalars, not in complex elements.
 */
static void
cgemm_ukernel(int32_t kc, const float *restrict ap, const float *restrict bp,
        float *restrict c, ptrdiff_t ldc)
{
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 neg = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f,
                                    -0.0f, 0.0f, -0.0f, 0.0f);

    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

    for (int32_t p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(bp);
        __m256 b1 = _mm256_loadu_ps(bp + 8);
        __m256 b0s = _mm256_xor_ps(_mm256_permute_ps(b0, 0xB1), neg);
        __m256 b1s = _mm256_xor_ps(_mm256_permute_ps(b1, 0xB1), neg);
        __m256 ar, ai;

        ar = _mm256_broadcast_ss(ap + 0);
        ai = _mm256_broadcast_ss(ap + 1);
        c00 = _mm256_fmadd_ps(ar, b0, c00);
        c00 = _mm256_fmadd_ps(ai, b0s, c00);
        c01 = _mm256_fmadd_ps(ar, b1, c01);
        c01 = _mm256_fmadd_ps(ai, b1s, c01);

        ar = _mm256_broadcast_ss(ap + 2);
        ai = _mm256_broadcast_ss(ap + 3);
        c10 = _mm256_fmadd_ps(ar, b0, c10);
        c10 = _mm256_fmadd_ps(ai, b0s, c10);
        c11 = _mm256_fmadd_ps(ar, b1, c11);
        c11 = _mm256_fmadd_ps(ai, b1s, c11);

        ar = _mm256_broadcast_ss(ap + 4);
        ai = _mm256_broadcast_ss(ap + 5);
        c20 = _mm256_fmadd_ps(ar, b0, c20);
        c20 = _mm256_fmadd_ps(ai, b0s, c20);
        c21 = _mm256_fmadd_ps(ar, b1, c21);
        c21 = _mm256_fmadd_ps(ai, b1s, c21);

        ar = _mm256_broadcast_ss(ap + 6);
        ai = _mm256_broadcast_ss(ap + 7);
        c30 = _mm256_fmadd_ps(ar, b0, c30);
        c30 = _mm256_fmadd_ps(ai, b0s, c30);
        c31 = _mm256_fmadd_ps(ar, b1, c31);
        c31 = _mm256_fmadd_ps(ai, b1s, c31);

        ap += 2 * C_MR;
        bp += 2 * C_NR;
    }

#define STORE_ROW(r, v0, v1) do { \
    float *cr = c + (ptrdiff_t)(r) * ldc; \
    _mm256_storeu_ps(cr,     _mm256_add_ps(_mm256_loadu_ps(cr),     (v0))); \
    _mm256_storeu_ps(cr + 8, _mm256_add_ps(_mm256_loadu_ps(cr + 8), (v1))); \
} while (0)

    STORE_ROW(0, c00, c01);
    STORE_ROW(1, c10, c11);
    STORE_ROW(2, c20, c21);
    STORE_ROW(3, c30, c31);

#undef STORE_ROW
#else
    /* Portable path: plain C99 over the same packed layout. */
    float acc[C_MR][2 * C_NR];

    for (int32_t i = 0; i < C_MR; ++i)
        for (int32_t j = 0; j < 2 * C_NR; ++j)
            acc[i][j] = 0.0f;

    for (int32_t p = 0; p < kc; ++p) {
        for (int32_t i = 0; i < C_MR; ++i) {
            float ar = ap[p * 2 * C_MR + 2 * i];
            float ai = ap[p * 2 * C_MR + 2 * i + 1];

            for (int32_t j = 0; j < C_NR; ++j) {
                float br = bp[p * 2 * C_NR + 2 * j];
                float bi = bp[p * 2 * C_NR + 2 * j + 1];

                acc[i][2 * j]     += ar * br - ai * bi;
                acc[i][2 * j + 1] += ar * bi + ai * br;
            }
        }
    }

    for (int32_t i = 0; i < C_MR; ++i)
        for (int32_t j = 0; j < 2 * C_NR; ++j)
            c[(ptrdiff_t)i * ldc + j] += acc[i][j];
#endif
}

/*
 *  Pack an MC by KC block of A into 4 by KC micro-panels
 *
 *  alpha and the conjugate are both folded in here, which is why the kernel
 *  sees neither. Short edge panels are zero-filled, as in the real packers.
 */
static void
pack_a_c(int32_t mc, int32_t kc, float complex alpha,
        const float complex *restrict a, ptrdiff_t rs, ptrdiff_t cs, int conj,
        float *restrict ap)
{
    float alr = crealf(alpha), ali = cimagf(alpha);

    for (int32_t i = 0; i < mc; i += C_MR) {
        int32_t mr = (mc - i < C_MR) ? mc - i : C_MR;

        for (int32_t p = 0; p < kc; ++p) {
            int32_t ii;

            for (ii = 0; ii < mr; ++ii) {
                float complex z = a[(ptrdiff_t)(i + ii) * rs + (ptrdiff_t)p * cs];
                float zr = crealf(z), zi = conj ? -cimagf(z) : cimagf(z);

                *ap++ = alr * zr - ali * zi;
                *ap++ = alr * zi + ali * zr;
            }

            for (; ii < C_MR; ++ii) {
                *ap++ = 0.0f;
                *ap++ = 0.0f;
            }
        }
    }
}

/*
 *  Pack a KC by NC block of B into KC by 8 micro-panels
 *
 *  One k step is 8 consecutive complex numbers, which is 64 bytes: one
 *  cache line and one pair of vector loads, exactly as in the real packers.
 */
static void
pack_b_c(int32_t kc, int32_t nc,
        const float complex *restrict b, ptrdiff_t rs, ptrdiff_t cs, int conj,
        float *restrict bp)
{
    for (int32_t j = 0; j < nc; j += C_NR) {
        int32_t nr = (nc - j < C_NR) ? nc - j : C_NR;

        for (int32_t p = 0; p < kc; ++p) {
            int32_t jj;

            for (jj = 0; jj < nr; ++jj) {
                float complex z = b[(ptrdiff_t)p * rs + (ptrdiff_t)(j + jj) * cs];

                *bp++ = crealf(z);
                *bp++ = conj ? -cimagf(z) : cimagf(z);
            }

            for (; jj < C_NR; ++jj) {
                *bp++ = 0.0f;
                *bp++ = 0.0f;
            }
        }
    }
}

/*
 *  Sweep the packed panels over one MC by NC block of C
 */
static void
macro_kernel_c(int32_t mc, int32_t nc, int32_t kc,
        const float *restrict ap, const float *restrict bp,
        float complex *restrict c, int32_t ldc)
{
    for (int32_t j = 0; j < nc; j += C_NR) {
        int32_t nr = (nc - j < C_NR) ? nc - j : C_NR;
        const float *bpj = bp + (ptrdiff_t)(j / C_NR) * kc * 2 * C_NR;

        for (int32_t i = 0; i < mc; i += C_MR) {
            int32_t mr = (mc - i < C_MR) ? mc - i : C_MR;
            const float *api = ap + (ptrdiff_t)(i / C_MR) * kc * 2 * C_MR;
            float complex *cij = c + (ptrdiff_t)i * ldc + j;

            if (mr == C_MR && nr == C_NR) {
                cgemm_ukernel(kc, api, bpj, (float *)(void *)cij,
                                2 * (ptrdiff_t)ldc);

                continue;
            }

            {
                float ct[C_MR * 2 * C_NR];

                for (int32_t t = 0; t < C_MR * 2 * C_NR; ++t)
                    ct[t] = 0.0f;

                cgemm_ukernel(kc, api, bpj, ct, 2 * C_NR);

                for (int32_t ii = 0; ii < mr; ++ii) {
                    for (int32_t jj = 0; jj < nr; ++jj) {
                        float pr = ct[ii * 2 * C_NR + 2 * jj];
                        float pi = ct[ii * 2 * C_NR + 2 * jj + 1];
                        float complex z = cij[(ptrdiff_t)ii * ldc + jj];

                        cij[(ptrdiff_t)ii * ldc + jj] = (crealf(z) + pr)
                                                      + (cimagf(z) + pi) * I;
                    }
                }
            }
        }
    }
}

/*
 *  The five-loop nest around the complex micro-kernel
 */
static void
gemm_packed_c(int32_t m, int32_t n, int32_t k, float complex alpha,
        const float complex *restrict a, ptrdiff_t ars, ptrdiff_t acs, int conja,
        const float complex *restrict b, ptrdiff_t brs, ptrdiff_t bcs, int conjb,
        float complex *restrict c, int32_t ldc,
        float *restrict ap, float *restrict bp)
{
    for (int32_t jc = 0; jc < n; jc += C_NC) {
        int32_t nc = (n - jc < C_NC) ? n - jc : C_NC;

        for (int32_t pc = 0; pc < k; pc += C_KC) {
            int32_t kc = (k - pc < C_KC) ? k - pc : C_KC;

            pack_b_c(kc, nc,
                       b + (ptrdiff_t)pc * brs + (ptrdiff_t)jc * bcs,
                       brs, bcs, conjb, bp);

            for (int32_t ic = 0; ic < m; ic += C_MC) {
                int32_t mc = (m - ic < C_MC) ? m - ic : C_MC;

                pack_a_c(mc, kc, alpha,
                           a + (ptrdiff_t)ic * ars + (ptrdiff_t)pc * acs,
                           ars, acs, conja, ap);

                macro_kernel_c(mc, nc, kc, ap, bp,
                                 c + (ptrdiff_t)ic * ldc + jc, ldc);
            }
        }
    }
}

/*
 *  Single-precision complex general matrix multiply: cgemm
 */
void
tinyblas_cgemm(enum tinyblas_op transa, enum tinyblas_op transb,
        int32_t m, int32_t n, int32_t k,
        float complex alpha,
        const float complex *restrict a, int32_t lda,
        const float complex *restrict b, int32_t ldb,
        float complex beta,
        float complex *restrict c, int32_t ldc)
{
    if (m <= 0 || n <= 0) return;

    assert(c);
    assert(ldc >= n);

    scale_block_c(m, n, beta, c, ldc);

    if (k <= 0 || alpha == 0.0f) return;

    assert(a && b);

    {
        ptrdiff_t ars = transa == TINYBLAS_NONE ? (ptrdiff_t)lda : 1;
        ptrdiff_t acs = transa == TINYBLAS_NONE ? 1 : (ptrdiff_t)lda;
        ptrdiff_t brs = transb == TINYBLAS_NONE ? (ptrdiff_t)ldb : 1;
        ptrdiff_t bcs = transb == TINYBLAS_NONE ? 1 : (ptrdiff_t)ldb;

        int conja = transa == TINYBLAS_CONJ_TRANS;
        int conjb = transb == TINYBLAS_CONJ_TRANS;

        int32_t mcl = m < C_MC ? m : C_MC;
        int32_t kcl = k < C_KC ? k : C_KC;
        int32_t ncl = n < C_NC ? n : C_NC;

        char *raw;
        float *ap, *bp;

        mcl = ((mcl + C_MR - 1) / C_MR) * C_MR;
        ncl = ((ncl + C_NR - 1) / C_NR) * C_NR;

        raw = malloc(2u * ((size_t)mcl * (size_t)kcl
                           + (size_t)kcl * (size_t)ncl) * sizeof(float) + 64u);

        if (raw == NULL || (int64_t)m * (int64_t)n * (int64_t)k < D_SMALL) {
            free(raw);
            gemm_naive_c(m, n, k, alpha, a, ars, acs, conja,
                         b, brs, bcs, conjb, c, ldc);

            return;
        }

        ap = (float *)(void *)(((uintptr_t)raw + 63u) & ~(uintptr_t)63);
        bp = ap + 2u * (size_t)mcl * (size_t)kcl;

        gemm_packed_c(m, n, k, alpha, a, ars, acs, conja,
                      b, brs, bcs, conjb, c, ldc, ap, bp);

        free(raw);
    }
}

/*
 *  Double-precision complex general matrix multiply: zgemm
 */
void
tinyblas_zgemm(enum tinyblas_op transa, enum tinyblas_op transb,
        int32_t m, int32_t n, int32_t k,
        double complex alpha,
        const double complex *restrict a, int32_t lda,
        const double complex *restrict b, int32_t ldb,
        double complex beta,
        double complex *restrict c, int32_t ldc)
{
    if (m <= 0 || n <= 0) return;

    assert(c);
    assert(ldc >= n);

    scale_block_z(m, n, beta, c, ldc);

    if (k <= 0 || alpha == 0.0) return;

    assert(a && b);

    {
        ptrdiff_t ars = transa == TINYBLAS_NONE ? (ptrdiff_t)lda : 1;
        ptrdiff_t acs = transa == TINYBLAS_NONE ? 1 : (ptrdiff_t)lda;
        ptrdiff_t brs = transb == TINYBLAS_NONE ? (ptrdiff_t)ldb : 1;
        ptrdiff_t bcs = transb == TINYBLAS_NONE ? 1 : (ptrdiff_t)ldb;

        int conja = transa == TINYBLAS_CONJ_TRANS;
        int conjb = transb == TINYBLAS_CONJ_TRANS;

        int32_t mcl = m < Z_MC ? m : Z_MC;
        int32_t kcl = k < Z_KC ? k : Z_KC;
        int32_t ncl = n < Z_NC ? n : Z_NC;

        char *raw;
        double *ap, *bp;

        mcl = ((mcl + Z_MR - 1) / Z_MR) * Z_MR;
        ncl = ((ncl + Z_NR - 1) / Z_NR) * Z_NR;

        raw = malloc(2u * ((size_t)mcl * (size_t)kcl
                           + (size_t)kcl * (size_t)ncl) * sizeof(double) + 64u);

        if (raw == NULL || (int64_t)m * (int64_t)n * (int64_t)k < D_SMALL) {
            free(raw);
            gemm_naive_z(m, n, k, alpha, a, ars, acs, conja,
                         b, brs, bcs, conjb, c, ldc);

            return;
        }

        ap = (double *)(void *)(((uintptr_t)raw + 63u) & ~(uintptr_t)63);
        bp = ap + 2u * (size_t)mcl * (size_t)kcl;

        gemm_packed_z(m, n, k, alpha, a, ars, acs, conja,
                      b, brs, bcs, conjb, c, ldc, ap, bp);

        free(raw);
    }
}
