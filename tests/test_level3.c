#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <complex.h>

#include "tinyblas.h"

#define CHECK(a, b, eps) do { \
    if (!((fabs((double)(a) - (double)(b)) <= (eps)))) { \
        printf("FAIL: %s:%d  got=%g expected=%g\n", \
               __FILE__, __LINE__, (double)(a), (double)(b)); \
        return 1; \
    } \
} while (0)

#define CHECK_C(z, re, im, eps) do { \
    CHECK(creal((double complex)(z)), (re), (eps)); \
    CHECK(cimag((double complex)(z)), (im), (eps)); \
} while (0)

/* Largest element count any single matrix needs. The shape sweep tops out at
 * 65 + 3 padding squared and the deep-k case is 8 by 320, but the blocked
 * trsm cases below run to 200 squared, which is what sets this. */
#define MAXSZ 65536

static double complex ma[MAXSZ], mb[MAXSZ], mc[MAXSZ];
static double complex ref[MAXSZ], res[MAXSZ];

static float          sa[MAXSZ], sb[MAXSZ], sc[MAXSZ];
static double         da[MAXSZ], db[MAXSZ], dc[MAXSZ];
static float complex  ca[MAXSZ], cb[MAXSZ], cc[MAXSZ];
static double complex za[MAXSZ], zb[MAXSZ], zc[MAXSZ];

/* Deterministic so a failure is reproducible from the seed alone. */
static uint32_t rng = 12345u;

static double
rnd(void)
{
    rng = rng * 1103515245u + 12345u;

    return (double)(rng >> 8) / (double)(1u << 24) * 2.0 - 1.0;
}

/*
 *  The reference: C <- alpha * op(A) * op(B) + beta * C0, unblocked
 *
 *  One reference in double complex covers all four types. Real inputs carry
 *  zero imaginary parts, and 0*0 and a*0 are exact, so the real result is
 *  bit-identical to what a real-only reference would produce.
 *
 *  It accumulates into a register and applies alpha once at the end, which is
 *  deliberately a different rounding order from the library. Agreement to the
 *  error bound is the claim; bit equality is not.
 */
static void
ref_gemm(int32_t m, int32_t n, int32_t k, double complex alpha,
        const double complex *a, ptrdiff_t ars, ptrdiff_t acs, int conja,
        const double complex *b, ptrdiff_t brs, ptrdiff_t bcs, int conjb,
        double complex beta, const double complex *c0, int32_t ldc0,
        double complex *r)
{
    double alr = creal(alpha), ali = cimag(alpha);
    double ber = creal(beta),  bei = cimag(beta);

    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            double sr = 0.0, si = 0.0;

            for (int32_t p = 0; p < k; ++p) {
                double complex az = a[(ptrdiff_t)i * ars + (ptrdiff_t)p * acs];
                double complex bz = b[(ptrdiff_t)p * brs + (ptrdiff_t)j * bcs];

                double ar = creal(az), ai = conja ? -cimag(az) : cimag(az);
                double br = creal(bz), bi = conjb ? -cimag(bz) : cimag(bz);

                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
            }

            {
                double complex z = c0[(ptrdiff_t)i * ldc0 + j];
                double zr = creal(z), zi = cimag(z);

                /* beta == 0 wins over whatever C held, NaN included */
                double cr = (ber == 0.0 && bei == 0.0)
                          ? 0.0 : ber * zr - bei * zi;
                double ci = (ber == 0.0 && bei == 0.0)
                          ? 0.0 : ber * zi + bei * zr;

                r[(ptrdiff_t)i * n + j] = (cr + (alr * sr - ali * si))
                                        + (ci + (alr * si + ali * sr)) * I;
            }
        }
    }
}

static int
cmp_block(const char *name, int32_t m, int32_t n, int32_t k,
        enum tinyblas_op ta, enum tinyblas_op tb, double tol)
{
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            double complex g = res[(ptrdiff_t)i * n + j];
            double complex w = ref[(ptrdiff_t)i * n + j];

            if (!(fabs(creal(g) - creal(w)) <= tol) ||
                !(fabs(cimag(g) - cimag(w)) <= tol)) {
                printf("FAIL: %s:%d  %s ta=%d tb=%d m=%d n=%d k=%d "
                       "at (%d,%d) got=(%g,%g) want=(%g,%g) tol=%g\n",
                       __FILE__, __LINE__, name, (int)ta, (int)tb,
                       m, n, k, i, j,
                       creal(g), cimag(g), creal(w), cimag(w), tol);

                return 1;
            }
        }
    }

    return 0;
}

/*
 *  Run one gemm shape through every type and compare against the reference.
 *
 *  cplx == 0 fills the imaginary parts with zero, which makes the case legal
 *  input for sgemm and dgemm as well; the complex routines still run on it,
 *  so real data costs nothing and gets four checks instead of two.
 *
 *  The padding columns of every matrix are poisoned with NaN. Any read outside
 *  the logical extent turns the result NaN and fails the comparison.
 */
static int
check_gemm(enum tinyblas_op ta, enum tinyblas_op tb,
        int32_t m, int32_t n, int32_t k,
        double complex alpha, double complex beta,
        int32_t pada, int32_t padb, int32_t padc, int cplx)
{
    int32_t arows = (ta == TINYBLAS_NONE) ? m : k;
    int32_t acols = (ta == TINYBLAS_NONE) ? k : m;
    int32_t brows = (tb == TINYBLAS_NONE) ? k : n;
    int32_t bcols = (tb == TINYBLAS_NONE) ? n : k;

    int32_t lda = acols + pada;
    int32_t ldb = bcols + padb;
    int32_t ldc = n + padc;

    int conja = (ta == TINYBLAS_CONJ_TRANS);
    int conjb = (tb == TINYBLAS_CONJ_TRANS);

    ptrdiff_t ars = (ta == TINYBLAS_NONE) ? (ptrdiff_t)lda : 1;
    ptrdiff_t acs = (ta == TINYBLAS_NONE) ? 1 : (ptrdiff_t)lda;
    ptrdiff_t brs = (tb == TINYBLAS_NONE) ? (ptrdiff_t)ldb : 1;
    ptrdiff_t bcs = (tb == TINYBLAS_NONE) ? 1 : (ptrdiff_t)ldb;

    double tol;

    assert((size_t)arows * (size_t)lda <= MAXSZ);
    assert((size_t)brows * (size_t)ldb <= MAXSZ);
    assert((size_t)m * (size_t)ldc <= MAXSZ);

    for (int32_t i = 0; i < MAXSZ; ++i)
        ma[i] = mb[i] = mc[i] = NAN + NAN * I;

    for (int32_t i = 0; i < arows; ++i)
        for (int32_t j = 0; j < acols; ++j)
            ma[(ptrdiff_t)i * lda + j] = rnd() + (cplx ? rnd() : 0.0) * I;

    for (int32_t i = 0; i < brows; ++i)
        for (int32_t j = 0; j < bcols; ++j)
            mb[(ptrdiff_t)i * ldb + j] = rnd() + (cplx ? rnd() : 0.0) * I;

    for (int32_t i = 0; i < m; ++i)
        for (int32_t j = 0; j < n; ++j)
            mc[(ptrdiff_t)i * ldc + j] = rnd() + (cplx ? rnd() : 0.0) * I;

    ref_gemm(m, n, k, alpha, ma, ars, acs, conja, mb, brs, bcs, conjb,
             beta, mc, ldc, ref);

    /* Inputs live in [-1, 1), so amax and bmax are 1 and drop out of the
     * usual k * eps * amax * bmax forward-error bound. */
    tol = 16.0 * (double)(k + 1) * (cabs(alpha) + cabs(beta) + 1.0);

    for (int32_t i = 0; i < MAXSZ; ++i) {
        sa[i] = (float)creal(ma[i]);   sb[i] = (float)creal(mb[i]);
        sc[i] = (float)creal(mc[i]);
        da[i] = creal(ma[i]);          db[i] = creal(mb[i]);
        dc[i] = creal(mc[i]);
        ca[i] = (float complex)ma[i];  cb[i] = (float complex)mb[i];
        cc[i] = (float complex)mc[i];
        za[i] = ma[i];                 zb[i] = mb[i];
        zc[i] = mc[i];
    }

    if (!cplx) {
        tinyblas_dgemm(ta, tb, m, n, k, creal(alpha), da, lda, db, ldb,
                       creal(beta), dc, ldc);

        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                res[(ptrdiff_t)i * n + j] = dc[(ptrdiff_t)i * ldc + j];

        if (cmp_block("dgemm", m, n, k, ta, tb, tol * DBL_EPSILON)) return 1;

        tinyblas_sgemm(ta, tb, m, n, k, (float)creal(alpha), sa, lda, sb, ldb,
                       (float)creal(beta), sc, ldc);

        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                res[(ptrdiff_t)i * n + j] = sc[(ptrdiff_t)i * ldc + j];

        if (cmp_block("sgemm", m, n, k, ta, tb, tol * FLT_EPSILON)) return 1;
    }

    tinyblas_zgemm(ta, tb, m, n, k, alpha, za, lda, zb, ldb, beta, zc, ldc);

    for (int32_t i = 0; i < m; ++i)
        for (int32_t j = 0; j < n; ++j)
            res[(ptrdiff_t)i * n + j] = zc[(ptrdiff_t)i * ldc + j];

    if (cmp_block("zgemm", m, n, k, ta, tb, tol * DBL_EPSILON)) return 1;

    tinyblas_cgemm(ta, tb, m, n, k, (float complex)alpha, ca, lda, cb, ldb,
                   (float complex)beta, cc, ldc);

    for (int32_t i = 0; i < m; ++i)
        for (int32_t j = 0; j < n; ++j)
            res[(ptrdiff_t)i * n + j] = cc[(ptrdiff_t)i * ldc + j];

    if (cmp_block("cgemm", m, n, k, ta, tb, tol * FLT_EPSILON)) return 1;

    return 0;
}

static double complex mdn[MAXSZ];

/* mirror the stored triangle of ma into a dense na by na square in mdn */
static void
dense_sym(enum tinyblas_uplo uplo, int32_t na, int32_t lda, int herm)
{
    for (int32_t i = 0; i < na; ++i) {
        for (int32_t j = 0; j < na; ++j) {
            int stored = (uplo == TINYBLAS_UPPER) ? (j >= i) : (j <= i);
            double complex v = stored ? ma[(ptrdiff_t)i * lda + j]
                                      : ma[(ptrdiff_t)j * lda + i];

            if (herm && !stored) v = conj(v);
            if (herm && i == j)  v = creal(v);

            mdn[(ptrdiff_t)i * na + j] = v;
        }
    }
}

/* write op(A) out densely, zeros outside the triangle */
static void
dense_tri(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t na, int32_t lda)
{
    for (int32_t i = 0; i < na; ++i) {
        for (int32_t j = 0; j < na; ++j) {
            int32_t r = (trans == TINYBLAS_NONE) ? i : j;
            int32_t c = (trans == TINYBLAS_NONE) ? j : i;
            int in = (uplo == TINYBLAS_UPPER) ? (c >= r) : (c <= r);
            double complex v = 0.0;

            if (in) {
                v = (r == c && diag == TINYBLAS_UNIT)
                  ? 1.0 : ma[(ptrdiff_t)r * lda + c];

                if (trans == TINYBLAS_CONJ_TRANS && !(r == c && diag == TINYBLAS_UNIT))
                    v = conj(v);
            }

            mdn[(ptrdiff_t)i * na + j] = v;
        }
    }
}

static void
fill_master(int32_t rows, int32_t cols, int32_t ld, double complex *dst, int cplx)
{
    for (int32_t i = 0; i < rows; ++i)
        for (int32_t j = 0; j < cols; ++j)
            dst[(ptrdiff_t)i * ld + j] = rnd() + (cplx ? rnd() : 0.0) * I;
}

/* copy the master buffers into every typed working array */
static void
seed_all(int32_t na, int32_t lda, int32_t bm, int32_t ldb, int32_t cm, int32_t ldc)
{
    for (int32_t i = 0; i < na * lda && i < MAXSZ; ++i) {
        sa[i] = (float)creal(ma[i]);  da[i] = creal(ma[i]);
        ca[i] = (float complex)ma[i]; za[i] = ma[i];
    }

    for (int32_t i = 0; i < bm * ldb && i < MAXSZ; ++i) {
        sb[i] = (float)creal(mb[i]);  db[i] = creal(mb[i]);
        cb[i] = (float complex)mb[i]; zb[i] = mb[i];
    }

    for (int32_t i = 0; i < cm * ldc && i < MAXSZ; ++i) {
        sc[i] = (float)creal(mc[i]);  dc[i] = creal(mc[i]);
        cc[i] = (float complex)mc[i]; zc[i] = mc[i];
    }
}

/* pull a typed result matrix into res for comparison */
static void
harvest(char type, int32_t m, int32_t n, int32_t ld)
{
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            ptrdiff_t o = (ptrdiff_t)i * ld + j;

            switch (type) {
            case 's': res[(ptrdiff_t)i * n + j] = sc[o]; break;
            case 'd': res[(ptrdiff_t)i * n + j] = dc[o]; break;
            case 'c': res[(ptrdiff_t)i * n + j] = cc[o]; break;
            default:  res[(ptrdiff_t)i * n + j] = zc[o]; break;
            }
        }
    }
}

static void
harvest_b(char type, int32_t m, int32_t n, int32_t ld)
{
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            ptrdiff_t o = (ptrdiff_t)i * ld + j;

            switch (type) {
            case 's': res[(ptrdiff_t)i * n + j] = sb[o]; break;
            case 'd': res[(ptrdiff_t)i * n + j] = db[o]; break;
            case 'c': res[(ptrdiff_t)i * n + j] = cb[o]; break;
            default:  res[(ptrdiff_t)i * n + j] = zb[o]; break;
            }
        }
    }
}

/* compare res against ref, optionally only inside one triangle */
static int
cmp_tri(const char *name, int32_t m, int32_t n, enum tinyblas_uplo uplo,
        int tri, double tol)
{
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            double complex g, w;

            if (tri && ((uplo == TINYBLAS_UPPER) ? (j < i) : (j > i))) continue;

            g = res[(ptrdiff_t)i * n + j];
            w = ref[(ptrdiff_t)i * n + j];

            if (!(fabs(creal(g) - creal(w)) <= tol) ||
                !(fabs(cimag(g) - cimag(w)) <= tol)) {
                printf("FAIL: %s:%d  %s at (%d,%d) got=(%g,%g) want=(%g,%g) "
                       "tol=%g\n", __FILE__, __LINE__, name, i, j,
                       creal(g), cimag(g), creal(w), cimag(w), tol);

                return 1;
            }
        }
    }

    return 0;
}

/*
 *  symm and hemm against a mirrored dense operand
 */
static int
check_symm(enum tinyblas_side side, enum tinyblas_uplo uplo, int herm,
        int32_t m, int32_t n, double complex alpha, double complex beta,
        int cplx)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    int32_t lda = na, ldb = n, ldc = n;
    double tol = 16.0 * (double)(na + 1) * (cabs(alpha) + cabs(beta) + 1.0);

    assert((size_t)na * (size_t)na <= MAXSZ);

    fill_master(na, na, lda, ma, cplx);
    fill_master(m, n, ldb, mb, cplx);
    fill_master(m, n, ldc, mc, cplx);

    /* a hermitian operand must start with a real diagonal */
    if (herm)
        for (int32_t i = 0; i < na; ++i)
            ma[(ptrdiff_t)i * lda + i] = creal(ma[(ptrdiff_t)i * lda + i]);

    dense_sym(uplo, na, lda, herm);

    if (side == TINYBLAS_LEFT)
        ref_gemm(m, n, na, alpha, mdn, na, 1, 0, mb, ldb, 1, 0,
                 beta, mc, ldc, ref);
    else
        ref_gemm(m, n, na, alpha, mb, ldb, 1, 0, mdn, na, 1, 0,
                 beta, mc, ldc, ref);

    seed_all(na, lda, m, ldb, m, ldc);

    /* the real routines only when the data has no imaginary part; the
     * complex ones always, since real data is legal complex input */
    if (!herm && !cplx) {
        tinyblas_dsymm(side, uplo, m, n, creal(alpha), da, lda, db, ldb,
                       creal(beta), dc, ldc);
        harvest('d', m, n, ldc);
        if (cmp_tri("dsymm", m, n, uplo, 0, tol * DBL_EPSILON)) return 1;

        tinyblas_ssymm(side, uplo, m, n, (float)creal(alpha), sa, lda,
                       sb, ldb, (float)creal(beta), sc, ldc);
        harvest('s', m, n, ldc);
        if (cmp_tri("ssymm", m, n, uplo, 0, tol * FLT_EPSILON)) return 1;
    }

    if (herm) {
        tinyblas_zhemm(side, uplo, m, n, alpha, za, lda, zb, ldb,
                       beta, zc, ldc);
        harvest('z', m, n, ldc);
        if (cmp_tri("zhemm", m, n, uplo, 0, tol * DBL_EPSILON)) return 1;

        tinyblas_chemm(side, uplo, m, n, (float complex)alpha, ca, lda,
                       cb, ldb, (float complex)beta, cc, ldc);
        harvest('c', m, n, ldc);
        if (cmp_tri("chemm", m, n, uplo, 0, tol * FLT_EPSILON)) return 1;
    } else {
        tinyblas_zsymm(side, uplo, m, n, alpha, za, lda, zb, ldb,
                       beta, zc, ldc);
        harvest('z', m, n, ldc);
        if (cmp_tri("zsymm", m, n, uplo, 0, tol * DBL_EPSILON)) return 1;

        tinyblas_csymm(side, uplo, m, n, (float complex)alpha, ca, lda,
                       cb, ldb, (float complex)beta, cc, ldc);
        harvest('c', m, n, ldc);
        if (cmp_tri("csymm", m, n, uplo, 0, tol * FLT_EPSILON)) return 1;
    }

    return 0;
}

/*
 *  syrk, syr2k, herk and her2k
 *
 *  rank2 adds the second pass; herm conjugates the second operand and pins
 *  the diagonal of C to the real axis, which is the whole difference between
 *  the symmetric and the hermitian forms.
 */
static int
check_syrk(enum tinyblas_uplo uplo, enum tinyblas_op trans, int rank2,
        int herm, int32_t n, int32_t k, double complex alpha,
        double complex beta, int cplx)
{
    int notrans = (trans == TINYBLAS_NONE);
    int32_t arows = notrans ? n : k;
    int32_t acols = notrans ? k : n;
    int32_t lda = acols, ldb = acols, ldc = n;
    ptrdiff_t rs1 = notrans ? lda : 1, cs1 = notrans ? 1 : lda;
    ptrdiff_t rs2 = notrans ? 1 : ldb, cs2 = notrans ? ldb : 1;
    double tol = 16.0 * (double)(k + 1) * (cabs(alpha) + cabs(beta) + 1.0);

    /* A^H is a conjugate on whichever operand carries the transpose */
    int cj1 = herm && !notrans;
    int cj2 = herm && notrans;

    fill_master(arows, acols, lda, ma, cplx);
    fill_master(arows, acols, ldb, mb, cplx);
    fill_master(n, n, ldc, mc, cplx);

    /* a hermitian C is defined to have a real diagonal on input too */
    if (herm)
        for (int32_t i = 0; i < n; ++i)
            mc[(ptrdiff_t)i * ldc + i] = creal(mc[(ptrdiff_t)i * ldc + i]);

    /* first term: alpha * op(A) * op(B or A)^T, or ^H when hermitian */
    ref_gemm(n, n, k, alpha, ma, rs1, cs1, cj1,
             rank2 ? mb : ma, rs2, cs2, cj2, beta, mc, ldc, ref);

    /* second term accumulates on top; ref is safe as its own beta operand
     * because each element is read before it is written. her2k conjugates
     * alpha there, which is what makes the sum hermitian. */
    if (rank2)
        ref_gemm(n, n, k, herm ? conj(alpha) : alpha, mb, rs1, cs1, cj1,
                 ma, rs2, cs2, cj2, 1.0, ref, n, ref);

    seed_all(arows, lda, arows, ldb, n, ldc);

    if (!herm && !cplx) {
        if (rank2)
            tinyblas_dsyr2k(uplo, trans, n, k, creal(alpha), da, lda, db, ldb,
                            creal(beta), dc, ldc);
        else
            tinyblas_dsyrk(uplo, trans, n, k, creal(alpha), da, lda,
                           creal(beta), dc, ldc);

        harvest('d', n, n, ldc);
        if (cmp_tri(rank2 ? "dsyr2k" : "dsyrk", n, n, uplo, 1,
                    tol * DBL_EPSILON)) return 1;

        if (rank2)
            tinyblas_ssyr2k(uplo, trans, n, k, (float)creal(alpha), sa, lda,
                            sb, ldb, (float)creal(beta), sc, ldc);
        else
            tinyblas_ssyrk(uplo, trans, n, k, (float)creal(alpha), sa, lda,
                           (float)creal(beta), sc, ldc);

        harvest('s', n, n, ldc);
        if (cmp_tri(rank2 ? "ssyr2k" : "ssyrk", n, n, uplo, 1,
                    tol * FLT_EPSILON)) return 1;
    }

    if (herm) {
        /* herk takes a real alpha; her2k takes a complex one and a real beta */
        if (rank2)
            tinyblas_zher2k(uplo, trans, n, k, alpha, za, lda, zb, ldb,
                            creal(beta), zc, ldc);
        else
            tinyblas_zherk(uplo, trans, n, k, creal(alpha), za, lda,
                           creal(beta), zc, ldc);

        harvest('z', n, n, ldc);
        if (cmp_tri(rank2 ? "zher2k" : "zherk", n, n, uplo, 1,
                    tol * DBL_EPSILON)) return 1;

        if (rank2)
            tinyblas_cher2k(uplo, trans, n, k, (float complex)alpha, ca, lda,
                            cb, ldb, (float)creal(beta), cc, ldc);
        else
            tinyblas_cherk(uplo, trans, n, k, (float)creal(alpha), ca, lda,
                           (float)creal(beta), cc, ldc);

        harvest('c', n, n, ldc);
        if (cmp_tri(rank2 ? "cher2k" : "cherk", n, n, uplo, 1,
                    tol * FLT_EPSILON)) return 1;
    } else {
        if (rank2)
            tinyblas_zsyr2k(uplo, trans, n, k, alpha, za, lda, zb, ldb,
                            beta, zc, ldc);
        else
            tinyblas_zsyrk(uplo, trans, n, k, alpha, za, lda, beta, zc, ldc);

        harvest('z', n, n, ldc);
        if (cmp_tri(rank2 ? "zsyr2k" : "zsyrk", n, n, uplo, 1,
                    tol * DBL_EPSILON)) return 1;

        if (rank2)
            tinyblas_csyr2k(uplo, trans, n, k, (float complex)alpha, ca, lda,
                            cb, ldb, (float complex)beta, cc, ldc);
        else
            tinyblas_csyrk(uplo, trans, n, k, (float complex)alpha, ca, lda,
                           (float complex)beta, cc, ldc);

        harvest('c', n, n, ldc);
        if (cmp_tri(rank2 ? "csyr2k" : "csyrk", n, n, uplo, 1,
                    tol * FLT_EPSILON)) return 1;
    }

    return 0;
}

/*
 *  trmm against a densely written op(A)
 */
static int
check_trmm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, double complex alpha, int cplx)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    int32_t lda = na, ldb = n;
    double tol = 16.0 * (double)(na + 1) * (cabs(alpha) + 1.0);

    fill_master(na, na, lda, ma, cplx);
    fill_master(m, n, ldb, mb, cplx);

    for (int32_t i = 0; i < MAXSZ; ++i) mc[i] = 0.0;

    dense_tri(uplo, trans, diag, na, lda);

    if (side == TINYBLAS_LEFT)
        ref_gemm(m, n, na, alpha, mdn, na, 1, 0, mb, ldb, 1, 0,
                 0.0, mc, n, ref);
    else
        ref_gemm(m, n, na, alpha, mb, ldb, 1, 0, mdn, na, 1, 0,
                 0.0, mc, n, ref);

    seed_all(na, lda, m, ldb, m, n);

    if (!cplx) {
        tinyblas_dtrmm(side, uplo, trans, diag, m, n, creal(alpha),
                       da, lda, db, ldb);
        harvest_b('d', m, n, ldb);
        if (cmp_tri("dtrmm", m, n, uplo, 0, tol * DBL_EPSILON)) return 1;

        tinyblas_strmm(side, uplo, trans, diag, m, n, (float)creal(alpha),
                       sa, lda, sb, ldb);
        harvest_b('s', m, n, ldb);
        if (cmp_tri("strmm", m, n, uplo, 0, tol * FLT_EPSILON)) return 1;
    }

    tinyblas_ztrmm(side, uplo, trans, diag, m, n, alpha, za, lda, zb, ldb);
    harvest_b('z', m, n, ldb);
    if (cmp_tri("ztrmm", m, n, uplo, 0, tol * DBL_EPSILON)) return 1;

    tinyblas_ctrmm(side, uplo, trans, diag, m, n, (float complex)alpha,
                   ca, lda, cb, ldb);
    harvest_b('c', m, n, ldb);
    if (cmp_tri("ctrmm", m, n, uplo, 0, tol * FLT_EPSILON)) return 1;

    return 0;
}

/*
 *  Multiply the solution in res back through op(A) in mdn and compare
 *  against alpha * B. Called once per type; mb and mdn survive each pass.
 */
static int
trsm_residual(const char *name, enum tinyblas_side side,
        int32_t m, int32_t n, int32_t na, int32_t ldb,
        double complex alpha, double tol)
{
    for (int32_t i = 0; i < m; ++i)
        for (int32_t j = 0; j < n; ++j)
            mc[(ptrdiff_t)i * n + j] = res[(ptrdiff_t)i * n + j];

    if (side == TINYBLAS_LEFT)
        ref_gemm(m, n, na, 1.0, mdn, na, 1, 0, mc, n, 1, 0, 0.0, mc, n, ref);
    else
        ref_gemm(m, n, na, 1.0, mc, n, 1, 0, mdn, na, 1, 0, 0.0, mc, n, ref);

    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            double complex want = alpha * mb[(ptrdiff_t)i * ldb + j];
            double complex got = ref[(ptrdiff_t)i * n + j];

            if (!(cabs(got - want) <= tol)) {
                printf("FAIL: %s:%d  %s residual side=%d m=%d n=%d at (%d,%d) "
                       "got=(%g,%g) want=(%g,%g) tol=%g\n", __FILE__, __LINE__,
                       name, (int)side, m, n, i, j,
                       creal(got), cimag(got), creal(want), cimag(want), tol);

                return 1;
            }
        }
    }

    return 0;
}

/*
 *  trsm checked by its own residual: op(A) * X must reproduce alpha * B
 */
static int
check_trsm(enum tinyblas_side side, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag,
        int32_t m, int32_t n, double complex alpha, int cplx)
{
    int32_t na = (side == TINYBLAS_LEFT) ? m : n;
    int32_t lda = na, ldb = n;
    double scale = 64.0 * (double)na * (cabs(alpha) + 1.0);
    double dtol = scale * DBL_EPSILON, stol = scale * FLT_EPSILON;
    double off = (diag == TINYBLAS_UNIT) ? 1.0 / (double)na : 1.0;

    /* Diagonally dominant, so the residual measures the solver rather than
     * the conditioning of a random triangle. The diagonal carries an
     * imaginary part too: a real one leaves the complex reciprocal in the
     * blocked solver untested, since the sign of its conjugate stops
     * mattering the moment the imaginary part is zero. */
    for (int32_t i = 0; i < na; ++i)
        for (int32_t j = 0; j < na; ++j)
            ma[(ptrdiff_t)i * lda + j] = (i == j)
                    ? ((double)na + rnd() + (cplx ? rnd() : 0.0) * I)
                    : off * (rnd() + (cplx ? rnd() : 0.0) * I);

    fill_master(m, n, ldb, mb, cplx);

    for (int32_t i = 0; i < MAXSZ; ++i) mc[i] = 0.0;

    dense_tri(uplo, trans, diag, na, lda);

    seed_all(na, lda, m, ldb, m, n);

    if (!cplx) {
        tinyblas_dtrsm(side, uplo, trans, diag, m, n, creal(alpha),
                       da, lda, db, ldb);
        harvest_b('d', m, n, ldb);
        if (trsm_residual("dtrsm", side, m, n, na, ldb, alpha, dtol)) return 1;

        tinyblas_strsm(side, uplo, trans, diag, m, n, (float)creal(alpha),
                       sa, lda, sb, ldb);
        harvest_b('s', m, n, ldb);
        if (trsm_residual("strsm", side, m, n, na, ldb, alpha, stol)) return 1;
    }

    tinyblas_ztrsm(side, uplo, trans, diag, m, n, alpha, za, lda, zb, ldb);
    harvest_b('z', m, n, ldb);
    if (trsm_residual("ztrsm", side, m, n, na, ldb, alpha, dtol)) return 1;

    tinyblas_ctrsm(side, uplo, trans, diag, m, n, (float complex)alpha,
                   ca, lda, cb, ldb);
    harvest_b('c', m, n, ldb);
    if (trsm_residual("ctrsm", side, m, n, na, ldb, alpha, stol)) return 1;

    return 0;
}

int main(void) {
    static const enum tinyblas_op tr[3] = {
        TINYBLAS_NONE, TINYBLAS_TRANS, TINYBLAS_CONJ_TRANS
    };

    /* Every one of these is a non-multiple of 6, 8 or 16, which is where the
     * packed path's edge handling will have to earn its keep. */
    static const int32_t dims[] = {1, 2, 3, 5, 7, 8, 13, 16, 17, 31, 40, 65};
    static const int32_t ndims  = (int32_t)(sizeof dims / sizeof dims[0]);

    /* --- the reference, pinned by hand ---------------------------------- */
    {
        double complex a[4] = {1.0, 2.0, 3.0, 4.0};      /* [[1,2],[3,4]] */
        double complex b[4] = {5.0, 6.0, 7.0, 8.0};      /* [[5,6],[7,8]] */
        double complex z[4] = {0.0, 0.0, 0.0, 0.0};
        double complex r[6];

        /* A*B = [[19,22],[43,50]] */
        ref_gemm(2, 2, 2, 1.0, a, 2, 1, 0, b, 2, 1, 0, 0.0, z, 2, r);
        CHECK_C(r[0], 19.0, 0.0, 0.0);  CHECK_C(r[1], 22.0, 0.0, 0.0);
        CHECK_C(r[2], 43.0, 0.0, 0.0);  CHECK_C(r[3], 50.0, 0.0, 0.0);

        /* A^T*B = [[26,30],[38,44]] */
        ref_gemm(2, 2, 2, 1.0, a, 1, 2, 0, b, 2, 1, 0, 0.0, z, 2, r);
        CHECK_C(r[0], 26.0, 0.0, 0.0);  CHECK_C(r[1], 30.0, 0.0, 0.0);
        CHECK_C(r[2], 38.0, 0.0, 0.0);  CHECK_C(r[3], 44.0, 0.0, 0.0);

        /* A*B^T = [[17,23],[39,53]] */
        ref_gemm(2, 2, 2, 1.0, a, 2, 1, 0, b, 1, 2, 0, 0.0, z, 2, r);
        CHECK_C(r[0], 17.0, 0.0, 0.0);  CHECK_C(r[1], 23.0, 0.0, 0.0);
        CHECK_C(r[2], 39.0, 0.0, 0.0);  CHECK_C(r[3], 53.0, 0.0, 0.0);

        /* a 2x2 by 2x3, so a non-square n reaches the right elements */
        {
            double complex b3[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
            double complex z3[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            ref_gemm(2, 3, 2, 1.0, a, 2, 1, 0, b3, 3, 1, 0, 0.0, z3, 3, r);
            CHECK_C(r[0],  9.0, 0.0, 0.0);
            CHECK_C(r[1], 12.0, 0.0, 0.0);
            CHECK_C(r[2], 15.0, 0.0, 0.0);
            CHECK_C(r[3], 19.0, 0.0, 0.0);
            CHECK_C(r[4], 26.0, 0.0, 0.0);
            CHECK_C(r[5], 33.0, 0.0, 0.0);
        }

        /* alpha and beta both live: 2*(A*B) + 3*C */
        {
            double complex c1[4] = {1.0, 1.0, 1.0, 1.0};

            ref_gemm(2, 2, 2, 2.0, a, 2, 1, 0, b, 2, 1, 0, 3.0, c1, 2, r);
            CHECK_C(r[0], 41.0, 0.0, 0.0);   /* 2*19 + 3 */
            CHECK_C(r[3], 103.0, 0.0, 0.0);  /* 2*50 + 3 */
        }

        /* (1+i)(2+3i) = -1+5i, and conj(1+i)(2+3i) = 5+i */
        {
            double complex ac[1] = {1.0 + 1.0 * I};
            double complex bc[1] = {2.0 + 3.0 * I};
            double complex zc1[1] = {0.0};

            ref_gemm(1, 1, 1, 1.0, ac, 1, 1, 0, bc, 1, 1, 0, 0.0, zc1, 1, r);
            CHECK_C(r[0], -1.0, 5.0, 1e-12);

            ref_gemm(1, 1, 1, 1.0, ac, 1, 1, 1, bc, 1, 1, 0, 0.0, zc1, 1, r);
            CHECK_C(r[0], 5.0, 1.0, 1e-12);
        }
    }

    /* --- the library, pinned by the same hand-computed values ----------- */
    {
        double dA[4] = {1.0, 2.0, 3.0, 4.0};
        double dB[4] = {5.0, 6.0, 7.0, 8.0};
        double dC[4] = {0.0, 0.0, 0.0, 0.0};

        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_NONE, 2, 2, 2,
                       1.0, dA, 2, dB, 2, 0.0, dC, 2);
        CHECK(dC[0], 19.0, 1e-12);  CHECK(dC[1], 22.0, 1e-12);
        CHECK(dC[2], 43.0, 1e-12);  CHECK(dC[3], 50.0, 1e-12);

        tinyblas_dgemm(TINYBLAS_TRANS, TINYBLAS_NONE, 2, 2, 2,
                       1.0, dA, 2, dB, 2, 0.0, dC, 2);
        CHECK(dC[0], 26.0, 1e-12);  CHECK(dC[3], 44.0, 1e-12);

        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_TRANS, 2, 2, 2,
                       1.0, dA, 2, dB, 2, 0.0, dC, 2);
        CHECK(dC[0], 17.0, 1e-12);  CHECK(dC[3], 53.0, 1e-12);
    }
    {
        double complex zA[1] = {1.0 + 1.0 * I};
        double complex zB[1] = {2.0 + 3.0 * I};
        double complex zC[1] = {0.0};

        tinyblas_zgemm(TINYBLAS_NONE, TINYBLAS_NONE, 1, 1, 1,
                       1.0, zA, 1, zB, 1, 0.0, zC, 1);
        CHECK_C(zC[0], -1.0, 5.0, 1e-12);

        tinyblas_zgemm(TINYBLAS_CONJ_TRANS, TINYBLAS_NONE, 1, 1, 1,
                       1.0, zA, 1, zB, 1, 0.0, zC, 1);
        CHECK_C(zC[0], 5.0, 1.0, 1e-12);
    }

    /* --- square shape sweep, every trans pair, all four types ----------- */
    for (int32_t d = 0; d < ndims; ++d) {
        for (int32_t ia = 0; ia < 3; ++ia) {
            for (int32_t ib = 0; ib < 3; ++ib) {
                int32_t s = dims[d];

                if (check_gemm(tr[ia], tr[ib], s, s, s,
                               0.7 + 0.3 * I, -0.3 + 0.5 * I, 0, 0, 0, 1))
                    return 1;
            }
        }
    }

    /* --- mismatched m, n, k, plus padded leading dimensions ------------- */
    for (int32_t d = 0; d < ndims; ++d) {
        int32_t m = dims[d];
        int32_t n = dims[(d + 1) % ndims];
        int32_t k = dims[(d + 2) % ndims];

        for (int32_t ia = 0; ia < 3; ++ia) {
            for (int32_t ib = 0; ib < 3; ++ib) {
                if (check_gemm(tr[ia], tr[ib], m, n, k,
                               0.7 + 0.3 * I, -0.3 + 0.5 * I, 0, 0, 0, 1))
                    return 1;

                if (check_gemm(tr[ia], tr[ib], m, n, k,
                               1.0, 1.0, 3, 5, 2, 1))
                    return 1;
            }
        }
    }

    /* real data through every path, including the complex routines */
    for (int32_t d = 0; d < ndims; ++d) {
        int32_t s = dims[d];

        if (check_gemm(TINYBLAS_NONE, TINYBLAS_NONE, s, s, s,
                       0.7, -0.3, 0, 0, 0, 0))
            return 1;

        if (check_gemm(TINYBLAS_TRANS, TINYBLAS_NONE, s, s, s,
                       1.0, 0.0, 2, 0, 4, 0))
            return 1;
    }

    /* k deeper than one KC panel, so the k loop runs more than once and beta
     * is applied on the first panel only */
    if (check_gemm(TINYBLAS_NONE, TINYBLAS_NONE, 8, 8, 300,
                   0.7 + 0.3 * I, -0.3 + 0.5 * I, 0, 0, 0, 1))
        return 1;

    if (check_gemm(TINYBLAS_TRANS, TINYBLAS_TRANS, 7, 9, 260,
                   0.7, -0.3, 0, 0, 0, 0))
        return 1;

    /* both operands conjugated and k deeper than one KC panel: the complex
     * kernel folds the conjugate in while packing, so the two interact */
    if (check_gemm(TINYBLAS_CONJ_TRANS, TINYBLAS_CONJ_TRANS, 9, 11, 300,
                   0.7 + 0.3 * I, -0.3 + 0.5 * I, 0, 0, 0, 1))
        return 1;

    /* --- semantics ------------------------------------------------------ */
    {
        double a[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        double b[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
        double c[9];

        /* beta == 0 must overwrite C, never read it. A NaN C is legal input. */
        for (int32_t i = 0; i < 9; ++i) c[i] = NAN;

        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_NONE, 3, 3, 3,
                       1.0, a, 3, b, 3, 0.0, c, 3);

        for (int32_t i = 0; i < 9; ++i)
            CHECK(isfinite(c[i]) ? 0.0 : 1.0, 0.0, 0.0);

        CHECK(c[0], 30.0, 1e-12);   /* 1*9 + 2*6 + 3*3 */

        /* alpha == 0 leaves exactly beta * C, and must not read A or B */
        for (int32_t i = 0; i < 9; ++i) c[i] = 2.0;

        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_NONE, 3, 3, 3,
                       0.0, NULL, 3, NULL, 3, 3.0, c, 3);

        for (int32_t i = 0; i < 9; ++i) CHECK(c[i], 6.0, 0.0);

        /* k == 0 still applies beta: C := beta*C, not a no-op */
        for (int32_t i = 0; i < 9; ++i) c[i] = 2.0;

        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_NONE, 3, 3, 0,
                       1.0, NULL, 1, NULL, 1, 3.0, c, 3);

        for (int32_t i = 0; i < 9; ++i) CHECK(c[i], 6.0, 0.0);

        /* k == 0 with beta == 0 zeroes C rather than leaving it NaN */
        for (int32_t i = 0; i < 9; ++i) c[i] = NAN;

        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_NONE, 3, 3, 0,
                       1.0, NULL, 1, NULL, 1, 0.0, c, 3);

        for (int32_t i = 0; i < 9; ++i) CHECK(c[i], 0.0, 0.0);

        /* m == 0 and n == 0 are genuine no-ops that touch nothing */
        for (int32_t i = 0; i < 9; ++i) c[i] = 5.0;

        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_NONE, 0, 3, 3,
                       1.0, a, 3, b, 3, 0.0, c, 3);
        tinyblas_dgemm(TINYBLAS_NONE, TINYBLAS_NONE, 3, 0, 3,
                       1.0, a, 3, b, 3, 0.0, c, 3);

        for (int32_t i = 0; i < 9; ++i) CHECK(c[i], 5.0, 0.0);
    }

    /* --- the derived level 3 routines ----------------------------------- */
    {
        static const enum tinyblas_side sd[2] = { TINYBLAS_LEFT, TINYBLAS_RIGHT };
        static const enum tinyblas_uplo ul[2] = { TINYBLAS_UPPER, TINYBLAS_LOWER };
        static const enum tinyblas_diag dg[2] = { TINYBLAS_NON_UNIT, TINYBLAS_UNIT };
        static const int32_t sm[] = {1, 2, 3, 5, 8, 13, 17, 31, 40, 65, 70};
        static const int32_t nsm = (int32_t)(sizeof sm / sizeof sm[0]);

        for (int32_t d = 0; d < nsm; ++d) {
            int32_t p = sm[d];
            int32_t q = sm[(d + 1) % nsm];

            /* x == 0 is real data, which the complex routines must also
             * handle; x == 1 has imaginary parts and skips s and d */
            for (int32_t x = 0; x < 2; ++x) {
                double complex al = x ? 0.7 + 0.4 * I : 0.7;

                for (int32_t s = 0; s < 2; ++s) {
                    for (int32_t u = 0; u < 2; ++u) {
                        if (check_symm(sd[s], ul[u], 0, p, q, al, -0.3, x))
                            return 1;

                        if (check_symm(sd[s], ul[u], 1, p, q, al, -0.3, x))
                            return 1;

                        if (check_symm(sd[s], ul[u], 0, p, q, 1.0, 0.0, x))
                            return 1;

                        for (int32_t t = 0; t < 3; ++t) {
                            for (int32_t g = 0; g < 2; ++g) {
                                if (check_trmm(sd[s], ul[u], tr[t], dg[g],
                                               p, q, al, x)) return 1;

                                if (check_trsm(sd[s], ul[u], tr[t], dg[g],
                                               p, q, al, x)) return 1;
                            }
                        }
                    }
                }

                for (int32_t u = 0; u < 2; ++u) {
                    for (int32_t t = 0; t < 2; ++t) {
                        /* syrk and syr2k take NO_TRANS or TRANS */
                        if (check_syrk(ul[u], tr[t], 0, 0, p, q, al, -0.3, x))
                            return 1;

                        if (check_syrk(ul[u], tr[t], 1, 0, p, q, al, -0.3, x))
                            return 1;

                        if (check_syrk(ul[u], tr[t], 0, 0, p, q, 1.0, 0.0, x))
                            return 1;

                        /* herk and her2k take NO_TRANS or CONJ_TRANS, and a
                         * real alpha on the rank-k form */
                        {
                            enum tinyblas_op ht = t ? TINYBLAS_CONJ_TRANS
                                                       : TINYBLAS_NONE;

                            if (check_syrk(ul[u], ht, 0, 1, p, q, 0.7, -0.3, x))
                                return 1;

                            if (check_syrk(ul[u], ht, 1, 1, p, q, al, -0.3, x))
                                return 1;
                        }
                    }
                }
            }
        }

        /* the untouched triangle of a syrk target must survive untouched */
        {
            double c[9];
            double a[3] = {1.0, 2.0, 3.0};

            for (int32_t i = 0; i < 9; ++i) c[i] = 5.0;

            tinyblas_dsyrk(TINYBLAS_UPPER, TINYBLAS_NONE, 3, 1,
                           1.0, a, 1, 0.0, c, 3);

            CHECK(c[0], 1.0, 1e-12);    /* 1*1 */
            CHECK(c[1], 2.0, 1e-12);    /* 1*2 */
            CHECK(c[3], 5.0, 0.0);      /* strictly lower: untouched */
            CHECK(c[8], 9.0, 1e-12);    /* 3*3 */
        }

        /* trsm's blocked path only engages above TRSM_MB rows, which the
         * shape sweep stays under. These run several row blocks with a
         * partial one at the end, in both sweep directions. */
        for (int32_t x = 0; x < 2; ++x) {
            if (check_trsm(TINYBLAS_LEFT, TINYBLAS_UPPER, TINYBLAS_NONE,
                           TINYBLAS_NON_UNIT, 200, 3, 0.7, x)) return 1;

            if (check_trsm(TINYBLAS_LEFT, TINYBLAS_LOWER, TINYBLAS_NONE,
                           TINYBLAS_NON_UNIT, 140, 5, 0.7, x)) return 1;

            if (check_trsm(TINYBLAS_LEFT, TINYBLAS_UPPER, TINYBLAS_CONJ_TRANS,
                           TINYBLAS_UNIT, 140, 5, 0.7, x)) return 1;

            if (check_trsm(TINYBLAS_LEFT, TINYBLAS_LOWER, TINYBLAS_TRANS,
                           TINYBLAS_UNIT, 129, 4, 0.7, x)) return 1;

            /* side RIGHT is still the trsv sweep; keep it honest anyway */
            if (check_trsm(TINYBLAS_RIGHT, TINYBLAS_UPPER, TINYBLAS_TRANS,
                           TINYBLAS_NON_UNIT, 3, 140, 0.7, x)) return 1;
        }

        /* alpha == 0 on trmm zeroes B, matching the reference semantics */
        {
            double a[4] = {1, 2, 0, 3};
            double b[4] = {7, 7, 7, 7};

            tinyblas_dtrmm(TINYBLAS_LEFT, TINYBLAS_UPPER, TINYBLAS_NONE,
                           TINYBLAS_NON_UNIT, 2, 2, 0.0, a, 2, b, 2);

            for (int32_t i = 0; i < 4; ++i) CHECK(b[i], 0.0, 0.0);
        }

        /* A hermitian result has an exactly real diagonal, not a nearly real
         * one. The error bound of the shape sweep is too loose to see the
         * difference, so it is pinned here. */
        {
            double complex a2[4] = { 1.0 + 2.0 * I,  3.0 - 1.0 * I,
                                    -2.0 + 0.5 * I,  1.0 + 1.0 * I };
            double complex c2[4];
            float complex  a4[4], c4[4];

            for (int32_t i = 0; i < 4; ++i) {
                c2[i] = 7.0 + 3.0 * I;
                a4[i] = (float complex)a2[i];
                c4[i] = 7.0f + 3.0f * I;
            }

            tinyblas_zherk(TINYBLAS_UPPER, TINYBLAS_NONE, 2, 2,
                           0.5, a2, 2, 0.25, c2, 2);
            tinyblas_cherk(TINYBLAS_UPPER, TINYBLAS_NONE, 2, 2,
                           0.5f, a4, 2, 0.25f, c4, 2);

            CHECK(cimag(c2[0]), 0.0, 0.0);
            CHECK(cimag(c2[3]), 0.0, 0.0);
            CHECK(cimagf(c4[0]), 0.0, 0.0);
            CHECK(cimagf(c4[3]), 0.0, 0.0);

            for (int32_t i = 0; i < 4; ++i) c2[i] = 7.0 + 3.0 * I;

            tinyblas_zher2k(TINYBLAS_LOWER, TINYBLAS_CONJ_TRANS, 2, 2,
                            0.5 + 0.5 * I, a2, 2, a2, 2, 0.25, c2, 2);

            CHECK(cimag(c2[0]), 0.0, 0.0);
            CHECK(cimag(c2[3]), 0.0, 0.0);
        }

        /* alpha == 0 on the complex symm and hemm leaves exactly beta * C and
         * must not read A, and beta == 0 must overwrite a NaN C */
        {
            double complex b2[4] = {1.0, 1.0, 1.0, 1.0};
            double complex c2[4];

            for (int32_t i = 0; i < 4; ++i) c2[i] = 2.0 + 4.0 * I;

            tinyblas_zsymm(TINYBLAS_LEFT, TINYBLAS_UPPER, 2, 2, 0.0,
                           NULL, 2, b2, 2, 0.5, c2, 2);

            for (int32_t i = 0; i < 4; ++i) CHECK_C(c2[i], 1.0, 2.0, 0.0);

            for (int32_t i = 0; i < 4; ++i) c2[i] = 2.0 + 4.0 * I;

            tinyblas_zhemm(TINYBLAS_RIGHT, TINYBLAS_LOWER, 2, 2, 0.0,
                           NULL, 2, b2, 2, 0.0, c2, 2);

            for (int32_t i = 0; i < 4; ++i) CHECK_C(c2[i], 0.0, 0.0, 0.0);

            {
                float complex b4[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                float complex c4[4];

                for (int32_t i = 0; i < 4; ++i) c4[i] = 2.0f + 4.0f * I;

                tinyblas_csymm(TINYBLAS_LEFT, TINYBLAS_UPPER, 2, 2, 0.0f,
                               NULL, 2, b4, 2, 0.5f, c4, 2);

                for (int32_t i = 0; i < 4; ++i) CHECK_C(c4[i], 1.0, 2.0, 0.0);

                for (int32_t i = 0; i < 4; ++i) c4[i] = 2.0f + 4.0f * I;

                tinyblas_chemm(TINYBLAS_RIGHT, TINYBLAS_LOWER, 2, 2, 0.0f,
                               NULL, 2, b4, 2, 0.0f, c4, 2);

                for (int32_t i = 0; i < 4; ++i) CHECK_C(c4[i], 0.0, 0.0, 0.0);
            }

            for (int32_t i = 0; i < 4; ++i) c2[i] = NAN + NAN * I;

            tinyblas_zhemm(TINYBLAS_LEFT, TINYBLAS_UPPER, 2, 2, 1.0,
                           b2, 2, b2, 2, 0.0, c2, 2);

            for (int32_t i = 0; i < 4; ++i)
                CHECK(isfinite(creal(c2[i])) && isfinite(cimag(c2[i]))
                      ? 0.0 : 1.0, 0.0, 0.0);

            /* the same for a rank-k update, which scales C on its own */
            for (int32_t i = 0; i < 4; ++i) c2[i] = NAN + NAN * I;

            tinyblas_zherk(TINYBLAS_LOWER, TINYBLAS_NONE, 2, 2,
                           1.0, b2, 2, 0.0, c2, 2);

            CHECK(isfinite(creal(c2[0])) && isfinite(cimag(c2[0]))
                  ? 0.0 : 1.0, 0.0, 0.0);
            CHECK(isfinite(creal(c2[2])) && isfinite(cimag(c2[2]))
                  ? 0.0 : 1.0, 0.0, 0.0);
        }
    }

    printf("level3: all tests passed\n");

    return 0;
}
