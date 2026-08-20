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

#define MAXN 48
#define MAXA (MAXN * MAXN)
#define MAXV (MAXN * 4)          /* room for an increment of up to 4 */

/* master data, and the dense operator the routine ought to be applying */
static double complex ma[MAXA], mop[MAXA];
static double complex mx[MAXN], my[MAXN], mref[MAXN];

static float          sa[MAXA], sx[MAXV], sy[MAXV];
static double         da[MAXA], dx[MAXV], dy[MAXV];
static float complex  ca[MAXA], cx[MAXV], cy[MAXV];
static double complex za[MAXA], zx[MAXV], zy[MAXV];

static uint32_t rng = 987654321u;

static double
rnd(void)
{
    rng = rng * 1103515245u + 12345u;

    return (double)(rng >> 8) / (double)(1u << 24) * 2.0 - 1.0;
}

/* the level 1 increment convention: a negative increment walks backwards */
static ptrdiff_t
vidx(int32_t inc, int32_t len, int32_t i)
{
    return (inc < 0) ? (ptrdiff_t)(1 - len) * inc + (ptrdiff_t)i * inc
                     : (ptrdiff_t)i * inc;
}

/*
 *  Build the dense operator each routine is supposed to apply
 *
 *  Once the triangle is mirrored, or the transpose taken, or the unreferenced
 *  half zeroed, every one of gemv, symv, hemv and trmv is just a dense
 *  matrix-vector product, so they can share a single reference.
 */
static void
build_gen(int32_t m, int32_t n, int32_t lda, enum tinyblas_op trans)
{
    int32_t rows = (trans == TINYBLAS_NONE) ? m : n;
    int32_t cols = (trans == TINYBLAS_NONE) ? n : m;

    for (int32_t i = 0; i < rows; ++i) {
        for (int32_t j = 0; j < cols; ++j) {
            double complex v = (trans == TINYBLAS_NONE)
                             ? ma[(ptrdiff_t)i * lda + j]
                             : ma[(ptrdiff_t)j * lda + i];

            if (trans == TINYBLAS_CONJ_TRANS) v = conj(v);

            mop[(ptrdiff_t)i * cols + j] = v;
        }
    }
}

static void
build_sym(int32_t n, int32_t lda, enum tinyblas_uplo uplo, int herm)
{
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            int stored = (uplo == TINYBLAS_UPPER) ? (j >= i) : (j <= i);
            double complex v = stored ? ma[(ptrdiff_t)i * lda + j]
                                      : ma[(ptrdiff_t)j * lda + i];

            if (herm && !stored) v = conj(v);
            if (herm && i == j)  v = creal(v);

            mop[(ptrdiff_t)i * n + j] = v;
        }
    }
}

static void
build_tri(int32_t n, int32_t lda, enum tinyblas_uplo uplo,
        enum tinyblas_op trans, enum tinyblas_diag diag)
{
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            int32_t r = (trans == TINYBLAS_NONE) ? i : j;
            int32_t c = (trans == TINYBLAS_NONE) ? j : i;
            int in = (uplo == TINYBLAS_UPPER) ? (c >= r) : (c <= r);
            double complex v = 0.0;

            if (in) {
                v = ma[(ptrdiff_t)r * lda + c];

                if (trans == TINYBLAS_CONJ_TRANS) v = conj(v);
                if (r == c && diag == TINYBLAS_UNIT) v = 1.0;
            }

            mop[(ptrdiff_t)i * n + j] = v;
        }
    }
}

/* mref <- alpha * mop * mx + beta * my */
static void
ref_matvec(int32_t rows, int32_t cols, double complex alpha, double complex beta)
{
    for (int32_t i = 0; i < rows; ++i) {
        double sr = 0.0, si = 0.0;

        for (int32_t j = 0; j < cols; ++j) {
            double complex v = mop[(ptrdiff_t)i * cols + j];
            double vr = creal(v), vi = cimag(v);
            double xr = creal(mx[j]), xi = cimag(mx[j]);

            sr += vr * xr - vi * xi;
            si += vr * xi + vi * xr;
        }

        {
            double br = creal(beta), bi = cimag(beta);
            double yr = creal(my[i]), yi = cimag(my[i]);
            double cr = (beta == 0.0) ? 0.0 : br * yr - bi * yi;
            double ci = (beta == 0.0) ? 0.0 : br * yi + bi * yr;
            double ar = creal(alpha), ai = cimag(alpha);

            mref[i] = (cr + (ar * sr - ai * si))
                    + (ci + (ar * si + ai * sr)) * I;
        }
    }
}

static int
cmp_vec(const char *name, int32_t len, double tol)
{
    for (int32_t i = 0; i < len; ++i) {
        double complex g = my[i], w = mref[i];

        if (!(fabs(creal(g) - creal(w)) <= tol) ||
            !(fabs(cimag(g) - cimag(w)) <= tol)) {
            printf("FAIL: %s:%d  %s at %d got=(%g,%g) want=(%g,%g) tol=%g\n",
                   __FILE__, __LINE__, name, i,
                   creal(g), cimag(g), creal(w), cimag(w), tol);

            return 1;
        }
    }

    return 0;
}

/*
 *  Drive one shape of gemv, symv/hemv or trmv through every type
 *
 *  kind is 'g' for gemv, 's' for the symmetric and hermitian pair, and 't' for
 *  trmv, which writes its result back over x rather than into y.
 */
static int
check_matvec(char kind, enum tinyblas_op trans, enum tinyblas_uplo uplo,
        enum tinyblas_diag diag, int32_t m, int32_t n, int32_t lda,
        double complex alpha, double complex beta,
        int32_t incx, int32_t incy, int cplx)
{
    int32_t arows = (kind == 'g') ? m : n;
    int32_t rows, cols;
    double tol;

    if (kind == 'g') {
        rows = (trans == TINYBLAS_NONE) ? m : n;
        cols = (trans == TINYBLAS_NONE) ? n : m;
    } else {
        rows = cols = n;
    }

    assert((size_t)arows * (size_t)lda <= MAXA);
    assert((size_t)rows * (size_t)cols <= MAXA);

    for (int32_t i = 0; i < MAXA; ++i) ma[i] = NAN + NAN * I;

    for (int32_t i = 0; i < arows; ++i)
        for (int32_t j = 0; j < ((kind == 'g') ? n : n); ++j)
            ma[(ptrdiff_t)i * lda + j] = rnd() + (cplx ? rnd() : 0.0) * I;

    for (int32_t j = 0; j < cols; ++j) mx[j] = rnd() + (cplx ? rnd() : 0.0) * I;
    for (int32_t i = 0; i < rows; ++i) my[i] = rnd() + (cplx ? rnd() : 0.0) * I;

    if (kind == 'g')      build_gen(m, n, lda, trans);
    else if (kind == 's') build_sym(n, lda, uplo, cplx);
    else                  build_tri(n, lda, uplo, trans, diag);

    /* trmv overwrites x, so its "previous y" is x and there is no beta */
    if (kind == 't')
        for (int32_t i = 0; i < rows; ++i) my[i] = mx[i];

    ref_matvec(rows, cols, alpha, beta);

    tol = 16.0 * (double)(cols + 1) * (cabs(alpha) + cabs(beta) + 1.0);

    for (int32_t i = 0; i < MAXA; ++i) {
        sa[i] = (float)creal(ma[i]);   da[i] = creal(ma[i]);
        ca[i] = (float complex)ma[i];  za[i] = ma[i];
    }

    for (int32_t i = 0; i < MAXV; ++i) {
        sx[i] = sy[i] = 0.0f;  dx[i] = dy[i] = 0.0;
        cx[i] = cy[i] = 0.0f;  zx[i] = zy[i] = 0.0;
    }

    for (int32_t j = 0; j < cols; ++j) {
        ptrdiff_t o = vidx(incx, cols, j);

        sx[o] = (float)creal(mx[j]);   dx[o] = creal(mx[j]);
        cx[o] = (float complex)mx[j];  zx[o] = mx[j];
    }

    for (int32_t i = 0; i < rows; ++i) {
        ptrdiff_t o = vidx(incy, rows, i);

        sy[o] = (float)creal(my[i]);   dy[o] = creal(my[i]);
        cy[o] = (float complex)my[i];  zy[o] = my[i];
    }

#define HARVEST(buf, len, inc) do { \
    for (int32_t q = 0; q < (len); ++q) \
        my[q] = (buf)[vidx((inc), (len), q)]; \
} while (0)

    if (!cplx && kind != 's') {
        if (kind == 'g')
            tinyblas_dgemv(trans, m, n, creal(alpha), da, lda, dx, incx,
                           creal(beta), dy, incy);
        else
            tinyblas_dtrmv(uplo, trans, diag, n, da, lda, dx, incx);

        HARVEST(kind == 't' ? dx : dy, rows, kind == 't' ? incx : incy);
        if (cmp_vec("double", rows, tol * DBL_EPSILON)) return 1;

        if (kind == 'g')
            tinyblas_sgemv(trans, m, n, (float)creal(alpha), sa, lda, sx, incx,
                           (float)creal(beta), sy, incy);
        else
            tinyblas_strmv(uplo, trans, diag, n, sa, lda, sx, incx);

        HARVEST(kind == 't' ? sx : sy, rows, kind == 't' ? incx : incy);
        if (cmp_vec("float", rows, tol * FLT_EPSILON)) return 1;
    }

    if (!cplx && kind == 's') {
        tinyblas_dsymv(uplo, n, creal(alpha), da, lda, dx, incx,
                       creal(beta), dy, incy);
        HARVEST(dy, rows, incy);
        if (cmp_vec("dsymv", rows, tol * DBL_EPSILON)) return 1;

        tinyblas_ssymv(uplo, n, (float)creal(alpha), sa, lda, sx, incx,
                       (float)creal(beta), sy, incy);
        HARVEST(sy, rows, incy);
        if (cmp_vec("ssymv", rows, tol * FLT_EPSILON)) return 1;
    }

    if (cplx) {
        if (kind == 'g')
            tinyblas_zgemv(trans, m, n, alpha, za, lda, zx, incx, beta, zy, incy);
        else if (kind == 's')
            tinyblas_zhemv(uplo, n, alpha, za, lda, zx, incx, beta, zy, incy);
        else
            tinyblas_ztrmv(uplo, trans, diag, n, za, lda, zx, incx);

        HARVEST(kind == 't' ? zx : zy, rows, kind == 't' ? incx : incy);
        if (cmp_vec("zcomplex", rows, tol * DBL_EPSILON)) return 1;

        if (kind == 'g')
            tinyblas_cgemv(trans, m, n, (float complex)alpha, ca, lda, cx, incx,
                           (float complex)beta, cy, incy);
        else if (kind == 's')
            tinyblas_chemv(uplo, n, (float complex)alpha, ca, lda, cx, incx,
                           (float complex)beta, cy, incy);
        else
            tinyblas_ctrmv(uplo, trans, diag, n, ca, lda, cx, incx);

        HARVEST(kind == 't' ? cx : cy, rows, kind == 't' ? incx : incy);
        if (cmp_vec("ccomplex", rows, tol * FLT_EPSILON)) return 1;
    }

#undef HARVEST

    return 0;
}

/*
 *  trsv needs no reference: solve, then multiply back and compare to the
 *  original right-hand side. A wrong solve cannot survive its own residual.
 */
static int
check_trsv(enum tinyblas_uplo uplo, enum tinyblas_op trans,
        enum tinyblas_diag diag, int32_t n, int32_t lda, int32_t incx, int cplx)
{
    double tol = 64.0 * (double)n * DBL_EPSILON;

    for (int32_t i = 0; i < MAXA; ++i) ma[i] = NAN + NAN * I;

    /* Diagonally dominant, so the residual measures the algorithm rather than
     * the matrix. A unit diagonal pins the diagonal at 1, so there the
     * off-diagonals have to shrink instead: unit diagonal with O(1)
     * off-diagonals is exponentially ill conditioned and any correct solver
     * would fail this test. */
    {
        double off = (diag == TINYBLAS_UNIT) ? 1.0 / (double)n : 1.0;

        for (int32_t i = 0; i < n; ++i) {
            for (int32_t j = 0; j < n; ++j) {
                double complex v = off * (rnd() + (cplx ? rnd() : 0.0) * I);

                ma[(ptrdiff_t)i * lda + j] = (i == j)
                        ? ((double)n + creal(v)) : v;
            }
        }
    }

    for (int32_t j = 0; j < n; ++j) mx[j] = rnd() + (cplx ? rnd() : 0.0) * I;

    build_tri(n, lda, uplo, trans, diag);

    for (int32_t i = 0; i < MAXA; ++i) { da[i] = creal(ma[i]); za[i] = ma[i]; }
    for (int32_t i = 0; i < MAXV; ++i) { dx[i] = 0.0; zx[i] = 0.0; }

    for (int32_t j = 0; j < n; ++j) {
        dx[vidx(incx, n, j)] = creal(mx[j]);
        zx[vidx(incx, n, j)] = mx[j];
    }

    if (cplx) {
        tinyblas_ztrsv(uplo, trans, diag, n, za, lda, zx, incx);

        for (int32_t j = 0; j < n; ++j) my[j] = zx[vidx(incx, n, j)];
    } else {
        tinyblas_dtrsv(uplo, trans, diag, n, da, lda, dx, incx);

        for (int32_t j = 0; j < n; ++j) my[j] = dx[vidx(incx, n, j)];
    }

    /* residual: mop * solution must reproduce the original right-hand side */
    for (int32_t i = 0; i < n; ++i) {
        double complex s = 0.0;

        for (int32_t j = 0; j < n; ++j) {
            double complex v = mop[(ptrdiff_t)i * n + j];
            double vr = creal(v), vi = cimag(v);
            double yr = creal(my[j]), yi = cimag(my[j]);

            s += (vr * yr - vi * yi) + (vr * yi + vi * yr) * I;
        }

        if (!(cabs(s - mx[i]) <= tol)) {
            printf("FAIL: %s:%d  trsv residual uplo=%d trans=%d diag=%d n=%d "
                   "at %d got=(%g,%g) want=(%g,%g) tol=%g\n",
                   __FILE__, __LINE__, (int)uplo, (int)trans, (int)diag, n, i,
                   creal(s), cimag(s), creal(mx[i]), cimag(mx[i]), tol);

            return 1;
        }
    }

    return 0;
}

/* Compare one typed matrix against the expected dense result left in mop.
 * tri restricts the comparison to the triangle a symmetric or hermitian
 * update is allowed to touch; the other half must be untouched, which is
 * checked separately by seeding it with a value the update would change. */
static int
cmp_mat(const char *name, char type, int32_t m, int32_t n, int32_t lda,
        enum tinyblas_uplo uplo, int tri, double tol)
{
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            ptrdiff_t o = (ptrdiff_t)i * lda + j;
            double complex got, want = mop[(ptrdiff_t)i * n + j];

            if (tri && ((uplo == TINYBLAS_UPPER) ? (j < i) : (j > i))) continue;

            switch (type) {
            case 's': got = sa[o]; break;
            case 'd': got = da[o]; break;
            case 'c': got = ca[o]; break;
            default:  got = za[o]; break;
            }

            if (!(cabs(got - want) <= tol)) {
                printf("FAIL: %s:%d  %s at (%d,%d) got=(%g,%g) want=(%g,%g) "
                       "tol=%g\n", __FILE__, __LINE__, name, i, j,
                       creal(got), cimag(got), creal(want), cimag(want), tol);

                return 1;
            }
        }
    }

    return 0;
}

/* seed every typed copy of A from the master, so all four see the same values */
static void
seed_mats(int32_t rows, int32_t lda)
{
    for (int32_t i = 0; i < rows * lda && i < MAXA; ++i) {
        sa[i] = (float)creal(ma[i]);   da[i] = creal(ma[i]);
        ca[i] = (float complex)ma[i];  za[i] = ma[i];
    }
}

static void
seed_vecs(int32_t lenx, int32_t leny)
{
    for (int32_t i = 0; i < lenx; ++i) {
        sx[i] = (float)creal(mx[i]);   dx[i] = creal(mx[i]);
        cx[i] = (float complex)mx[i];  zx[i] = mx[i];
    }

    for (int32_t j = 0; j < leny; ++j) {
        sy[j] = (float)creal(my[j]);   dy[j] = creal(my[j]);
        cy[j] = (float complex)my[j];  zy[j] = my[j];
    }
}

int main(void) {
    static const enum tinyblas_op tr[3] = {
        TINYBLAS_NONE, TINYBLAS_TRANS, TINYBLAS_CONJ_TRANS
    };
    static const enum tinyblas_uplo ul[2] = { TINYBLAS_UPPER, TINYBLAS_LOWER };
    static const enum tinyblas_diag dg[2] = { TINYBLAS_NON_UNIT, TINYBLAS_UNIT };
    static const int32_t dims[] = {1, 2, 3, 5, 8, 13, 17, 32, 40};
    static const int32_t ndims = (int32_t)(sizeof dims / sizeof dims[0]);

    /* --- hand-computed pins --------------------------------------------- */
    {
        double a[6] = {1, 2, 3,
                       4, 5, 6};             /* 2x3 */
        double x[3] = {1, 1, 1};
        double y[2] = {10, 20};

        /* A*x = (6, 15); 2*A*x + 3*y = (42, 90) */
        tinyblas_dgemv(TINYBLAS_NONE, 2, 3, 2.0, a, 3, x, 1, 3.0, y, 1);
        CHECK(y[0], 42.0, 1e-12);
        CHECK(y[1], 90.0, 1e-12);
    }
    {
        double a[6] = {1, 2, 3, 4, 5, 6};
        double x[2] = {1, 2};
        double y[3] = {0, 0, 0};

        /* A^T*x = (1+8, 2+10, 3+12) = (9, 12, 15) */
        tinyblas_dgemv(TINYBLAS_TRANS, 2, 3, 1.0, a, 3, x, 1, 0.0, y, 1);
        CHECK(y[0], 9.0, 1e-12);
        CHECK(y[1], 12.0, 1e-12);
        CHECK(y[2], 15.0, 1e-12);
    }
    {
        /* upper triangular [[1,2],[.,3]] times (1,1) = (3, 3) */
        double a[4] = {1, 2, 0, 3};
        double x[2] = {1, 1};

        tinyblas_dtrmv(TINYBLAS_UPPER, TINYBLAS_NONE, TINYBLAS_NON_UNIT,
                       2, a, 2, x, 1);
        CHECK(x[0], 3.0, 1e-12);
        CHECK(x[1], 3.0, 1e-12);

        /* and the solve undoes it */
        tinyblas_dtrsv(TINYBLAS_UPPER, TINYBLAS_NONE, TINYBLAS_NON_UNIT,
                       2, a, 2, x, 1);
        CHECK(x[0], 1.0, 1e-12);
        CHECK(x[1], 1.0, 1e-12);
    }
    {
        /* unit diagonal ignores what is stored on it */
        double a[4] = {99, 2, 0, 99};
        double x[2] = {1, 1};

        tinyblas_dtrmv(TINYBLAS_UPPER, TINYBLAS_NONE, TINYBLAS_UNIT,
                       2, a, 2, x, 1);
        CHECK(x[0], 3.0, 1e-12);   /* 1*1 + 2*1 */
        CHECK(x[1], 1.0, 1e-12);
    }
    {
        double x[2] = {1, 2}, y[3] = {1, 10, 100};
        double a[6] = {0, 0, 0, 0, 0, 0};

        /* rank 1: A += 2 * x y^T */
        tinyblas_dger(2, 3, 2.0, x, 1, y, 1, a, 3);
        CHECK(a[0], 2.0, 1e-12);    CHECK(a[1], 20.0, 1e-12);
        CHECK(a[2], 200.0, 1e-12);  CHECK(a[3], 4.0, 1e-12);
        CHECK(a[5], 400.0, 1e-12);
    }
    {
        double x[2] = {1, 2};
        double a[4] = {0, 0, 0, 0};

        /* upper syr: only the upper triangle is written */
        tinyblas_dsyr(TINYBLAS_UPPER, 2, 1.0, x, 1, a, 2);
        CHECK(a[0], 1.0, 1e-12);   CHECK(a[1], 2.0, 1e-12);
        CHECK(a[2], 0.0, 0.0);     CHECK(a[3], 4.0, 1e-12);
    }
    {
        double complex x[2] = {1.0 + 1.0 * I, 2.0};
        double complex a[4] = {0.0, 0.0, 0.0, 0.0};

        /* her keeps the diagonal real: |1+i|^2 = 2 */
        tinyblas_zher(TINYBLAS_UPPER, 2, 1.0, x, 1, a, 2);
        CHECK_C(a[0], 2.0, 0.0, 1e-12);
        CHECK_C(a[1], 2.0, 2.0, 1e-12);   /* (1+i)*conj(2) */
        CHECK_C(a[3], 4.0, 0.0, 1e-12);
    }

    /* --- gemv across shapes, trans, increments and both data kinds ------ */
    for (int32_t d = 0; d < ndims; ++d) {
        int32_t m = dims[d];
        int32_t n = dims[(d + 1) % ndims];

        for (int32_t t = 0; t < 3; ++t) {
            if (check_matvec('g', tr[t], TINYBLAS_UPPER, TINYBLAS_NON_UNIT,
                             m, n, n, 0.7 + 0.3 * I, -0.3 + 0.5 * I, 1, 1, 1))
                return 1;

            if (check_matvec('g', tr[t], TINYBLAS_UPPER, TINYBLAS_NON_UNIT,
                             m, n, n + 3, 0.7, -0.3, 1, 1, 0))
                return 1;

            /* strided and reversed vectors */
            if (check_matvec('g', tr[t], TINYBLAS_UPPER, TINYBLAS_NON_UNIT,
                             m, n, n, 1.0, 0.0, 2, 3, 1))
                return 1;

            if (check_matvec('g', tr[t], TINYBLAS_UPPER, TINYBLAS_NON_UNIT,
                             m, n, n, 1.0, 1.0, -1, -2, 1))
                return 1;
        }
    }

    /* --- symv, hemv ----------------------------------------------------- */
    for (int32_t d = 0; d < ndims; ++d) {
        int32_t n = dims[d];

        for (int32_t u = 0; u < 2; ++u) {
            if (check_matvec('s', TINYBLAS_NONE, ul[u], TINYBLAS_NON_UNIT,
                             n, n, n, 0.7, -0.3, 1, 1, 0))
                return 1;

            if (check_matvec('s', TINYBLAS_NONE, ul[u], TINYBLAS_NON_UNIT,
                             n, n, n + 2, 0.7 + 0.3 * I, -0.3 + 0.5 * I, 1, 1, 1))
                return 1;

            if (check_matvec('s', TINYBLAS_NONE, ul[u], TINYBLAS_NON_UNIT,
                             n, n, n, 1.0, 0.0, -2, 3, 1))
                return 1;
        }
    }

    /* --- trmv over every uplo, trans and diag --------------------------- */
    for (int32_t d = 0; d < ndims; ++d) {
        int32_t n = dims[d];

        for (int32_t u = 0; u < 2; ++u) {
            for (int32_t t = 0; t < 3; ++t) {
                for (int32_t g = 0; g < 2; ++g) {
                    if (check_matvec('t', tr[t], ul[u], dg[g],
                                     n, n, n, 1.0, 0.0, 1, 1, 1))
                        return 1;

                    if (check_matvec('t', tr[t], ul[u], dg[g],
                                     n, n, n + 1, 1.0, 0.0, 1, 1, 0))
                        return 1;

                    if (check_matvec('t', tr[t], ul[u], dg[g],
                                     n, n, n, 1.0, 0.0, -2, 1, 1))
                        return 1;
                }
            }
        }
    }

    /* --- trsv, checked by its own residual ------------------------------ */
    for (int32_t d = 0; d < ndims; ++d) {
        int32_t n = dims[d];

        for (int32_t u = 0; u < 2; ++u) {
            for (int32_t t = 0; t < 3; ++t) {
                for (int32_t g = 0; g < 2; ++g) {
                    if (check_trsv(ul[u], tr[t], dg[g], n, n, 1, 0)) return 1;
                    if (check_trsv(ul[u], tr[t], dg[g], n, n, 1, 1)) return 1;
                    if (check_trsv(ul[u], tr[t], dg[g], n, n + 2, -2, 1))
                        return 1;
                }
            }
        }
    }

    /* --- rank updates, every routine against an explicit outer product -- */
    for (int32_t d = 0; d < ndims; ++d) {
        int32_t n = dims[d];
        int32_t m = dims[(d + 1) % ndims];
        double complex al = 0.6 + 0.4 * I;
        double tolf = 1e-4, told = 1e-12;

        /* --- ger, geru, gerc --- */
        for (int32_t i = 0; i < MAXA; ++i) ma[i] = 0.0;

        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                ma[(ptrdiff_t)i * n + j] = rnd() + rnd() * I;

        for (int32_t i = 0; i < m; ++i) mx[i] = rnd() + rnd() * I;
        for (int32_t j = 0; j < n; ++j) my[j] = rnd() + rnd() * I;

        /* unconjugated */
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                mop[(ptrdiff_t)i * n + j] = ma[(ptrdiff_t)i * n + j]
                                          + al * mx[i] * my[j];

        seed_mats(m, n); seed_vecs(m, n);
        tinyblas_zgeru(m, n, al, zx, 1, zy, 1, za, n);
        if (cmp_mat("zgeru", 'z', m, n, n, ul[0], 0, told)) return 1;

        tinyblas_cgeru(m, n, (float complex)al, cx, 1, cy, 1, ca, n);
        if (cmp_mat("cgeru", 'c', m, n, n, ul[0], 0, tolf)) return 1;

        /* the real routines see only the real parts of everything */
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                mop[(ptrdiff_t)i * n + j] = creal(ma[(ptrdiff_t)i * n + j])
                                          + creal(al) * creal(mx[i])
                                            * creal(my[j]);

        tinyblas_dger(m, n, creal(al), dx, 1, dy, 1, da, n);
        if (cmp_mat("dger", 'd', m, n, n, ul[0], 0, told)) return 1;

        tinyblas_sger(m, n, (float)creal(al), sx, 1, sy, 1, sa, n);
        if (cmp_mat("sger", 's', m, n, n, ul[0], 0, tolf)) return 1;

        /* conjugated */
        for (int32_t i = 0; i < m; ++i)
            for (int32_t j = 0; j < n; ++j)
                mop[(ptrdiff_t)i * n + j] = ma[(ptrdiff_t)i * n + j]
                                          + al * mx[i] * conj(my[j]);

        seed_mats(m, n); seed_vecs(m, n);
        tinyblas_zgerc(m, n, al, zx, 1, zy, 1, za, n);
        if (cmp_mat("zgerc", 'z', m, n, n, ul[0], 0, told)) return 1;

        tinyblas_cgerc(m, n, (float complex)al, cx, 1, cy, 1, ca, n);
        if (cmp_mat("cgerc", 'c', m, n, n, ul[0], 0, tolf)) return 1;

        /* --- syr, her, syr2, her2 over both triangles --- */
        for (int32_t u = 0; u < 2; ++u) {
            for (int32_t i = 0; i < MAXA; ++i) ma[i] = 0.0;

            /* a hermitian operand needs a real diagonal to start with */
            for (int32_t i = 0; i < n; ++i)
                for (int32_t j = 0; j < n; ++j)
                    ma[(ptrdiff_t)i * n + j] = (i == j) ? rnd()
                                                        : (rnd() + rnd() * I);

            for (int32_t i = 0; i < n; ++i) {
                mx[i] = rnd() + rnd() * I;
                my[i] = rnd() + rnd() * I;
            }

            /* syr: A += alpha * x x^T, real */
            for (int32_t i = 0; i < n; ++i)
                for (int32_t j = 0; j < n; ++j)
                    mop[(ptrdiff_t)i * n + j] = creal(ma[(ptrdiff_t)i * n + j])
                            + creal(al) * creal(mx[i]) * creal(mx[j]);

            seed_mats(n, n); seed_vecs(n, n);
            tinyblas_dsyr(ul[u], n, creal(al), dx, 1, da, n);
            if (cmp_mat("dsyr", 'd', n, n, n, ul[u], 1, told)) return 1;

            tinyblas_ssyr(ul[u], n, (float)creal(al), sx, 1, sa, n);
            if (cmp_mat("ssyr", 's', n, n, n, ul[u], 1, tolf)) return 1;

            /* her: A += alpha * x conj(x)^T with alpha real, diagonal real */
            for (int32_t i = 0; i < n; ++i) {
                for (int32_t j = 0; j < n; ++j) {
                    double complex w = ma[(ptrdiff_t)i * n + j]
                                     + creal(al) * mx[i] * conj(mx[j]);

                    mop[(ptrdiff_t)i * n + j] = (i == j) ? creal(w) : w;
                }
            }

            seed_mats(n, n); seed_vecs(n, n);
            tinyblas_zher(ul[u], n, creal(al), zx, 1, za, n);
            if (cmp_mat("zher", 'z', n, n, n, ul[u], 1, told)) return 1;

            tinyblas_cher(ul[u], n, (float)creal(al), cx, 1, ca, n);
            if (cmp_mat("cher", 'c', n, n, n, ul[u], 1, tolf)) return 1;

            /* syr2: A += alpha*(x y^T + y x^T), real */
            for (int32_t i = 0; i < n; ++i)
                for (int32_t j = 0; j < n; ++j)
                    mop[(ptrdiff_t)i * n + j] = creal(ma[(ptrdiff_t)i * n + j])
                            + creal(al) * (creal(mx[i]) * creal(my[j])
                                         + creal(my[i]) * creal(mx[j]));

            seed_mats(n, n); seed_vecs(n, n);
            tinyblas_dsyr2(ul[u], n, creal(al), dx, 1, dy, 1, da, n);
            if (cmp_mat("dsyr2", 'd', n, n, n, ul[u], 1, told)) return 1;

            tinyblas_ssyr2(ul[u], n, (float)creal(al), sx, 1, sy, 1, sa, n);
            if (cmp_mat("ssyr2", 's', n, n, n, ul[u], 1, tolf)) return 1;

            /* her2: A += alpha*x conj(y)^T + conj(alpha)*y conj(x)^T */
            for (int32_t i = 0; i < n; ++i) {
                for (int32_t j = 0; j < n; ++j) {
                    double complex w = ma[(ptrdiff_t)i * n + j]
                                     + al * mx[i] * conj(my[j])
                                     + conj(al) * my[i] * conj(mx[j]);

                    mop[(ptrdiff_t)i * n + j] = (i == j) ? creal(w) : w;
                }
            }

            seed_mats(n, n); seed_vecs(n, n);
            tinyblas_zher2(ul[u], n, al, zx, 1, zy, 1, za, n);
            if (cmp_mat("zher2", 'z', n, n, n, ul[u], 1, told)) return 1;

            tinyblas_cher2(ul[u], n, (float complex)al, cx, 1, cy, 1, ca, n);
            if (cmp_mat("cher2", 'c', n, n, n, ul[u], 1, tolf)) return 1;

            /* the untouched triangle must still hold what it was seeded with */
            for (int32_t i = 0; i < n; ++i) {
                for (int32_t j = 0; j < n; ++j) {
                    if ((ul[u] == TINYBLAS_UPPER) ? (j >= i) : (j <= i))
                        continue;

                    if (!(cabs(za[(ptrdiff_t)i * n + j]
                               - ma[(ptrdiff_t)i * n + j]) == 0.0)) {
                        printf("FAIL: %s:%d  zher2 wrote outside uplo=%d "
                               "at (%d,%d)\n",
                               __FILE__, __LINE__, (int)ul[u], i, j);

                        return 1;
                    }
                }
            }
        }
    }

    /* --- semantics ------------------------------------------------------ */
    {
        double a[4] = {1, 2, 3, 4};
        double x[2] = {1, 1};
        double y[2];

        /* beta == 0 must overwrite y, never read it */
        y[0] = y[1] = NAN;
        tinyblas_dgemv(TINYBLAS_NONE, 2, 2, 1.0, a, 2, x, 1, 0.0, y, 1);
        CHECK(isfinite(y[0]) ? 0.0 : 1.0, 0.0, 0.0);
        CHECK(y[0], 3.0, 1e-12);

        /* alpha == 0 leaves exactly beta*y and must not read A or x */
        y[0] = y[1] = 2.0;
        tinyblas_dgemv(TINYBLAS_NONE, 2, 2, 0.0, NULL, 2, NULL, 1, 3.0, y, 1);
        CHECK(y[0], 6.0, 0.0);
        CHECK(y[1], 6.0, 0.0);

        /* an empty problem touches nothing */
        y[0] = y[1] = 7.0;
        tinyblas_dgemv(TINYBLAS_NONE, 0, 2, 1.0, a, 2, x, 1, 0.0, y, 1);
        CHECK(y[0], 7.0, 0.0);

        /* alpha == 0 makes the rank updates no-ops */
        {
            double aa[4] = {1, 2, 3, 4};

            tinyblas_dger(2, 2, 0.0, NULL, 1, NULL, 1, aa, 2);
            CHECK(aa[0], 1.0, 0.0);
            CHECK(aa[3], 4.0, 0.0);
        }
    }

    printf("level2: all tests passed\n");

    return 0;
}
