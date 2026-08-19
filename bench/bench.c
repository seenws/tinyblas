/* clock_gettime is POSIX, and -std=iso9899:1999 hides it. */
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdint.h>
#include <complex.h>

#include "tinyblas.h"

#ifdef TINYBLAS_HAVE_OPENBLAS
#include <cblas.h>
/* Not declared in cblas.h, and setenv() would be too late: OpenBLAS reads the
 * environment from a library constructor, before main runs. */
extern void openblas_set_num_threads(int);
#endif

/* A single rep must not exceed this, or the size is skipped. The naive path is
 * an order of magnitude off the packed one, so without a budget the first
 * bench run of a new kernel takes all afternoon. */
#define BUDGET_SECONDS 4.0

/* Above this the result is wrong, not merely rounded differently. It is a
 * multiple of the naive forward-error bound, so one threshold fits all sizes. */
#define ERR_LIMIT 32.0

struct prob {
    int32_t m, n, k;
    const void *a, *b;
    void *c;
    int32_t lda, ldb, ldc;
};

static uint32_t rng = 12345u;

static double
rnd(void)
{
    rng = rng * 1103515245u + 12345u;

    return (double)(rng >> 8) / (double)(1u << 24) * 2.0 - 1.0;
}

static double
now(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static size_t
esize(char t)
{
    switch (t) {
    case 's': return sizeof(float);
    case 'd': return sizeof(double);
    case 'c': return sizeof(float complex);
    default:  return sizeof(double complex);
    }
}

#ifdef TINYBLAS_HAVE_OPENBLAS
static double
eps_of(char t)
{
    return (t == 's' || t == 'c') ? (double)FLT_EPSILON : DBL_EPSILON;
}
#endif

/* Real gemm is 2 flops per multiply-add. A complex multiply-add is 4 real
 * multiplies and 4 real adds, which the literature counts as 8. */
static double
flop_mul(char t)
{
    return (t == 's' || t == 'd') ? 2.0 : 8.0;
}

/* 2 FMA ports, 256 bits wide, 2 flops per lane per FMA. A complex kernel
 * reaches the same ceiling as its component type: 4 FMAs produce the 8 flops.
 * The clock is the calibration knob; this box boosts to 4.0 GHz on AVX2. */
static double
peak_gflops(char t)
{
    const char *e   = getenv("TINYBLAS_PEAK_GHZ");
    double      ghz = e ? atof(e) : 4.0;
    double      lanes = (t == 's' || t == 'c') ? 8.0 : 4.0;

    return ghz * 2.0 * lanes * 2.0;
}

static void
fill_rand(char t, void *p, size_t nelem)
{
    size_t i;

    switch (t) {
    case 's': {
        float *q = p;
        for (i = 0; i < nelem; ++i) q[i] = (float)rnd();
        break;
    }
    case 'd': {
        double *q = p;
        for (i = 0; i < nelem; ++i) q[i] = rnd();
        break;
    }
    case 'c': {
        float complex *q = p;
        for (i = 0; i < nelem; ++i) q[i] = (float)rnd() + (float)rnd() * I;
        break;
    }
    default: {
        double complex *q = p;
        for (i = 0; i < nelem; ++i) q[i] = rnd() + rnd() * I;
        break;
    }
    }
}

static void
call_tb(char t, const struct prob *p, double complex al, double complex be)
{
    switch (t) {
    case 's':
        tinyblas_sgemm(TINYBLAS_NO_TRANS, TINYBLAS_NO_TRANS, p->m, p->n, p->k,
                       (float)creal(al), p->a, p->lda, p->b, p->ldb,
                       (float)creal(be), p->c, p->ldc);
        break;
    case 'd':
        tinyblas_dgemm(TINYBLAS_NO_TRANS, TINYBLAS_NO_TRANS, p->m, p->n, p->k,
                       creal(al), p->a, p->lda, p->b, p->ldb,
                       creal(be), p->c, p->ldc);
        break;
    case 'c':
        tinyblas_cgemm(TINYBLAS_NO_TRANS, TINYBLAS_NO_TRANS, p->m, p->n, p->k,
                       (float complex)al, p->a, p->lda, p->b, p->ldb,
                       (float complex)be, p->c, p->ldc);
        break;
    default:
        tinyblas_zgemm(TINYBLAS_NO_TRANS, TINYBLAS_NO_TRANS, p->m, p->n, p->k,
                       al, p->a, p->lda, p->b, p->ldb,
                       be, p->c, p->ldc);
        break;
    }
}

#ifdef TINYBLAS_HAVE_OPENBLAS
static void
call_ob(char t, const struct prob *p, double complex al, double complex be)
{
    int m = (int)p->m, n = (int)p->n, k = (int)p->k;
    int lda = (int)p->lda, ldb = (int)p->ldb, ldc = (int)p->ldc;

    switch (t) {
    case 's':
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                    (float)creal(al), p->a, lda, p->b, ldb,
                    (float)creal(be), p->c, ldc);
        break;
    case 'd':
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                    creal(al), p->a, lda, p->b, ldb,
                    creal(be), p->c, ldc);
        break;
    case 'c': {
        float complex a1 = (float complex)al, b1 = (float complex)be;
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                    &a1, p->a, lda, p->b, ldb, &b1, p->c, ldc);
        break;
    }
    default: {
        double complex a1 = al, b1 = be;
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                    &a1, p->a, lda, p->b, ldb, &b1, p->c, ldc);
        break;
    }
    }
}
#endif

#ifdef TINYBLAS_HAVE_OPENBLAS
static double
maxdiff(char t, const void *x, const void *y, int32_t m, int32_t n, int32_t ld)
{
    double worst = 0.0;
    int32_t i, j;

    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            size_t o = (size_t)i * (size_t)ld + (size_t)j;
            double d;

            switch (t) {
            case 's': d = fabs((double)((const float *)x)[o]
                             - (double)((const float *)y)[o]); break;
            case 'd': d = fabs(((const double *)x)[o]
                             - ((const double *)y)[o]); break;
            case 'c': d = cabs((double complex)((const float complex *)x)[o]
                             - (double complex)((const float complex *)y)[o]); break;
            default:  d = cabs(((const double complex *)x)[o]
                             - ((const double complex *)y)[o]); break;
            }

            if (d > worst) worst = d;
        }
    }

    return worst;
}
#endif

/* Minimum of repeated runs. For a deterministic kernel the minimum is the
 * cleanest estimate of what the machine can do; the mean just measures how
 * busy the rest of the box was. */
static double
best_time(char t, const struct prob *p, int use_ob)
{
    double best = 1e30, total = 0.0;
    int reps = 0;

    (void)use_ob;

    /* untimed warm-up, which also spins the core up to its boost clock */
#ifdef TINYBLAS_HAVE_OPENBLAS
    if (use_ob) call_ob(t, p, 1.0, 0.0); else
#endif
    call_tb(t, p, 1.0, 0.0);

    /* At least five samples even when one rep already exceeds the time target,
     * or the big sizes report a single unrepeated measurement. This box is a
     * 4-core laptop-class part under WSL2, and anything less than five is
     * visibly at the mercy of whatever else the scheduler is doing. */
    while ((reps < 5 || total < 0.40) && reps < 200) {
        double t0 = now(), dt;

#ifdef TINYBLAS_HAVE_OPENBLAS
        if (use_ob) call_ob(t, p, 1.0, 0.0); else
#endif
        call_tb(t, p, 1.0, 0.0);

        dt = now() - t0;

        if (dt < best) best = dt;
        total += dt;
        ++reps;
    }

    return best;
}

static void
call_tb_gemv(char t, const struct prob *p, double complex al, double complex be)
{
    switch (t) {
    case 's':
        tinyblas_sgemv(TINYBLAS_NO_TRANS, p->m, p->n, (float)creal(al),
                       p->a, p->lda, p->b, 1, (float)creal(be), p->c, 1);
        break;
    case 'd':
        tinyblas_dgemv(TINYBLAS_NO_TRANS, p->m, p->n, creal(al),
                       p->a, p->lda, p->b, 1, creal(be), p->c, 1);
        break;
    case 'c':
        tinyblas_cgemv(TINYBLAS_NO_TRANS, p->m, p->n, (float complex)al,
                       p->a, p->lda, p->b, 1, (float complex)be, p->c, 1);
        break;
    default:
        tinyblas_zgemv(TINYBLAS_NO_TRANS, p->m, p->n, al,
                       p->a, p->lda, p->b, 1, be, p->c, 1);
        break;
    }
}

#ifdef TINYBLAS_HAVE_OPENBLAS
static void
call_ob_gemv(char t, const struct prob *p, double complex al, double complex be)
{
    int m = (int)p->m, n = (int)p->n, lda = (int)p->lda;

    switch (t) {
    case 's':
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, (float)creal(al),
                    p->a, lda, p->b, 1, (float)creal(be), p->c, 1);
        break;
    case 'd':
        cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, creal(al),
                    p->a, lda, p->b, 1, creal(be), p->c, 1);
        break;
    case 'c': {
        float complex a1 = (float complex)al, b1 = (float complex)be;
        cblas_cgemv(CblasRowMajor, CblasNoTrans, m, n, &a1,
                    p->a, lda, p->b, 1, &b1, p->c, 1);
        break;
    }
    default: {
        double complex a1 = al, b1 = be;
        cblas_zgemv(CblasRowMajor, CblasNoTrans, m, n, &a1,
                    p->a, lda, p->b, 1, &b1, p->c, 1);
        break;
    }
    }
}
#endif

static double
best_time_gemv(char t, const struct prob *p, int use_ob)
{
    double best = 1e30, total = 0.0;
    int reps = 0;

    (void)use_ob;

#ifdef TINYBLAS_HAVE_OPENBLAS
    if (use_ob) call_ob_gemv(t, p, 1.0, 0.0); else
#endif
    call_tb_gemv(t, p, 1.0, 0.0);

    while ((reps < 5 || total < 0.40) && reps < 500) {
        double t0 = now(), dt;

#ifdef TINYBLAS_HAVE_OPENBLAS
        if (use_ob) call_ob_gemv(t, p, 1.0, 0.0); else
#endif
        call_tb_gemv(t, p, 1.0, 0.0);

        dt = now() - t0;

        if (dt < best) best = dt;
        total += dt;
        ++reps;
    }

    return best;
}

/* Streaming bandwidth for context. gemv reads the matrix once and does two
 * flops per element, so it cannot beat this number however good the kernel is. */
static double
memcpy_gbs(size_t bytes)
{
    char *a = malloc(bytes), *b = malloc(bytes);
    double best = 1e30;
    int r;

    if (!a || !b) { free(a); free(b); return 0.0; }

    memset(a, 1, bytes);
    memcpy(b, a, bytes);

    for (r = 0; r < 5; ++r) {
        double t0 = now(), dt;

        memcpy(b, a, bytes);
        dt = now() - t0;

        if (dt < best) best = dt;
    }

    free(a);
    free(b);

    /* a copy touches the source and the destination */
    return 2.0 * (double)bytes / best / 1e9;
}

/*
 *  gemv is a bandwidth benchmark wearing a BLAS costume
 *
 *  It moves m*n elements to do 2*m*n flops, so GFLOP/s alone is misleading:
 *  the GB/s column next to it is the one that says whether there is any room
 *  left, and the memcpy line is the ceiling.
 */
static int
bench_gemv(char t, const int32_t *sizes, int nsizes)
{
    size_t es = esize(t);
    double last_gflops = 0.0;
    int i;

    printf("%cgemv   single thread, %g*m*n flops, bandwidth bound\n",
           t, flop_mul(t));
    printf("        memcpy reference: %.1f GB/s\n\n",
           memcpy_gbs(64u * 1024u * 1024u));
    printf("%6s %5s %11s %11s %8s %9s\n",
           "m", "n", "tinyblas", "openblas", "ratio", "GB/s");

    for (i = 0; i < nsizes; ++i) {
        int32_t s = sizes[i];
        double flops = flop_mul(t) * (double)s * (double)s;
        double bytes = (double)s * (double)s * (double)es;
        struct prob p;
        void *a, *x, *y;
        double tb_s, tb_g, ob_g = 0.0;

        if (last_gflops > 0.0 && flops / (last_gflops * 1e9) > BUDGET_SECONDS)
            continue;

        a = malloc((size_t)s * (size_t)s * es);
        x = malloc((size_t)s * es);
        y = malloc((size_t)s * es);

        if (!a || !x || !y) { free(a); free(x); free(y); continue; }

        fill_rand(t, a, (size_t)s * (size_t)s);
        fill_rand(t, x, (size_t)s);
        fill_rand(t, y, (size_t)s);

        p.m = p.n = s;
        p.k = s;
        p.lda = p.ldb = p.ldc = s;
        p.a = a; p.b = x; p.c = y;

        tb_s = best_time_gemv(t, &p, 0);
        tb_g = flops / tb_s / 1e9;
        last_gflops = tb_g;

#ifdef TINYBLAS_HAVE_OPENBLAS
        ob_g = flops / best_time_gemv(t, &p, 1) / 1e9;
#endif

        printf("%6d %5d %11.2f", s, s, tb_g);

        if (ob_g > 0.0) printf(" %11.2f %7.1f%%", ob_g, 100.0 * tb_g / ob_g);
        else            printf(" %11s %8s", "-", "-");

        printf(" %8.1f\n", bytes / tb_s / 1e9);
        fflush(stdout);

        free(a); free(x); free(y);
    }

    printf("\n");

    return 0;
}

static int
bench_gemm(char t, const int32_t *sizes, int nsizes)
{
    size_t es = esize(t);
    double peak = peak_gflops(t);
    double last_gflops = 0.0;
    int i, rc = 0;

    printf("%cgemm   single thread, %g*m*n*k flops, peak %.2f GFLOP/s\n\n",
           t, flop_mul(t), peak);
    printf("%6s %5s %5s %11s %11s %8s %7s %7s\n",
           "m", "n", "k", "tinyblas", "openblas", "ratio", "%peak", "err");

    for (i = 0; i < nsizes; ++i) {
        int32_t s = sizes[i];
        double flops = flop_mul(t) * (double)s * (double)s * (double)s;
        struct prob p;
        void *a, *b, *c, *r, *c0;
        double tb_s, tb_g, ob_g = 0.0, err = -1.0;

        if (last_gflops > 0.0 && flops / (last_gflops * 1e9) > BUDGET_SECONDS) {
            printf("%6d %5d %5d   (skipped: over the %.0fs budget at the "
                   "measured rate)\n", s, s, s, BUDGET_SECONDS);
            continue;
        }

        a  = malloc((size_t)s * (size_t)s * es);
        b  = malloc((size_t)s * (size_t)s * es);
        c  = malloc((size_t)s * (size_t)s * es);
        r  = malloc((size_t)s * (size_t)s * es);
        c0 = malloc((size_t)s * (size_t)s * es);

        if (!a || !b || !c || !r || !c0) {
            printf("%6d %5d %5d   (skipped: out of memory)\n", s, s, s);
            free(a); free(b); free(c); free(r); free(c0);
            continue;
        }

        fill_rand(t, a, (size_t)s * (size_t)s);
        fill_rand(t, b, (size_t)s * (size_t)s);
        fill_rand(t, c0, (size_t)s * (size_t)s);

        p.m = p.n = p.k = s;
        p.lda = p.ldb = p.ldc = s;
        p.a = a; p.b = b; p.c = c;

        /* Correctness before speed, with both scaling paths live, so a fast
         * but wrong kernel cannot post a number. */
#ifdef TINYBLAS_HAVE_OPENBLAS
        {
            double bound = (double)s * eps_of(t);

            memcpy(c, c0, (size_t)s * (size_t)s * es);
            call_tb(t, &p, 0.7 + 0.3 * I, -0.3 + 0.5 * I);

            memcpy(r, c0, (size_t)s * (size_t)s * es);
            p.c = r;
            call_ob(t, &p, 0.7 + 0.3 * I, -0.3 + 0.5 * I);
            p.c = c;

            err = maxdiff(t, c, r, s, s, s) / bound;
        }
#endif

        tb_s = best_time(t, &p, 0);
        tb_g = flops / tb_s / 1e9;
        last_gflops = tb_g;

#ifdef TINYBLAS_HAVE_OPENBLAS
        p.c = r;
        ob_g = flops / best_time(t, &p, 1) / 1e9;
        p.c = c;
#endif

        printf("%6d %5d %5d %11.2f", s, s, s, tb_g);

        if (ob_g > 0.0)
            printf(" %11.2f %7.1f%%", ob_g, 100.0 * tb_g / ob_g);
        else
            printf(" %11s %8s", "-", "-");

        printf(" %6.1f%%", 100.0 * tb_g / peak);

        if (err >= 0.0) printf(" %7.1f", err); else printf(" %7s", "-");

        printf("\n");
        fflush(stdout);

        if (err > ERR_LIMIT) {
            printf("\nFAIL: %cgemm at %d is %.1fx the naive error bound, "
                   "which is wrong rather than merely rounded\n", t, s, err);
            rc = 1;
        }

        free(a); free(b); free(c); free(r); free(c0);

        if (rc) return rc;
    }

    printf("\n");

    return 0;
}

/*
 *  The derived level 3 routines
 *
 *  Every one of them is the gemm core wearing a different hat, so the number
 *  that matters is the fraction of gemm speed the wrapper keeps. One real and
 *  one complex type says that: s is d's code with a narrower element, and c
 *  is z's.
 *
 *  Fixed selectors throughout: side LEFT, uplo UPPER, no transpose, non-unit
 *  diagonal. The shape sweep in make test owns the other combinations.
 */
struct l3case {
    char t;
    const char *name;
    int inplace;            /* trmm and trsm overwrite B */
};

static const struct l3case l3list[] = {
    {'d', "symm",  0}, {'d', "syrk",  0}, {'d', "syr2k", 0},
    {'d', "trmm",  1}, {'d', "trsm",  1},
    {'z', "symm",  0}, {'z', "hemm",  0}, {'z', "syrk",  0}, {'z', "herk",  0},
    {'z', "syr2k", 0}, {'z', "her2k", 0}, {'z', "trmm",  1}, {'z', "trsm",  1}
};

/* symm, hemm, syr2k and her2k are full products; syrk, herk, trmm and trsm
 * touch one triangle and so do half the work of a gemm of the same shape. */
static double
flop_l3(char t, const char *name, int32_t s)
{
    double base = (t == 's' || t == 'd') ? 1.0 : 4.0;
    double mul  = (!strcmp(name, "symm")  || !strcmp(name, "hemm") ||
                   !strcmp(name, "syr2k") || !strcmp(name, "her2k")) ? 2.0 : 1.0;

    return base * mul * (double)s * (double)s * (double)s;
}

/* trsm on a random triangle at n = 1024 is numerically hopeless, and the
 * denormals that fall out of it measure the FPU's slow path rather than the
 * kernel. A diagonal of n keeps the solve well conditioned.
 *
 * Only trsm gets this. The err column divides by a bound that assumes the
 * operands are in [-1,1), so inflating a diagonal anywhere else reports a
 * correct routine as broken -- which is exactly what it did the first time. */
static void
make_dominant(char t, void *a, int32_t s)
{
    int32_t i;

    for (i = 0; i < s; ++i) {
        size_t o = (size_t)i * (size_t)s + (size_t)i;

        switch (t) {
        case 's': ((float *)a)[o]          = (float)s;  break;
        case 'd': ((double *)a)[o]         = (double)s; break;
        case 'c': ((float complex *)a)[o]  = (float)s;  break;
        default:  ((double complex *)a)[o] = (double)s; break;
        }
    }
}

static void
call_l3_tb(char t, const char *name, int32_t s,
        const void *a, void *b, void *c, double al, double be)
{
    if (t == 'd') {
        if      (!strcmp(name, "symm"))
            tinyblas_dsymm(TINYBLAS_LEFT, TINYBLAS_UPPER, s, s, al,
                           a, s, b, s, be, c, s);
        else if (!strcmp(name, "syrk"))
            tinyblas_dsyrk(TINYBLAS_UPPER, TINYBLAS_NO_TRANS, s, s, al,
                           a, s, be, c, s);
        else if (!strcmp(name, "syr2k"))
            tinyblas_dsyr2k(TINYBLAS_UPPER, TINYBLAS_NO_TRANS, s, s, al,
                            a, s, b, s, be, c, s);
        else if (!strcmp(name, "trmm"))
            tinyblas_dtrmm(TINYBLAS_LEFT, TINYBLAS_UPPER, TINYBLAS_NO_TRANS,
                           TINYBLAS_NON_UNIT, s, s, al, a, s, b, s);
        else
            tinyblas_dtrsm(TINYBLAS_LEFT, TINYBLAS_UPPER, TINYBLAS_NO_TRANS,
                           TINYBLAS_NON_UNIT, s, s, al, a, s, b, s);

        return;
    }

    if      (!strcmp(name, "symm"))
        tinyblas_zsymm(TINYBLAS_LEFT, TINYBLAS_UPPER, s, s, al,
                       a, s, b, s, be, c, s);
    else if (!strcmp(name, "hemm"))
        tinyblas_zhemm(TINYBLAS_LEFT, TINYBLAS_UPPER, s, s, al,
                       a, s, b, s, be, c, s);
    else if (!strcmp(name, "syrk"))
        tinyblas_zsyrk(TINYBLAS_UPPER, TINYBLAS_NO_TRANS, s, s, al,
                       a, s, be, c, s);
    else if (!strcmp(name, "herk"))
        tinyblas_zherk(TINYBLAS_UPPER, TINYBLAS_NO_TRANS, s, s, al,
                       a, s, be, c, s);
    else if (!strcmp(name, "syr2k"))
        tinyblas_zsyr2k(TINYBLAS_UPPER, TINYBLAS_NO_TRANS, s, s, al,
                        a, s, b, s, be, c, s);
    else if (!strcmp(name, "her2k"))
        tinyblas_zher2k(TINYBLAS_UPPER, TINYBLAS_NO_TRANS, s, s, al,
                        a, s, b, s, be, c, s);
    else if (!strcmp(name, "trmm"))
        tinyblas_ztrmm(TINYBLAS_LEFT, TINYBLAS_UPPER, TINYBLAS_NO_TRANS,
                       TINYBLAS_NON_UNIT, s, s, al, a, s, b, s);
    else
        tinyblas_ztrsm(TINYBLAS_LEFT, TINYBLAS_UPPER, TINYBLAS_NO_TRANS,
                       TINYBLAS_NON_UNIT, s, s, al, a, s, b, s);
}

#ifdef TINYBLAS_HAVE_OPENBLAS
static void
call_l3_ob(char t, const char *name, int32_t s,
        const void *a, void *b, void *c, double al, double be)
{
    int n = (int)s;

    if (t == 'd') {
        if      (!strcmp(name, "symm"))
            cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, n, n, al,
                        a, n, b, n, be, c, n);
        else if (!strcmp(name, "syrk"))
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, n, al,
                        a, n, be, c, n);
        else if (!strcmp(name, "syr2k"))
            cblas_dsyr2k(CblasRowMajor, CblasUpper, CblasNoTrans, n, n, al,
                         a, n, b, n, be, c, n);
        else if (!strcmp(name, "trmm"))
            cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                        CblasNonUnit, n, n, al, a, n, b, n);
        else
            cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                        CblasNonUnit, n, n, al, a, n, b, n);

        return;
    }

    {
        double complex az = al, bz = be;

        if      (!strcmp(name, "symm"))
            cblas_zsymm(CblasRowMajor, CblasLeft, CblasUpper, n, n, &az,
                        a, n, b, n, &bz, c, n);
        else if (!strcmp(name, "hemm"))
            cblas_zhemm(CblasRowMajor, CblasLeft, CblasUpper, n, n, &az,
                        a, n, b, n, &bz, c, n);
        else if (!strcmp(name, "syrk"))
            cblas_zsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, n, &az,
                        a, n, &bz, c, n);
        else if (!strcmp(name, "herk"))
            cblas_zherk(CblasRowMajor, CblasUpper, CblasNoTrans, n, n, al,
                        a, n, be, c, n);
        else if (!strcmp(name, "syr2k"))
            cblas_zsyr2k(CblasRowMajor, CblasUpper, CblasNoTrans, n, n, &az,
                         a, n, b, n, &bz, c, n);
        else if (!strcmp(name, "her2k"))
            cblas_zher2k(CblasRowMajor, CblasUpper, CblasNoTrans, n, n, &az,
                         a, n, b, n, be, c, n);
        else if (!strcmp(name, "trmm"))
            cblas_ztrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                        CblasNonUnit, n, n, &az, a, n, b, n);
        else
            cblas_ztrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                        CblasNonUnit, n, n, &az, a, n, b, n);
    }
}
#endif

/* As best_time, but B is restored from a pristine copy between reps for the
 * routines that overwrite it. The restore is outside the timed region, so a
 * trsm cannot quietly measure its own output decaying into denormals. */
static double
best_time_l3(const struct l3case *lc, int32_t s, const void *a, void *b,
        const void *b0, void *c, int use_ob)
{
    size_t sz = (size_t)s * (size_t)s * esize(lc->t);
    double best = 1e30, total = 0.0;
    int reps = 0;

    (void)use_ob;

    if (lc->inplace) memcpy(b, b0, sz);

#ifdef TINYBLAS_HAVE_OPENBLAS
    if (use_ob) call_l3_ob(lc->t, lc->name, s, a, b, c, 1.0, 0.0); else
#endif
    call_l3_tb(lc->t, lc->name, s, a, b, c, 1.0, 0.0);

    while ((reps < 3 || total < 0.30) && reps < 50) {
        double t0, dt;

        if (lc->inplace) memcpy(b, b0, sz);

        t0 = now();
#ifdef TINYBLAS_HAVE_OPENBLAS
        if (use_ob) call_l3_ob(lc->t, lc->name, s, a, b, c, 1.0, 0.0); else
#endif
        call_l3_tb(lc->t, lc->name, s, a, b, c, 1.0, 0.0);
        dt = now() - t0;

        if (dt < best) best = dt;
        total += dt;
        ++reps;
    }

    return best;
}

static int
bench_l3(const char *filter, int32_t s)
{
    int ncase = (int)(sizeof l3list / sizeof l3list[0]);
    double gemm_g[2] = {0.0, 0.0};      /* d, then z, at this same size */
    int i, rc = 0, printed = 0;

    for (i = 0; i < ncase; ++i) {
        const struct l3case *lc = &l3list[i];
        char name[16];
        size_t es = esize(lc->t);
        size_t sz = (size_t)s * (size_t)s * es;
        double flops = flop_l3(lc->t, lc->name, s);
        void *a, *b, *b0, *c, *c0, *r;
        double tb_s, tb_g, ob_g = 0.0, err = -1.0, ref;
        int ti = (lc->t == 'd') ? 0 : 1;

        sprintf(name, "%c%s", lc->t, lc->name);

        if (filter && !strstr(name, filter)) continue;

        if (!printed) {
            printf("level 3 derived   single thread, n = %d, side LEFT, "
                   "uplo UPPER, no transpose\n\n", s);
            printf("%-8s %11s %11s %8s %9s %7s\n",
                   "routine", "tinyblas", "openblas", "ratio", "of gemm",
                   "err");
            printed = 1;
        }

        a  = malloc(sz); b  = malloc(sz); b0 = malloc(sz);
        c  = malloc(sz); c0 = malloc(sz); r  = malloc(sz);

        if (!a || !b || !b0 || !c || !c0 || !r) {
            printf("%-8s   (skipped: out of memory)\n", name);
            free(a); free(b); free(b0); free(c); free(c0); free(r);
            continue;
        }

        fill_rand(lc->t, a, (size_t)s * (size_t)s);
        fill_rand(lc->t, b0, (size_t)s * (size_t)s);
        fill_rand(lc->t, c0, (size_t)s * (size_t)s);
        if (!strcmp(lc->name, "trsm")) make_dominant(lc->t, a, s);

        memcpy(b, b0, sz);

        /* Correctness before speed, both scaling paths live, so a fast but
         * wrong wrapper cannot post a number. */
#ifdef TINYBLAS_HAVE_OPENBLAS
        {
            double bound = (double)s * eps_of(lc->t);

            memcpy(b, b0, sz);
            memcpy(c, c0, sz);
            call_l3_tb(lc->t, lc->name, s, a, b, c, 0.7, -0.3);
            memcpy(r, lc->inplace ? b : c, sz);

            memcpy(b, b0, sz);
            memcpy(c, c0, sz);
            call_l3_ob(lc->t, lc->name, s, a, b, c, 0.7, -0.3);

            err = maxdiff(lc->t, r, lc->inplace ? b : c, s, s, s) / bound;
        }
#endif

        tb_s = best_time_l3(lc, s, a, b, b0, c, 0);
        tb_g = flops / tb_s / 1e9;

#ifdef TINYBLAS_HAVE_OPENBLAS
        ob_g = flops / best_time_l3(lc, s, a, b, b0, c, 1) / 1e9;
#endif

        /* the yardstick: a gemm of the same order and element type */
        if (gemm_g[ti] == 0.0) {
            struct prob p;

            p.m = p.n = p.k = s;
            p.lda = p.ldb = p.ldc = s;
            p.a = a; p.b = b0; p.c = c;

            gemm_g[ti] = flop_mul(lc->t) * (double)s * (double)s * (double)s
                       / best_time(lc->t, &p, 0) / 1e9;
        }

        ref = gemm_g[ti];

        printf("%-8s %11.2f", name, tb_g);

        if (ob_g > 0.0) printf(" %11.2f %7.1f%%", ob_g, 100.0 * tb_g / ob_g);
        else            printf(" %11s %8s", "-", "-");

        printf(" %8.1f%%", 100.0 * tb_g / ref);

        if (err >= 0.0) printf(" %7.1f", err); else printf(" %7s", "-");

        printf("\n");
        fflush(stdout);

        if (err > ERR_LIMIT) {
            printf("\nFAIL: %s at %d is %.1fx the naive error bound, which is "
                   "wrong rather than merely rounded\n", name, s, err);
            rc = 1;
        }

        free(a); free(b); free(b0); free(c); free(c0); free(r);

        if (rc) return rc;
    }

    if (printed) printf("\n");

    return 0;
}

int main(int argc, char **argv)
{
    /* Powers of two, plus deliberate non-multiples of 6, 8 and 16 so an edge
     * path that falls off a cliff shows up as a dent in the table. */
    static const int32_t defsizes[] = {
        64, 128, 192, 256, 257, 384, 512, 768, 1000, 1023, 1024, 1025, 1536, 2048
    };
    static const char types[] = "sdcz";

    const char *filter = (argc > 1) ? argv[1] : NULL;

    /* usage: bench [routine-substring] [size ...] */
    int32_t  own[64];
    const int32_t *sizes = defsizes;
    int nsizes = (int)(sizeof defsizes / sizeof defsizes[0]);
    int i;

    if (argc > 2) {
        nsizes = 0;

        for (i = 2; i < argc && nsizes < (int)(sizeof own / sizeof own[0]); ++i)
            own[nsizes++] = (int32_t)atoi(argv[i]);

        sizes = own;
    }

#ifdef TINYBLAS_HAVE_OPENBLAS
    openblas_set_num_threads(1);
#else
    printf("note: built without OpenBLAS, so there is no baseline and no\n"
           "      independent reference. %%peak is still the honest number,\n"
           "      and make test owns correctness.\n\n");
#endif

    for (i = 0; types[i] != '\0'; ++i) {
        char name[8];

        sprintf(name, "%cgemm", types[i]);

        if (filter && !strstr(name, filter)) continue;

        if (bench_gemm(types[i], sizes, nsizes)) return 1;
    }

    for (i = 0; types[i] != '\0'; ++i) {
        char name[8];

        sprintf(name, "%cgemv", types[i]);

        if (filter && !strstr(name, filter)) continue;

        if (bench_gemv(types[i], sizes, nsizes)) return 1;
    }

    /* One order for the whole derived table: these are all wrappers over the
     * same core, so a size sweep of each would say the same thing 14 times. */
    if (bench_l3(filter, (argc > 2) ? sizes[nsizes - 1] : 1024)) return 1;

    return 0;
}
