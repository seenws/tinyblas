#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <complex.h>

#include "tinyblas.h"

#define CHECK(a, b, eps) do { \
    if (!((fabs((double)(a) - (double)(b)) <= (eps)))) { \
        printf("FAIL: %s:%d  got=%g expected=%g\n", \
               __FILE__, __LINE__, (double)(a), (double)(b)); \
        return 1; \
    } \
} while (0)

/* relative form, for values too big to compare with an absolute epsilon */
#define CHECK_REL(a, b, eps) CHECK((double)(a) / (double)(b), 1.0, (eps))

#define CHECK_C(z, re, im, eps) do { \
    CHECK(creal((double complex)(z)), (re), (eps)); \
    CHECK(cimag((double complex)(z)), (im), (eps)); \
} while (0)

int main(void) {
    float          x[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float          y[]  = {5.0f, 6.0f, 7.0f, 8.0f};
    double         dx[] = {1.0, 2.0, 3.0, 4.0};
    double         dy[] = {5.0, 6.0, 7.0, 8.0};
    float complex  cx[] = {1.0f + 2.0f * I, 5.0f + 6.0f * I};
    float complex  cy[] = {3.0f + 4.0f * I, 7.0f + 8.0f * I};
    double complex zx[] = {1.0 + 2.0 * I, 5.0 + 6.0 * I};
    double complex zy[] = {3.0 + 4.0 * I, 7.0 + 8.0 * I};

    /* --- dot products --------------------------------------------------- */

    CHECK(tinyblas_ddot (4, dx, 1, dy, 1), 70.0, 1e-12);        /* 5+12+21+32 */
    CHECK(tinyblas_sdot (4, x,  1, y,  1), 70.0, 1e-4);
    CHECK(tinyblas_dsdot(4, x,  1, y,  1), 70.0, 1e-12);
    CHECK_C(tinyblas_cdotu(2, cx, 1, cy, 1), -18.0, 92.0, 1e-4);

    CHECK(tinyblas_ddot(2, dx, 2, dy, 2), 26.0, 1e-12);         /* 1*5 + 3*7 */
    CHECK(tinyblas_ddot(4, dx, -1, dy, 1), 60.0, 1e-12);        /* x reversed */
    CHECK(tinyblas_ddot(0, dx, 1, dy, 1), 0.0, 0.0);
    CHECK(tinyblas_ddot(3, dx, 0, dy, 1), 18.0, 1e-12);         /* incx 0: 1*(5+6+7) */

    /* n = 19 is not a multiple of 4, 8 or 16, so this exercises the tail of
     * the vectorized unit-stride loop. sum(i*i) for i in 1..19 = 2470 */
    {
        float  tx[19], ty[19];
        double tdx[19], tdy[19];

        for (int32_t i = 0; i < 19; ++i) {
            tx[i]  = ty[i]  = (float)(i + 1);
            tdx[i] = tdy[i] = (double)(i + 1);
        }

        CHECK(tinyblas_ddot (19, tdx, 1, tdy, 1), 2470.0, 1e-9);
        CHECK(tinyblas_sdot (19, tx,  1, ty,  1), 2470.0, 1e-2);
        CHECK(tinyblas_dsdot(19, tx,  1, ty,  1), 2470.0, 1e-9);

        /* the strided path must agree: sum(i*i) for i in 1,3,..,19 = 1330 */
        CHECK(tinyblas_ddot (10, tdx, 2, tdy, 2), 1330.0, 1e-9);
        CHECK(tinyblas_dsdot(10, tx,  2, ty,  2), 1330.0, 1e-9);

        /* walking x backwards: sum(j*(20-j)) for j in 1..19 = 1330 */
        CHECK(tinyblas_ddot(19, tdx, -1, tdy, 1), 1330.0, 1e-9);
    }

    CHECK(tinyblas_sdsdot(4, 10.0f, x, 1, y, 1), 80.0, 1e-4);   /* 70 + sb */
    CHECK(tinyblas_sdsdot(0, 10.0f, x, 1, y, 1), 10.0, 0.0);

    /* conj(1+2i)(3+4i) + conj(5+6i)(7+8i) = 11-2i + 83-2i */
    CHECK_C(tinyblas_cdotc(2, cx, 1, cy, 1),  94.0, -4.0, 1e-4);
    CHECK_C(tinyblas_zdotu(2, zx, 1, zy, 1), -18.0, 92.0, 1e-12);
    CHECK_C(tinyblas_zdotc(2, zx, 1, zy, 1),  94.0, -4.0, 1e-12);

    /* --- norms ---------------------------------------------------------- */
    {
        float          p[] = {3.0f, 4.0f};
        double         dp[] = {3.0, 4.0};
        float complex  cp[] = {3.0f + 4.0f * I};
        double complex zp[] = {1.0 + 2.0 * I, 2.0 + 4.0 * I};

        CHECK(tinyblas_snrm2 (2, p,  1), 5.0, 1e-5);
        CHECK(tinyblas_dnrm2 (2, dp, 1), 5.0, 1e-12);
        CHECK(tinyblas_scnrm2(1, cp, 1), 5.0, 1e-5);
        CHECK(tinyblas_dznrm2(2, zp, 1), 5.0, 1e-12);
        CHECK(tinyblas_dnrm2 (0, dp, 1), 0.0, 0.0);
    }

    /* no overflow: the squares do not fit the input type */
    {
        float  big[]  = {3.0e19f, 4.0e19f};
        double dbig[] = {3.0e200, 4.0e200};

        CHECK_REL(tinyblas_snrm2(2, big,  1), 5.0e19,  1e-5);
        CHECK_REL(tinyblas_dnrm2(2, dbig, 1), 5.0e200, 1e-12);
    }

    /* --- absolute sums -------------------------------------------------- */
    {
        float          p[] = {1.0f, -2.0f, 3.0f};
        double         dp[] = {1.0, -2.0, 3.0};
        float complex  cp[] = {1.0f + 2.0f * I, -3.0f - 4.0f * I};
        double complex zp[] = {1.0 + 2.0 * I, -3.0 - 4.0 * I};

        CHECK(tinyblas_sasum (3, p,  1), 6.0,  1e-5);
        CHECK(tinyblas_dasum (3, dp, 1), 6.0,  1e-12);
        CHECK(tinyblas_scasum(2, cp, 1), 10.0, 1e-5);
        CHECK(tinyblas_dzasum(2, zp, 1), 10.0, 1e-12);
        CHECK(tinyblas_dasum (2, dp, 2), 4.0,  1e-12);   /* strided: 1 + 3 */
    }

    /* --- index of max magnitude, 0-based -------------------------------- */
    {
        float          p[] = {1.0f, -5.0f, 3.0f};
        double         dp[] = {1.0, -5.0, 3.0};
        float complex  cp[] = {1.0f + 1.0f * I, 0.0f, -3.0f};
        double complex zp[] = {1.0 + 1.0 * I, 0.0, -3.0};

        CHECK(tinyblas_isamax(3, p,  1),  1, 0.0);
        CHECK(tinyblas_idamax(3, dp, 1),  1, 0.0);
        CHECK(tinyblas_icamax(3, cp, 1),  2, 0.0);
        CHECK(tinyblas_izamax(3, zp, 1),  2, 0.0);
        CHECK(tinyblas_idamax(3, dp, -1), 1, 0.0);   /* reversed: {3,-5,1} */
        CHECK(tinyblas_idamax(0, dp, 1), -1, 0.0);
    }

    /* --- swap / copy ---------------------------------------------------- */

    tinyblas_dswap(4, dx, 1, dy, 1);
    CHECK(dx[0], 5.0, 0.0);
    CHECK(dy[3], 4.0, 0.0);
    tinyblas_dswap(4, dx, 1, dy, 1);
    CHECK(dx[0], 1.0, 0.0);

    tinyblas_sswap(2, x, 1, y, 1);
    CHECK(x[0], 5.0, 0.0);
    tinyblas_sswap(2, x, 1, y, 1);
    CHECK(x[0], 1.0, 0.0);

    tinyblas_cswap(2, cx, 1, cy, 1);
    CHECK_C(cx[0], 3.0, 4.0, 0.0);
    tinyblas_cswap(2, cx, 1, cy, 1);
    CHECK_C(cx[0], 1.0, 2.0, 0.0);

    tinyblas_zswap(2, zx, 1, zy, 1);
    CHECK_C(zx[0], 3.0, 4.0, 0.0);
    tinyblas_zswap(2, zx, 1, zy, 1);
    CHECK_C(zx[0], 1.0, 2.0, 0.0);

    {
        double out[4] = {0.0, 0.0, 0.0, 0.0};
        float  fout[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float complex  cout[2] = {0.0f, 0.0f};
        double complex zout[2] = {0.0, 0.0};

        tinyblas_dcopy(4, dx, -1, out, 1);           /* reversed copy */
        CHECK(out[0], 4.0, 0.0);
        CHECK(out[3], 1.0, 0.0);

        tinyblas_scopy(4, x, 1, fout, 1);
        CHECK(fout[3], 4.0, 0.0);

        tinyblas_ccopy(2, cx, 1, cout, 1);
        CHECK_C(cout[1], 5.0, 6.0, 0.0);

        tinyblas_zcopy(2, zx, 1, zout, 1);
        CHECK_C(zout[1], 5.0, 6.0, 0.0);
    }

    /* --- axpy ------------------------------------------------------------ */
    {
        double a[] = {1.0, 2.0, 3.0};
        double b[] = {1.0, 1.0, 1.0};
        float  fa[] = {1.0f, 2.0f, 3.0f};
        float  fb[] = {1.0f, 1.0f, 1.0f};
        float complex  ca[] = {1.0f + 1.0f * I};
        float complex  cb[] = {0.0f};
        double complex za[] = {1.0 + 1.0 * I};
        double complex zb[] = {0.0};

        tinyblas_daxpy(3, 2.0, a, 1, b, 1);
        CHECK(b[2], 7.0, 1e-12);
        tinyblas_daxpy(3, 0.0, a, 1, b, 1);          /* alpha == 0 is a no-op */
        CHECK(b[2], 7.0, 1e-12);
        tinyblas_daxpy(2, 1.0, a, 2, b, 2);          /* strided: b0+=1, b2+=3 */
        CHECK(b[0], 4.0, 1e-12);
        CHECK(b[2], 10.0, 1e-12);

        tinyblas_saxpy(3, 2.0f, fa, 1, fb, 1);
        CHECK(fb[2], 7.0, 1e-5);

        tinyblas_caxpy(1, 2.0f * I, ca, 1, cb, 1);   /* 2i*(1+i) = -2+2i */
        CHECK_C(cb[0], -2.0, 2.0, 1e-5);

        tinyblas_zaxpy(1, 2.0 * I, za, 1, zb, 1);
        CHECK_C(zb[0], -2.0, 2.0, 1e-12);
    }

    /* --- scal ------------------------------------------------------------ */
    {
        double a[] = {1.0, 2.0, 3.0};
        float  fa[] = {1.0f, 2.0f};
        float complex  ca[] = {1.0f + 1.0f * I, 2.0f};
        double complex za[] = {1.0 + 1.0 * I};

        tinyblas_dscal(3, 2.0, a, 1);
        CHECK(a[2], 6.0, 1e-12);
        tinyblas_dscal(2, 0.5, a, 2);                /* strided: a0 and a2 */
        CHECK(a[0], 1.0, 1e-12);
        CHECK(a[1], 4.0, 1e-12);
        CHECK(a[2], 3.0, 1e-12);

        tinyblas_sscal(2, 3.0f, fa, 1);
        CHECK(fa[1], 6.0, 1e-5);

        tinyblas_cscal(1, 1.0f * I, ca, 1);          /* i*(1+i) = -1+i */
        CHECK_C(ca[0], -1.0, 1.0, 1e-5);
        tinyblas_csscal(2, 2.0f, ca, 1);
        CHECK_C(ca[0], -2.0, 2.0, 1e-5);
        CHECK_C(ca[1], 4.0, 0.0, 1e-5);

        tinyblas_zscal(1, 1.0 * I, za, 1);
        CHECK_C(za[0], -1.0, 1.0, 1e-12);
        tinyblas_zdscal(1, 2.0, za, 1);
        CHECK_C(za[0], -2.0, 2.0, 1e-12);
    }

    /* --- rotg: c*a + s*b == r and -conj(s)*a + c*b == 0 ------------------ */
    {
        float  c, s, r;
        double dc, ds, dr;

        tinyblas_drotg(3.0, 4.0, &dc, &ds, &dr);
        CHECK(dc, 0.6, 1e-12);
        CHECK(ds, 0.8, 1e-12);
        CHECK(dr, 5.0, 1e-12);
        CHECK(-ds * 3.0 + dc * 4.0, 0.0, 1e-12);

        tinyblas_srotg(-3.0f, -4.0f, &c, &s, &r);    /* r follows the larger input */
        CHECK(r, -5.0, 1e-5);
        CHECK(c * -3.0f + s * -4.0f, -5.0, 1e-5);

        tinyblas_drotg(0.0, 0.0, &dc, &ds, &dr);
        CHECK(dc, 1.0, 0.0);
        CHECK(ds, 0.0, 0.0);
        CHECK(dr, 0.0, 0.0);
    }
    {
        float complex  a = 1.0f + 1.0f * I, b = 2.0f - 1.0f * I, cs, cr;
        double complex da = 1.0 + 1.0 * I, db = 2.0 - 1.0 * I, zs, zr;
        float  c;
        double dc;

        tinyblas_crotg(a, b, &c, &cs, &cr);
        CHECK_C(c * a + cs * b, creal(cr), cimag(cr), 1e-5);
        CHECK_C(-conjf(cs) * a + c * b, 0.0, 0.0, 1e-5);

        tinyblas_zrotg(da, db, &dc, &zs, &zr);
        CHECK_C(dc * da + zs * db, creal(zr), cimag(zr), 1e-12);
        CHECK_C(-conj(zs) * da + dc * db, 0.0, 0.0, 1e-12);

        tinyblas_zrotg(0.0, db, &dc, &zs, &zr);      /* a == 0 */
        CHECK(dc, 0.0, 0.0);
        CHECK_C(zr, cabs(db), 0.0, 1e-12);

        tinyblas_zrotg(da, 0.0, &dc, &zs, &zr);      /* b == 0 */
        CHECK(dc, 1.0, 0.0);
        CHECK_C(zr, 1.0, 1.0, 1e-12);
    }

    /* --- rot -------------------------------------------------------------- */
    {
        double a[] = {1.0, 0.0};
        double b[] = {0.0, 1.0};
        float  fa[] = {1.0f, 0.0f};
        float  fb[] = {0.0f, 1.0f};
        float complex  ca[] = {1.0f};
        float complex  cb[] = {1.0f * I};
        double complex za[] = {1.0};
        double complex zb[] = {1.0 * I};

        tinyblas_drot(2, a, 1, b, 1, 0.6, 0.8);
        CHECK(a[0],  0.6, 1e-12);
        CHECK(a[1],  0.8, 1e-12);
        CHECK(b[0], -0.8, 1e-12);
        CHECK(b[1],  0.6, 1e-12);

        tinyblas_srot(2, fa, 1, fb, 1, 0.6f, 0.8f);
        CHECK(fa[0],  0.6, 1e-5);
        CHECK(fb[0], -0.8, 1e-5);

        tinyblas_csrot(1, ca, 1, cb, 1, 0.6f, 0.8f);
        CHECK_C(ca[0], 0.6, 0.8, 1e-5);
        CHECK_C(cb[0], -0.8, 0.6, 1e-5);

        tinyblas_zdrot(1, za, 1, zb, 1, 0.6, 0.8);
        CHECK_C(za[0], 0.6, 0.8, 1e-12);
        CHECK_C(zb[0], -0.8, 0.6, 1e-12);
    }

    printf("level1: all tests passed\n");
    return 0;
}
