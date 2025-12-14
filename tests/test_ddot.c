#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "../headers/tinyblas_level1.h"

#define ASSERT_NEAR(a, b, eps) do { \
    if (fabs((a) - (b)) > (eps)) { \
        printf("FAIL: %s:%d  got=%g expected=%g\n", \
               __FILE__, __LINE__, (a), (b)); \
        return 1; \
    } \
    printf("PASS: got %g expected %g\n", (a),(b)); \
} while (0)

int main(void) {
    /* Test 1: basic */
    {
        double x[] = {1, 2, 3, 4};
        double y[] = {5, 6, 7, 8};
        double r = tinyblas_i64ddot(4, x, 1, y, 1);
        ASSERT_NEAR(r, 70.0, 1e-12); // 1*5+2*6+3*7+4*8
    }

    /* Test 2: non-unit stride */
    {
        double x[] = {1, 99, 2, 99, 3};
        double y[] = {4, 99, 5, 99, 6};
        double r = tinyblas_i64ddot(3, x, 2, y, 2);
        ASSERT_NEAR(r, 32.0, 1e-12); // 1*4+2*5+3*6
    }

    /* Test 3: negative stride */
    {
        double x[] = {1, 2, 3, 4};
        double y[] = {5, 6, 7, 8};
        double r = tinyblas_i64ddot(4, x, -1, y, 1);
        ASSERT_NEAR(r, 60.0, 1e-12);
    }

    /* Test 4: zero length */
    {
        double x[] = {1, 2, 3};
        double y[] = {4, 5, 6};
        double r = tinyblas_i64ddot(0, x, 1, y, 1);
        ASSERT_NEAR(r, 0.0, 0.0);
    }

    /* Test 5: incx == 0 */
    {
        double x[] = {2};
        double y[] = {1, 2, 3};
        double r = tinyblas_i64ddot(3, x, 0, y, 1);
        ASSERT_NEAR(r, 12.0, 1e-12); // 2*(1+2+3)
    }

    /* incx == 0 test */
    {
        double x[] = {3.0};
        double y[] = {1.0, 2.0, 3.0};
        double r = tinyblas_i64ddot(3, x, 0, y, 1);
        ASSERT_NEAR(r, 18.0, 1e-12); // 3*(1+2+3)
    }

    printf("ddot: all tests passed\n");
    return 0;
}

