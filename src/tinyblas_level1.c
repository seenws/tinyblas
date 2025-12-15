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

#include "../headers/tinyblas_level1.h"
#include "../headers/tinyblas_common.h"

/*
 *  Dot product with double precision arguments and accumulator
 */
double
tinyblas_i32ddot(
        int32_t n,
        const double *dx, int32_t incx,
        const double *dy, int32_t incy
) {
    if (n <= 0) return 0.0;

    double dot = 0.0;

    if (incx < 0) dx += (1 - n) * incx;
    if (incy < 0) dy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        int32_t n8 = n & ~7;

        for (int32_t i = 0; i < n16; i += 16) {
            dot += dx[i + 0]     * dy[i + 0]
                 + dx[i + 1]     * dy[i + 1]
                 + dx[i + 2]     * dy[i + 2]
                 + dx[i + 3]     * dy[i + 3]
                 + dx[i + 4]     * dy[i + 4]
                 + dx[i + 5]     * dy[i + 5]
                 + dx[i + 6]     * dy[i + 6]
                 + dx[i + 7]     * dy[i + 7]
        }

        // tail loop
        for (int32_t i = n16; i < n; ++i) {
            dot += dx[i] * dy[i];
        }

        return dot;
    }

    int32_t ix = 0;
    int32_t iy = 0;

    for (int32_t i = 0; i < n; ++i) {
        dot += dx[ix] * dy[iy];
        ix  += incx;
        iy  += incy;
    }

    return dot;
}

/*
 *  Dot product with single precision arguments and accumulator
 */
float
tinyblas_i32sdot(
        int32_t n,
        const float *dx, int32_t incx,
        const float *dy, int32_t incy
) {
    if (n <= 0) return 0.0;

    float dot = 0.0f;

    if (incx < 0) dx += (1 - n) * incx;
    if (incy < 0) dy += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        int32_t n16 = n & ~15;

        for (int32_t i = 0; i < n16; i += 16) {
            dot += dx[i + 0]     * dy[i + 0]
                 + dx[i + 1]     * dy[i + 1]
                 + dx[i + 2]     * dy[i + 2]
                 + dx[i + 3]     * dy[i + 3]
                 + dx[i + 4]     * dy[i + 4]
                 + dx[i + 5]     * dy[i + 5]
                 + dx[i + 6]     * dy[i + 6]
                 + dx[i + 7]     * dy[i + 7]
                 + dx[i + 8]     * dy[i + 8]
                 + dx[i + 9]     * dy[i + 9]
                 + dx[i + 10]    * dy[i + 10]
                 + dx[i + 11]    * dy[i + 11]
                 + dx[i + 12]    * dy[i + 12]
                 + dx[i + 13]    * dy[i + 13]
                 + dx[i + 14]    * dy[i + 14]
                 + dx[i + 15]    * dy[i + 15];
        }

        // tail loop
        for (int32_t i = n16; i < n; ++i) {
            dot += dx[i] * dy[i];
        }

        return dot;
    }

    int32_t ix = 0;
    int32_t iy = 0;

    for (int32_t i = 0; i < n; ++i) {
        dot += dx[ix] * dy[iy];
        ix  += incx;
        iy  += incy;
    }

    return dot;
}

/*
 *  Dot product with single precision arguments and double precision accumulator
 */
double
tinyblas_i32dsdot(
        int32_t n,
        const float *dx, int32_t incx,
        const float *dy, int32_t incy
) {
    if (n <= 0) return 0.0;

    double dot = 0.0;

    if (incx < 0) dx += (1 - n) * incx;
    if (incy < 0) dy += (1 - n) * incy;

        // tail loop
        for (int32_t i = n16; i < n; ++i) {
            dot += (double)dx[i] * (double)dy[i];
        }

        return dot;
    }

    int32_t ix = 0;
    int32_t iy = 0;

    for (int32_t i = 0; i < n; ++i) {
        dot += dx[ix] * dy[iy];
        ix  += incx;
        iy  += incy;
    }

    return dot;
}

