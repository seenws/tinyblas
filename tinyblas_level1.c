// MIT License
// 
// Copyright (c) [year] [fullname]
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

#include "tinyblas_level1.h"

double
tinyblas_i64ddot(int64_t n, const double *x, int64_t incx, const double *y, int64_t incy)
{
    if (n <= 0) return 0.0;

    double dot = 0.0;

    if (incx < 0) x += (1 - n) * incx;
    if (incy < 0) y += (1 - n) * incy;

    if (incx == 1 && incy == 1) {
        int64_t n8 = n & ~7;

        for (int64_t i = 0; i < n8; i += 8) {
            dot += x[i]     * y[i]
                 + x[i + 1] * y[i + 1]
                 + x[i + 2] * y[i + 2]
                 + x[i + 3] * y[i + 3]
                 + x[i + 4] * y[i + 4]
                 + x[i + 5] * y[i + 5]
                 + x[i + 6] * y[i + 6]
                 + x[i + 7] * y[i + 7];
        }

        // tail loop
        for (int64_t i = n8; i < n; ++i) {
            dot += x[i] * y[i];
        }

        return dot;
    }

    int64_t ix = 0;
    int64_t iy = 0;

    for (int64_t i = 0; i < n; ++i) {
        dot += x[ix] * y[iy];
        ix  += incx;
        iy  += incy;
    }

    return dot;
}
