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

#ifndef TINYBLAS_COMMON_H_
#define TINYBLAS_COMMON_H_

/* Conventions for levels 2 and 3, on top of the level 1 ones:
 *   - row-major only. lda/ldb/ldc are row strides; there is no layout argument
 *   - the selectors are enums, not 'N'/'T'/'U' chars: the compiler can check an
 *     enum, and a mistyped char is a runtime bug that reads the wrong triangle
 *   - m, n <= 0 is a no-op; k <= 0 still applies beta, since C := beta*C
 *   - beta == 0 means C is written and never read, so an uninitialised (or NaN)
 *     C is legal input
 *   - a, b and c must not overlap; every matrix argument is restrict
 *   - banded and packed storage are not implemented. They are historical
 *     Fortran storage conventions, the same reason rotm/rotmg is absent.
 */

/* op(A): none, transpose, conjugate transpose */
enum tinyblas_trans {
    TINYBLAS_NO_TRANS   = 0,
    TINYBLAS_TRANS      = 1,
    TINYBLAS_CONJ_TRANS = 2
};

/* which triangle of a symmetric, hermitian or triangular matrix is stored */
enum tinyblas_uplo {
    TINYBLAS_UPPER = 0,
    TINYBLAS_LOWER = 1
};

/* whether a triangular matrix has an implicit unit diagonal */
enum tinyblas_diag {
    TINYBLAS_NON_UNIT = 0,
    TINYBLAS_UNIT     = 1
};

/* which side the triangular or symmetric operand multiplies from */
enum tinyblas_side {
    TINYBLAS_LEFT  = 0,
    TINYBLAS_RIGHT = 1
};

#endif
