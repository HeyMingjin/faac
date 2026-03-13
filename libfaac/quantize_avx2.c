/*
 * FAAC - Freeware Advanced Audio Coder
 * Copyright (C) 2026 Nils Schimmelmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.

 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <immintrin.h>
#include "faac_real.h"
#include "quantize.h"

#if defined(HAVE_SSE2)
extern void quantize_sse2(const faac_real * __restrict xr, int * __restrict xi, int n, faac_real sfacfix);
#endif

void quantize_avx2(const faac_real * __restrict xr, int * __restrict xi, int n, faac_real sfacfix)
{
    const __m256 zero = _mm256_setzero_ps();
    const __m256 sfac = _mm256_set1_ps(sfacfix);
    const __m256 magic = _mm256_set1_ps(MAGIC_NUMBER);
    // Mask to strip the sign bit (0x7FFFFFFF)
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    int cnt = 0;

    prefetch0(xr);

    // Process 8 elements per iteration
    for (; cnt <= n - 8; cnt += 8)
    {
#ifdef FAAC_PRECISION_SINGLE
        __m256 x_orig = _mm256_loadu_ps((const float*)&xr[cnt]);
#else
        // Convert 8 doubles to 8 floats via two 256-bit loads
        __m128 lo_f = _mm256_cvtpd_ps(_mm256_loadu_pd(&xr[cnt]));
        __m128 hi_f = _mm256_cvtpd_ps(_mm256_loadu_pd(&xr[cnt + 4]));
        __m256 x_orig = _mm256_insertf128_ps(_mm256_castps128_ps256(lo_f), hi_f, 1);
#endif
        // Capture sign and Absolute value
        __m256 sign_mask = _mm256_cmp_ps(x_orig, zero, _CMP_LT_OQ);
        __m256 x = _mm256_and_ps(x_orig, abs_mask);

        // Math: (x * sfac)^0.75 + magic
        // Logic: sqrt( (x*sfac) * sqrt(x*sfac) )
        x = _mm256_mul_ps(x, sfac);
        x = _mm256_mul_ps(x, _mm256_sqrt_ps(x));
        x = _mm256_sqrt_ps(x);
        x = _mm256_add_ps(x, magic);

        // Convert to integer
        __m256i xi_vec = _mm256_cvttps_epi32(x);

        // Bitwise Sign Fix: (val ^ mask) - mask
        __m256i m_int = _mm256_castps_si256(sign_mask);
        xi_vec = _mm256_sub_epi32(_mm256_xor_si256(xi_vec, m_int), m_int);

        _mm256_storeu_si256((__m256i*)&xi[cnt], xi_vec);
    }

    // Remainder: try SSE2 (4 elements) first, then scalar for last 0-3
    if (cnt < n) {
#if defined(HAVE_SSE2)
        if (n - cnt >= 4) {
            quantize_sse2(&xr[cnt], &xi[cnt], n - cnt, sfacfix);
            cnt += 4;
        }
#endif
        for (; cnt < n; cnt++)
        {
            faac_real val = xr[cnt];
            faac_real tmp = FAAC_FABS(val);
            tmp *= sfacfix;
            tmp = FAAC_SQRT(tmp * FAAC_SQRT(tmp));
            int q = (int)(tmp + (faac_real)MAGIC_NUMBER);
            xi[cnt] = (val < 0) ? -q : q;
        }
    }
}
