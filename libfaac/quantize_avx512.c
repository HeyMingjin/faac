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

#if defined(HAVE_AVX2)
extern void quantize_avx2(const faac_real * __restrict xr, int * __restrict xi, int n, faac_real sfacfix);
#endif
#if defined(HAVE_SSE2)
extern void quantize_sse2(const faac_real * __restrict xr, int * __restrict xi, int n, faac_real sfacfix);
#endif

void quantize_avx512(const faac_real * __restrict xr, int * __restrict xi, int n, faac_real sfacfix)
{
    const __m512 zero = _mm512_setzero_ps();
    const __m512 sfac = _mm512_set1_ps(sfacfix);
    const __m512 magic = _mm512_set1_ps(MAGIC_NUMBER);
    // Mask to strip the sign bit (0x7FFFFFFF)
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    int cnt = 0;

    prefetch0(xr);
    prefetch0(xi);

    // Process 16 elements per iteration
    for (; cnt <= n - 16; cnt += 16)
    {
#ifdef FAAC_PRECISION_SINGLE
        __m512 x_orig = _mm512_loadu_ps((const float*)&xr[cnt]);
#else
        // Convert 16 doubles to 16 floats via two 512-bit loads (8 doubles each)
        __m256 lo_f = _mm512_cvtpd_ps(_mm512_loadu_pd(&xr[cnt]));
        __m256 hi_f = _mm512_cvtpd_ps(_mm512_loadu_pd(&xr[cnt + 8]));
        __m512 x_orig = _mm512_insertf32x8(_mm512_castps256_ps512(lo_f), hi_f, 1);
#endif
        // Capture sign (AVX512 uses mask for comparison)
        __mmask16 sign_mask = _mm512_cmp_ps_mask(x_orig, zero, _CMP_LT_OQ);
        __m512 x = _mm512_and_ps(x_orig, abs_mask);

        // Math: (x * sfac)^0.75 + magic
        // Logic: sqrt( (x*sfac) * sqrt(x*sfac) )
        x = _mm512_mul_ps(x, sfac);
        x = _mm512_mul_ps(x, _mm512_sqrt_ps(x));
        x = _mm512_sqrt_ps(x);
        x = _mm512_add_ps(x, magic);

        // Convert to integer
        __m512i xi_vec = _mm512_cvttps_epi32(x);

        // Bitwise Sign Fix: (val ^ mask) - mask
        // Convert mask to vector of -1/0 for negative/positive (AVX512DQ)
        __m512i m_int = _mm512_movm_epi32(sign_mask);
        xi_vec = _mm512_sub_epi32(_mm512_xor_epi32(xi_vec, m_int), m_int);

        _mm512_storeu_si512((__m512i*)&xi[cnt], xi_vec);
    }

    // Remainder: try AVX2 (8), then SSE2 (4), then scalar for last 0-3
    if (cnt < n) {
#if defined(HAVE_AVX2)
        if (n - cnt >= 8){
            quantize_avx2(&xr[cnt], &xi[cnt], n - cnt, sfacfix);
            cnt += 8;
        }
#endif
#if defined(HAVE_SSE2)
        if (n - cnt >= 4){
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
