/*
* Copyright (c) 2010 Mans Rullgard
* Copyright (c) 2014 James Yu <james.yu@linaro.org>
*
* This file is part of FFmpeg.
*
* FFmpeg is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* FFmpeg is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with FFmpeg; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

#include <arm_neon.h>
#include "config.h"

#include "libavutil/cpu.h"
#if   ARCH_AARCH64
#   include "libavutil/aarch64/cpu.h"
#elif ARCH_ARM
#   include "libavutil/arm/cpu.h"
#endif

#include "libavcodec/wmv2.h"

#define W0 2048
#define W1 2841 /* 2048*sqrt (2)*cos (1*pi/16) */
#define W2 2676 /* 2048*sqrt (2)*cos (2*pi/16) */
#define W3 2408 /* 2048*sqrt (2)*cos (3*pi/16) */
#define W4 2048 /* 2048*sqrt (2)*cos (4*pi/16) */
#define W5 1609 /* 2048*sqrt (2)*cos (5*pi/16) */
#define W6 1108 /* 2048*sqrt (2)*cos (6*pi/16) */
#define W7 565  /* 2048*sqrt (2)*cos (7*pi/16) */

static void wmv2_idct_row(short * b)
{
    int s1, s2;
    int a0, a1, a2, a3, a4, a5, a6, a7;

    /* step 1 */
    a1 = W1 * b[1] + W7 * b[7];
    a7 = W7 * b[1] - W1 * b[7];
    a5 = W5 * b[5] + W3 * b[3];
    a3 = W3 * b[5] - W5 * b[3];
    a2 = W2 * b[2] + W6 * b[6];
    a6 = W6 * b[2] - W2 * b[6];
    a0 = W0 * b[0] + W0 * b[4];
    a4 = W0 * b[0] - W0 * b[4];

    /* step 2 */
    s1 = (181 * (a1 - a5 + a7 - a3) + 128) >> 8; // 1, 3, 5, 7
    s2 = (181 * (a1 - a5 - a7 + a3) + 128) >> 8;

    /* step 3 */
    b[0] = (a0 + a2 + a1 + a5 + (1 << 7)) >> 8;
    b[1] = (a4 + a6 + s1 + (1 << 7)) >> 8;
    b[2] = (a4 - a6 + s2 + (1 << 7)) >> 8;
    b[3] = (a0 - a2 + a7 + a3 + (1 << 7)) >> 8;
    b[4] = (a0 - a2 - a7 - a3 + (1 << 7)) >> 8;
    b[5] = (a4 - a6 - s2 + (1 << 7)) >> 8;
    b[6] = (a4 + a6 - s1 + (1 << 7)) >> 8;
    b[7] = (a0 + a2 - a1 - a5 + (1 << 7)) >> 8;
}

static void wmv2_idct_col(short * b)
{
    int s1, s2;
    int a0, a1, a2, a3, a4, a5, a6, a7;

    /* step 1, with extended precision */
    a1 = (W1 * b[8 * 1] + W7 * b[8 * 7] + 4) >> 3;
    a7 = (W7 * b[8 * 1] - W1 * b[8 * 7] + 4) >> 3;
    a5 = (W5 * b[8 * 5] + W3 * b[8 * 3] + 4) >> 3;
    a3 = (W3 * b[8 * 5] - W5 * b[8 * 3] + 4) >> 3;
    a2 = (W2 * b[8 * 2] + W6 * b[8 * 6] + 4) >> 3;
    a6 = (W6 * b[8 * 2] - W2 * b[8 * 6] + 4) >> 3;
    a0 = (W0 * b[8 * 0] + W0 * b[8 * 4]) >> 3;
    a4 = (W0 * b[8 * 0] - W0 * b[8 * 4]) >> 3;

    /* step 2 */
    s1 = (181 * (a1 - a5 + a7 - a3) + 128) >> 8;
    s2 = (181 * (a1 - a5 - a7 + a3) + 128) >> 8;

    /* step 3 */
    b[8 * 0] = (a0 + a2 + a1 + a5 + (1 << 13)) >> 14;
    b[8 * 1] = (a4 + a6 + s1 + (1 << 13)) >> 14;
    b[8 * 2] = (a4 - a6 + s2 + (1 << 13)) >> 14;
    b[8 * 3] = (a0 - a2 + a7 + a3 + (1 << 13)) >> 14;

    b[8 * 4] = (a0 - a2 - a7 - a3 + (1 << 13)) >> 14;
    b[8 * 5] = (a4 - a6 - s2 + (1 << 13)) >> 14;
    b[8 * 6] = (a4 + a6 - s1 + (1 << 13)) >> 14;
    b[8 * 7] = (a0 + a2 - a1 - a5 + (1 << 13)) >> 14;
}

static void wmv2_idct_add_neon(uint8_t *dest, ptrdiff_t line_size, int16_t *block) {
    int i;

    for (i = 0; i < 64; i += 8)
        wmv2_idct_row(block + i);
    for (i = 0; i < 8; i++)
        wmv2_idct_col(block + i);

    for (i = 0; i < 8; i++) {
        uint8x8_t d = vld1_u8(dest);
        int16x8_t b = vld1q_s16(block);
        b = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(d)), b);
        d = vqmovun_s16(b);
        vst1_u8(dest, d);
        dest += line_size;
        block += 8;
    }
}

static void wmv2_idct_put_neon(uint8_t *dest, ptrdiff_t line_size, int16_t *block)
{
    int i;

    for (i = 0; i < 64; i += 8)
        wmv2_idct_row(block + i);
    for (i = 0; i < 8; i++)
        wmv2_idct_col(block + i);

    for (i = 0; i < 8; i++) {
        int16x8_t b = vld1q_s16(block);
        uint8x8_t d = vqmovun_s16(b);
        vst1_u8(dest, d);
        dest += line_size;
        block += 8;
    }
}

static void wmv2_mspel8_h_lowpass(uint8_t *dst, const uint8_t *src,
    int dstStride, int srcStride, int h)
{
    int i;

    int16x8_t const8 = vdupq_n_s16(8);
    for (i = 0; i < h; i++) {
        uint8x8_t s0 = vld1_u8(src);
        uint8x8_t s1 = vld1_u8(src + 1);
        uint8x8_t s2 = vld1_u8(src + 2);
        uint8x8_t sm1 = vld1_u8(src - 1);

        int16x8_t a = vreinterpretq_s16_u16(vmulq_n_u16(vaddl_u8(s0, s1), 9));
        int16x8_t b = vreinterpretq_s16_u16(vaddl_u8(sm1, s2));
        a = vaddq_s16(vsubq_s16(a, b), const8);
        s0 = vqshrun_n_s16(a, 4);
        vst1_u8(dst, s0);

        dst += dstStride;
        src += srcStride;
    }
}

static void wmv2_mspel8_v_lowpass(uint8_t *dst, const uint8_t *src,
    int dstStride, int srcStride, int w)
{
    int i;

    int16x8_t const8 = vdupq_n_s16(8);
    for (i = 0; i < w; i++) {
        uint8_t tmp[11] = {
            src[-srcStride],
            src[0],
            src[srcStride],
            src[2 * srcStride],
            src[3 * srcStride],
            src[4 * srcStride],
            src[5 * srcStride],
            src[6 * srcStride],
            src[7 * srcStride],
            src[8 * srcStride],
            src[9 * srcStride],
        };

        uint8x8_t s0 = vld1_u8(tmp + 1);
        uint8x8_t s1 = vld1_u8(tmp + 2);
        uint8x8_t s2 = vld1_u8(tmp + 3);
        uint8x8_t sm1 = vld1_u8(tmp);

        int16x8_t a = vreinterpretq_s16_u16(vmulq_n_u16(vaddl_u8(s0, s1), 9));
        int16x8_t b = vreinterpretq_s16_u16(vaddl_u8(sm1, s2));
        a = vaddq_s16(vsubq_s16(a, b), const8);
        s0 = vqshrun_n_s16(a, 4);
        vst1_u8(tmp, s0);
        dst[0 * dstStride] = tmp[0];
        dst[1 * dstStride] = tmp[1];
        dst[2 * dstStride] = tmp[2];
        dst[3 * dstStride] = tmp[3];
        dst[4 * dstStride] = tmp[4];
        dst[5 * dstStride] = tmp[5];
        dst[6 * dstStride] = tmp[6];
        dst[7 * dstStride] = tmp[7];

        src++;
        dst++;
    }
}

static void put_mspel8_mc10_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride)
{
    uint8_t half[64];

    wmv2_mspel8_h_lowpass(half, src, 8, stride, 8);
    ff_put_pixels8_l2_8(dst, src, half, stride, stride, 8, 8);
}

static void put_mspel8_mc20_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride)
{
    wmv2_mspel8_h_lowpass(dst, src, stride, stride, 8);
}

static void put_mspel8_mc30_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride)
{
    uint8_t half[64];

    wmv2_mspel8_h_lowpass(half, src, 8, stride, 8);
    ff_put_pixels8_l2_8(dst, src + 1, half, stride, stride, 8, 8);
}

static void put_mspel8_mc02_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride)
{
    wmv2_mspel8_v_lowpass(dst, src, stride, stride, 8);
}

static void put_mspel8_mc12_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride)
{
    uint8_t halfH[88];
    uint8_t halfV[64];
    uint8_t halfHV[64];

    wmv2_mspel8_h_lowpass(halfH, src - stride, 8, stride, 11);
    wmv2_mspel8_v_lowpass(halfV, src, 8, stride, 8);
    wmv2_mspel8_v_lowpass(halfHV, halfH + 8, 8, 8, 8);
    ff_put_pixels8_l2_8(dst, halfV, halfHV, stride, 8, 8, 8);
}

static void put_mspel8_mc32_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride)
{
    uint8_t halfH[88];
    uint8_t halfV[64];
    uint8_t halfHV[64];

    wmv2_mspel8_h_lowpass(halfH, src - stride, 8, stride, 11);
    wmv2_mspel8_v_lowpass(halfV, src + 1, 8, stride, 8);
    wmv2_mspel8_v_lowpass(halfHV, halfH + 8, 8, 8, 8);
    ff_put_pixels8_l2_8(dst, halfV, halfHV, stride, 8, 8, 8);
}

static void put_mspel8_mc22_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride)
{
    uint8_t halfH[88];

    wmv2_mspel8_h_lowpass(halfH, src - stride, 8, stride, 11);
    wmv2_mspel8_v_lowpass(dst, halfH + 8, stride, 8, 8);
}

void ff_wmv2dsp_init_neon(WMV2DSPContext *c) {
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        c->idct_add = wmv2_idct_add_neon;
        c->idct_put = wmv2_idct_put_neon;

        c->put_mspel_pixels_tab[0] = ff_put_pixels8x8_c;
        c->put_mspel_pixels_tab[1] = put_mspel8_mc10_neon;
        c->put_mspel_pixels_tab[2] = put_mspel8_mc20_neon;
        c->put_mspel_pixels_tab[3] = put_mspel8_mc30_neon;
        c->put_mspel_pixels_tab[4] = put_mspel8_mc02_neon;
        c->put_mspel_pixels_tab[5] = put_mspel8_mc12_neon;
        c->put_mspel_pixels_tab[6] = put_mspel8_mc22_neon;
        c->put_mspel_pixels_tab[7] = put_mspel8_mc32_neon;
    }
}
