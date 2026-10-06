/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include <assert.h>
#include <immintrin.h>  // AVX2
#include "avm_dsp/x86/mem_sse2.h"
#include "avm_dsp/x86/synonyms.h"
#include "avm_dsp/x86/synonyms_avx2.h"
#include "avm_dsp/x86/transpose_sse2.h"

#include "config/av2_rtcd.h"
#include "av2/common/restoration.h"
#include "av2/encoder/pickrst.h"

void av2_wienerns_accumulate_tap_diff_1x16_avx2(int16_t *feat,
                                                const uint16_t *nbr,
                                                const uint16_t *center) {
  const __m256i vnbr = _mm256_loadu_si256((const __m256i *)nbr);
  const __m256i vcenter = _mm256_loadu_si256((const __m256i *)center);
  const __m256i vdiff = _mm256_sub_epi16(vnbr, vcenter);
  __m256i vfeat = _mm256_loadu_si256((const __m256i *)feat);
  vfeat = _mm256_add_epi16(vfeat, vdiff);
  _mm256_storeu_si256((__m256i *)feat, vfeat);
}

static INLINE int32_t hsum_epi32_avx2(__m256i v) {
  const __m128i lo = _mm256_castsi256_si128(v);
  const __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i sum = _mm_add_epi32(lo, hi);
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
  return _mm_cvtsi128_si32(sum);
}

static INLINE __m128i hsum4_epi32_avx2(__m256i p0, __m256i p1, __m256i p2,
                                       __m256i p3) {
  const __m256i h01 = _mm256_hadd_epi32(p0, p1);
  const __m256i h23 = _mm256_hadd_epi32(p2, p3);
  const __m256i h = _mm256_hadd_epi32(h01, h23);
  return _mm_add_epi32(_mm256_castsi256_si128(h),
                       _mm256_extracti128_si256(h, 1));
}

static INLINE void add4_epi64_avx2(int64_t *dst, __m128i sums) {
  const __m256i acc = _mm256_loadu_si256((const __m256i *)dst);
  _mm256_storeu_si256((__m256i *)dst,
                      _mm256_add_epi64(acc, _mm256_cvtepi32_epi64(sums)));
}

static AVM_FORCE_INLINE void accumulate_row_avx2(int64_t *dst, const __m256i *v,
                                                 const __m256i *vfeat,
                                                 int count) {
  const __m256i vk = *v;
  for (int l = 0; l < count; l += 4) {
    const __m256i p0 = _mm256_madd_epi16(vk, vfeat[l]);
    const __m256i p1 = _mm256_madd_epi16(vk, vfeat[l + 1]);
    const __m256i p2 = _mm256_madd_epi16(vk, vfeat[l + 2]);
    const __m256i p3 = _mm256_madd_epi16(vk, vfeat[l + 3]);
    add4_epi64_avx2(dst + l, hsum4_epi32_avx2(p0, p1, p2, p3));
  }
}

static INLINE void accumulate_normal_equations_avx2(int64_t *A_base,
                                                    int64_t *b_base,
                                                    const __m256i *vfeat,
                                                    const __m256i *verr,
                                                    int num_feat) {
  for (int k = 0; k < num_feat; ++k) {
    accumulate_row_avx2(A_base + k * num_feat, &vfeat[k], vfeat, k + 1);
  }
  accumulate_row_avx2(b_base, verr, vfeat, num_feat);
}

void av2_wienerns_accumulate_correlation_1x16_avx2(int64_t *A_base,
                                                   int64_t *b_base,
                                                   const int16_t *feat,
                                                   const int16_t *err,
                                                   int num_feat) {
  const __m256i verr = _mm256_loadu_si256((const __m256i *)err);
  __m256i vfeat[WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD];
  for (int k = 0; k < num_feat; ++k) {
    vfeat[k] =
        _mm256_loadu_si256((const __m256i *)(feat + k * WIENERNS_STATS_BATCH));
  }
  for (int k = num_feat; k < num_feat + WIENERNS_STATS_PAD; ++k) {
    vfeat[k] = _mm256_setzero_si256();
  }
  accumulate_normal_equations_avx2(A_base, b_base, vfeat, &verr, num_feat);
}

int64_t av2_wienerns_compute_err_and_sse_mx16_avx2(
    int16_t *err, const uint16_t *src, int src_stride, const uint16_t *dgd,
    int dgd_stride, int num_rows) {
  assert(num_rows <= MI_SIZE);
  __m256i vsum = _mm256_setzero_si256();
  for (int i = 0; i < num_rows; ++i) {
    const __m256i vsrc =
        _mm256_loadu_si256((const __m256i *)(src + i * src_stride));
    const __m256i vdgd =
        _mm256_loadu_si256((const __m256i *)(dgd + i * dgd_stride));
    const __m256i verr = _mm256_sub_epi16(vsrc, vdgd);
    _mm256_storeu_si256((__m256i *)(err + i * 16), verr);
    vsum = _mm256_add_epi32(vsum, _mm256_madd_epi16(verr, verr));
  }
  return (int64_t)hsum_epi32_avx2(vsum);
}

static INLINE __m256i load_4x4_avx2(const int16_t *base, int row_stride) {
  const __m128i row01 = _mm_unpacklo_epi64(
      _mm_loadl_epi64((const __m128i *)(base + 0 * row_stride)),
      _mm_loadl_epi64((const __m128i *)(base + 1 * row_stride)));
  const __m128i row23 = _mm_unpacklo_epi64(
      _mm_loadl_epi64((const __m128i *)(base + 2 * row_stride)),
      _mm_loadl_epi64((const __m128i *)(base + 3 * row_stride)));
  return _mm256_inserti128_si256(_mm256_castsi128_si256(row01), row23, 1);
}

void av2_wienerns_accumulate_correlation_4x4_avx2(int64_t *A_base,
                                                  int64_t *b_base,
                                                  const int16_t *feat,
                                                  const int16_t *err,
                                                  int num_feat) {
  const int row_stride = WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH;
  const __m256i verr = load_4x4_avx2(err, WIENERNS_STATS_BATCH);
  __m256i vfeat[WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD];
  for (int k = 0; k < num_feat; ++k) {
    vfeat[k] = load_4x4_avx2(feat + k * WIENERNS_STATS_BATCH, row_stride);
  }
  for (int k = num_feat; k < num_feat + WIENERNS_STATS_PAD; ++k) {
    vfeat[k] = _mm256_setzero_si256();
  }
  accumulate_normal_equations_avx2(A_base, b_base, vfeat, &verr, num_feat);
}
