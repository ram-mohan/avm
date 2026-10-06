/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include <cmath>
#include <tuple>

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"

#include "config/avm_config.h"
#include "config/av2_rtcd.h"

#include "av2/common/restoration.h"
#include "av2/encoder/pickrst.h"

using libavm_test::ACMRandom;

namespace {

static const int kTestFeatCounts[] = {
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 18, 20, 25, 32
};
static const int kNumFeatTests =
    sizeof(kTestFeatCounts) / sizeof(kTestFeatCounts[0]);

typedef void (*WienernsTapDiffFunc)(int16_t *feat, const uint16_t *nbr,
                                    const uint16_t *center);

typedef std::tuple<WienernsTapDiffFunc> WienernsTapDiffParam;

class WienernsTapDiffTest
    : public ::testing::TestWithParam<WienernsTapDiffParam> {
 public:
  virtual void SetUp() {
    target_func_ = GET_PARAM(0);
    rnd_.Reset(ACMRandom::DeterministicSeed());
  }

  virtual void TearDown() { libavm_test::ClearSystemState(); }

  void RunRandomTest();
  void RunExtremeTest();
  void RunSpeedTest(int run_times);

 private:
  WienernsTapDiffFunc target_func_;
  ACMRandom rnd_;
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(WienernsTapDiffTest);

void WienernsTapDiffTest::RunRandomTest() {
  int16_t feat_ref[WIENERNS_STATS_BATCH];
  int16_t feat_tst[WIENERNS_STATS_BATCH];
  uint16_t nbr[WIENERNS_STATS_BATCH];
  uint16_t center[WIENERNS_STATS_BATCH];

  for (int iter = 0; iter < 1000 && !HasFatalFailure(); ++iter) {
    for (int l = 0; l < WIENERNS_STATS_BATCH; ++l) {
      const int16_t init_val =
          static_cast<int16_t>((rnd_.Rand16() % 4096) - 2048);
      feat_ref[l] = init_val;
      feat_tst[l] = init_val;
      nbr[l] = static_cast<uint16_t>(rnd_.Rand16() % 4096);
      center[l] = static_cast<uint16_t>(rnd_.Rand16() % 4096);
    }

    av2_wienerns_accumulate_tap_diff_1x16_c(feat_ref, nbr, center);
    target_func_(feat_tst, nbr, center);

    for (int l = 0; l < WIENERNS_STATS_BATCH; ++l) {
      ASSERT_EQ(feat_ref[l], feat_tst[l]) << "feat mismatch at index " << l;
    }
  }
}

void WienernsTapDiffTest::RunExtremeTest() {
  int16_t feat_ref[WIENERNS_STATS_BATCH];
  int16_t feat_tst[WIENERNS_STATS_BATCH];
  uint16_t nbr[WIENERNS_STATS_BATCH];
  uint16_t center[WIENERNS_STATS_BATCH];

  const uint16_t extreme_pixels[] = { 0, 1, 2047, 4095 };
  const int num_extreme = sizeof(extreme_pixels) / sizeof(extreme_pixels[0]);

  for (int e1 = 0; e1 < num_extreme; ++e1) {
    for (int e2 = 0; e2 < num_extreme; ++e2) {
      for (int l = 0; l < WIENERNS_STATS_BATCH; ++l) {
        feat_ref[l] = feat_tst[l] = 0;
        nbr[l] = (l % 2 == 0) ? extreme_pixels[e1] : extreme_pixels[e2];
        center[l] = extreme_pixels[e2];
      }

      av2_wienerns_accumulate_tap_diff_1x16_c(feat_ref, nbr, center);
      target_func_(feat_tst, nbr, center);

      for (int l = 0; l < WIENERNS_STATS_BATCH; ++l) {
        ASSERT_EQ(feat_ref[l], feat_tst[l])
            << "Extreme feat mismatch at index " << l;
      }
    }
  }
}

void WienernsTapDiffTest::RunSpeedTest(int run_times) {
  int16_t feat_ref[WIENERNS_STATS_BATCH] = { 0 };
  int16_t feat_tst[WIENERNS_STATS_BATCH] = { 0 };
  uint16_t nbr[WIENERNS_STATS_BATCH];
  uint16_t center[WIENERNS_STATS_BATCH];
  for (int l = 0; l < WIENERNS_STATS_BATCH; ++l) {
    nbr[l] = static_cast<uint16_t>(rnd_.Rand16() % 4096);
    center[l] = static_cast<uint16_t>(rnd_.Rand16() % 4096);
  }

  avm_usec_timer timer;
  avm_usec_timer_start(&timer);
  for (int i = 0; i < run_times; ++i) {
    av2_wienerns_accumulate_tap_diff_1x16_c(feat_ref, nbr, center);
  }
  avm_usec_timer_mark(&timer);
  const double time_c = static_cast<double>(avm_usec_timer_elapsed(&timer));

  avm_usec_timer_start(&timer);
  for (int i = 0; i < run_times; ++i) {
    target_func_(feat_tst, nbr, center);
  }
  avm_usec_timer_mark(&timer);
  const double time_simd = static_cast<double>(avm_usec_timer_elapsed(&timer));

  printf("tap_diff_1x16: C %7.2fus / SIMD %7.2fus (speedup: %3.2fx)\n", time_c,
         time_simd, time_c / time_simd);
}

TEST_P(WienernsTapDiffTest, RandomValues) { RunRandomTest(); }
TEST_P(WienernsTapDiffTest, ExtremeValues) { RunExtremeTest(); }
TEST_P(WienernsTapDiffTest, DISABLED_Speed) { RunSpeedTest(200000); }

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, WienernsTapDiffTest,
    ::testing::Values(av2_wienerns_accumulate_tap_diff_1x16_avx2));
#endif  // HAVE_AVX2

typedef void (*WienernsCorr1x16Func)(int64_t *A_base, int64_t *b_base,
                                     const int16_t *feat, const int16_t *err,
                                     int num_feat);

typedef std::tuple<WienernsCorr1x16Func> WienernsCorr1x16Param;

class WienernsCorr1x16Test
    : public ::testing::TestWithParam<WienernsCorr1x16Param> {
 public:
  virtual void SetUp() {
    target_func_ = GET_PARAM(0);
    rnd_.Reset(ACMRandom::DeterministicSeed());
  }

  virtual void TearDown() { libavm_test::ClearSystemState(); }

  void RunRandomTest();
  void RunExtremeTest();
  void RunSpeedTest(int run_times);

 private:
  WienernsCorr1x16Func target_func_;
  ACMRandom rnd_;
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(WienernsCorr1x16Test);

void WienernsCorr1x16Test::RunRandomTest() {
  int64_t A_ref[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX];
  int64_t A_tst[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD];
  int64_t b_ref[WIENERNS_TAPS_MAX];
  int64_t b_tst[WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD];
  int16_t feat[WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH];
  int16_t err[WIENERNS_STATS_BATCH];

  for (int f_idx = 0; f_idx < kNumFeatTests; ++f_idx) {
    const int num_feat = kTestFeatCounts[f_idx];
    if (num_feat > WIENERNS_TAPS_MAX) continue;

    for (int iter = 0; iter < 100 && !HasFatalFailure(); ++iter) {
      for (int i = 0; i < num_feat * num_feat; ++i) {
        const int64_t init_val = rnd_.Rand31() % 100000;
        A_ref[i] = init_val;
        A_tst[i] = init_val;
      }
      for (int i = 0; i < num_feat; ++i) {
        const int64_t init_val = rnd_.Rand31() % 100000;
        b_ref[i] = init_val;
        b_tst[i] = init_val;
      }
      for (int i = 0; i < num_feat * WIENERNS_STATS_BATCH; ++i) {
        feat[i] = static_cast<int16_t>((rnd_.Rand16() % 16384) - 8192);
      }
      for (int j = 0; j < WIENERNS_STATS_BATCH; ++j) {
        err[j] = static_cast<int16_t>((rnd_.Rand16() % 8192) - 4096);
      }

      av2_wienerns_accumulate_correlation_1x16_c(A_ref, b_ref, feat, err,
                                                 num_feat);
      target_func_(A_tst, b_tst, feat, err, num_feat);

      for (int k = 0; k < num_feat; ++k) {
        ASSERT_EQ(b_ref[k], b_tst[k])
            << "b mismatch at index " << k << " for num_feat=" << num_feat;
      }
      for (int i = 0; i < num_feat; ++i) {
        for (int j = 0; j <= i; ++j) {
          ASSERT_EQ(A_ref[i * num_feat + j], A_tst[i * num_feat + j])
              << "A mismatch at (" << i << ", " << j
              << ") for num_feat=" << num_feat;
        }
      }
    }
  }
}

void WienernsCorr1x16Test::RunExtremeTest() {
  int64_t A_ref[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX];
  int64_t A_tst[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD];
  int64_t b_ref[WIENERNS_TAPS_MAX];
  int64_t b_tst[WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD];
  int16_t feat[WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH];
  int16_t err[WIENERNS_STATS_BATCH];

  const int16_t extreme_vals[] = { -8192, -1, 0, 1, 8191 };
  const int num_extreme = sizeof(extreme_vals) / sizeof(extreme_vals[0]);

  for (int f_idx = 0; f_idx < kNumFeatTests; ++f_idx) {
    const int num_feat = kTestFeatCounts[f_idx];
    if (num_feat > WIENERNS_TAPS_MAX) continue;

    for (int e = 0; e < num_extreme; ++e) {
      memset(A_ref, 0, sizeof(A_ref));
      memset(A_tst, 0, sizeof(A_tst));
      memset(b_ref, 0, sizeof(b_ref));
      memset(b_tst, 0, sizeof(b_tst));
      for (int i = 0; i < num_feat * WIENERNS_STATS_BATCH; ++i) {
        feat[i] = extreme_vals[e];
      }
      for (int j = 0; j < WIENERNS_STATS_BATCH; ++j) {
        err[j] = extreme_vals[e];
      }

      av2_wienerns_accumulate_correlation_1x16_c(A_ref, b_ref, feat, err,
                                                 num_feat);
      target_func_(A_tst, b_tst, feat, err, num_feat);

      for (int k = 0; k < num_feat; ++k) {
        ASSERT_EQ(b_ref[k], b_tst[k])
            << "Extreme b mismatch for num_feat=" << num_feat;
      }
      for (int i = 0; i < num_feat; ++i) {
        for (int j = 0; j <= i; ++j) {
          ASSERT_EQ(A_ref[i * num_feat + j], A_tst[i * num_feat + j])
              << "Extreme A mismatch for num_feat=" << num_feat;
        }
      }
    }
  }
}

void WienernsCorr1x16Test::RunSpeedTest(int run_times) {
  int64_t A_ref[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX] = { 0 };
  int64_t A_tst[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD] = {
    0
  };
  int64_t b_ref[WIENERNS_TAPS_MAX] = { 0 };
  int64_t b_tst[WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD] = { 0 };
  int16_t feat[WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH];
  int16_t err[WIENERNS_STATS_BATCH];
  for (int i = 0; i < WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH; ++i) {
    feat[i] = static_cast<int16_t>((rnd_.Rand16() % 16384) - 8192);
  }
  for (int j = 0; j < WIENERNS_STATS_BATCH; ++j) {
    err[j] = static_cast<int16_t>((rnd_.Rand16() % 8192) - 4096);
  }

  for (int f_idx = 0; f_idx < kNumFeatTests; ++f_idx) {
    const int num_feat = kTestFeatCounts[f_idx];
    if (num_feat > WIENERNS_TAPS_MAX) continue;

    avm_usec_timer timer;
    avm_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      av2_wienerns_accumulate_correlation_1x16_c(A_ref, b_ref, feat, err,
                                                 num_feat);
    }
    avm_usec_timer_mark(&timer);
    const double time_c = static_cast<double>(avm_usec_timer_elapsed(&timer));

    avm_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      target_func_(A_tst, b_tst, feat, err, num_feat);
    }
    avm_usec_timer_mark(&timer);
    const double time_simd =
        static_cast<double>(avm_usec_timer_elapsed(&timer));

    printf("num_feat %2d: C %7.2fus / SIMD %7.2fus (speedup: %3.2fx)\n",
           num_feat, time_c, time_simd, time_c / time_simd);
  }
}

TEST_P(WienernsCorr1x16Test, RandomValues) { RunRandomTest(); }
TEST_P(WienernsCorr1x16Test, ExtremeValues) { RunExtremeTest(); }
TEST_P(WienernsCorr1x16Test, DISABLED_Speed) { RunSpeedTest(20000); }

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, WienernsCorr1x16Test,
    ::testing::Values(av2_wienerns_accumulate_correlation_1x16_avx2));
#endif  // HAVE_AVX2

typedef void (*WienernsCorr4x4Func)(int64_t *A_base, int64_t *b_base,
                                    const int16_t *feat, const int16_t *err,
                                    int num_feat);

typedef std::tuple<WienernsCorr4x4Func> WienernsCorr4x4Param;

class WienernsCorr4x4Test
    : public ::testing::TestWithParam<WienernsCorr4x4Param> {
 public:
  virtual void SetUp() {
    target_func_ = GET_PARAM(0);
    rnd_.Reset(ACMRandom::DeterministicSeed());
  }

  virtual void TearDown() { libavm_test::ClearSystemState(); }

  void RunRandomTest();
  void RunExtremeTest();
  void RunSpeedTest(int run_times);

 private:
  WienernsCorr4x4Func target_func_;
  ACMRandom rnd_;
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(WienernsCorr4x4Test);

static const int kCorr4x4FeatSize =
    MI_SIZE * WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH;
static const int kCorr4x4ErrSize = MI_SIZE * WIENERNS_STATS_BATCH;

void WienernsCorr4x4Test::RunRandomTest() {
  int64_t A_ref[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX];
  int64_t A_tst[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD];
  int64_t b_ref[WIENERNS_TAPS_MAX];
  int64_t b_tst[WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD];
  int16_t feat[kCorr4x4FeatSize];
  int16_t err[kCorr4x4ErrSize];

  for (int f_idx = 0; f_idx < kNumFeatTests; ++f_idx) {
    const int num_feat = kTestFeatCounts[f_idx];
    if (num_feat > WIENERNS_TAPS_MAX) continue;

    for (int block_col = 0; block_col + MI_SIZE <= WIENERNS_STATS_BATCH;
         block_col += MI_SIZE) {
      for (int iter = 0; iter < 100 && !HasFatalFailure(); ++iter) {
        for (int i = 0; i < num_feat * num_feat; ++i) {
          const int64_t init_val = rnd_.Rand31() % 100000;
          A_ref[i] = init_val;
          A_tst[i] = init_val;
        }
        for (int i = 0; i < num_feat; ++i) {
          const int64_t init_val = rnd_.Rand31() % 100000;
          b_ref[i] = init_val;
          b_tst[i] = init_val;
        }
        for (int i = 0; i < kCorr4x4FeatSize; ++i) {
          feat[i] = static_cast<int16_t>((rnd_.Rand16() % 16384) - 8192);
        }
        for (int i = 0; i < kCorr4x4ErrSize; ++i) {
          err[i] = static_cast<int16_t>((rnd_.Rand16() % 8192) - 4096);
        }

        av2_wienerns_accumulate_correlation_4x4_c(
            A_ref, b_ref, feat + block_col, err + block_col, num_feat);
        target_func_(A_tst, b_tst, feat + block_col, err + block_col,
                     num_feat);

        for (int k = 0; k < num_feat; ++k) {
          ASSERT_EQ(b_ref[k], b_tst[k])
              << "b mismatch at index " << k << " for num_feat=" << num_feat
              << " block_col=" << block_col;
        }
        for (int i = 0; i < num_feat; ++i) {
          for (int j = 0; j <= i; ++j) {
            ASSERT_EQ(A_ref[i * num_feat + j], A_tst[i * num_feat + j])
                << "A mismatch at (" << i << ", " << j
                << ") for num_feat=" << num_feat << " block_col=" << block_col;
          }
        }
      }
    }
  }
}

void WienernsCorr4x4Test::RunExtremeTest() {
  int64_t A_ref[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX];
  int64_t A_tst[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD];
  int64_t b_ref[WIENERNS_TAPS_MAX];
  int64_t b_tst[WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD];
  int16_t feat[kCorr4x4FeatSize];
  int16_t err[kCorr4x4ErrSize];

  const int16_t extreme_vals[] = { -8192, -1, 0, 1, 8191 };
  const int num_extreme = sizeof(extreme_vals) / sizeof(extreme_vals[0]);

  for (int f_idx = 0; f_idx < kNumFeatTests; ++f_idx) {
    const int num_feat = kTestFeatCounts[f_idx];
    if (num_feat > WIENERNS_TAPS_MAX) continue;

    for (int e1 = 0; e1 < num_extreme; ++e1) {
      for (int e2 = 0; e2 < num_extreme; ++e2) {
        memset(A_ref, 0, sizeof(A_ref));
        memset(A_tst, 0, sizeof(A_tst));
        memset(b_ref, 0, sizeof(b_ref));
        memset(b_tst, 0, sizeof(b_tst));
        for (int i = 0; i < kCorr4x4FeatSize; ++i) {
          feat[i] = (i % 2 == 0) ? extreme_vals[e1] : extreme_vals[e2];
        }
        for (int i = 0; i < kCorr4x4ErrSize; ++i) err[i] = extreme_vals[e1];

        av2_wienerns_accumulate_correlation_4x4_c(A_ref, b_ref, feat, err,
                                                  num_feat);
        target_func_(A_tst, b_tst, feat, err, num_feat);

        for (int k = 0; k < num_feat; ++k) {
          ASSERT_EQ(b_ref[k], b_tst[k])
              << "Extreme b mismatch at index " << k
              << " for num_feat=" << num_feat;
        }
        for (int i = 0; i < num_feat; ++i) {
          for (int j = 0; j <= i; ++j) {
            ASSERT_EQ(A_ref[i * num_feat + j], A_tst[i * num_feat + j])
                << "Extreme A mismatch at (" << i << ", " << j
                << ") for num_feat=" << num_feat;
          }
        }
      }
    }
  }
}

void WienernsCorr4x4Test::RunSpeedTest(int run_times) {
  int64_t A_ref[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX] = { 0 };
  int64_t A_tst[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD] = {
    0
  };
  int64_t b_ref[WIENERNS_TAPS_MAX] = { 0 };
  int64_t b_tst[WIENERNS_TAPS_MAX + WIENERNS_STATS_PAD] = { 0 };
  int16_t feat[kCorr4x4FeatSize];
  int16_t err[kCorr4x4ErrSize];
  for (int i = 0; i < kCorr4x4FeatSize; ++i) {
    feat[i] = static_cast<int16_t>((rnd_.Rand16() % 16384) - 8192);
  }
  for (int i = 0; i < kCorr4x4ErrSize; ++i) {
    err[i] = static_cast<int16_t>((rnd_.Rand16() % 8192) - 4096);
  }

  for (int f_idx = 0; f_idx < kNumFeatTests; ++f_idx) {
    const int num_feat = kTestFeatCounts[f_idx];
    if (num_feat > WIENERNS_TAPS_MAX) continue;

    avm_usec_timer timer;
    avm_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      av2_wienerns_accumulate_correlation_4x4_c(A_ref, b_ref, feat, err,
                                                num_feat);
    }
    avm_usec_timer_mark(&timer);
    const double time_c = static_cast<double>(avm_usec_timer_elapsed(&timer));

    avm_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      target_func_(A_tst, b_tst, feat, err, num_feat);
    }
    avm_usec_timer_mark(&timer);
    const double time_simd =
        static_cast<double>(avm_usec_timer_elapsed(&timer));

    printf("num_feat %2d: C %7.2fus / SIMD %7.2fus (speedup: %3.2fx)\n",
           num_feat, time_c, time_simd, time_c / time_simd);
  }
}

TEST_P(WienernsCorr4x4Test, RandomValues) { RunRandomTest(); }
TEST_P(WienernsCorr4x4Test, ExtremeValues) { RunExtremeTest(); }
TEST_P(WienernsCorr4x4Test, DISABLED_Speed) { RunSpeedTest(20000); }

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, WienernsCorr4x4Test,
    ::testing::Values(av2_wienerns_accumulate_correlation_4x4_avx2));
#endif  // HAVE_AVX2

typedef int64_t (*WienernsComputeErrSseFunc)(int16_t *err, const uint16_t *src,
                                             int src_stride,
                                             const uint16_t *dgd,
                                             int dgd_stride, int num_rows);

typedef std::tuple<WienernsComputeErrSseFunc> WienernsComputeErrSseParam;

class WienernsComputeErrSseTest
    : public ::testing::TestWithParam<WienernsComputeErrSseParam> {
 public:
  virtual void SetUp() {
    target_func_ = GET_PARAM(0);
    rnd_.Reset(ACMRandom::DeterministicSeed());
  }

  virtual void TearDown() { libavm_test::ClearSystemState(); }

  void RunRandomTest();
  void RunExtremeTest();
  void RunSpeedTest(int run_times);

 private:
  WienernsComputeErrSseFunc target_func_;
  ACMRandom rnd_;
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(WienernsComputeErrSseTest);

static const int kErrSseStride = 40;
static const int kErrSseMaxRows = 4;

void WienernsComputeErrSseTest::RunRandomTest() {
  int16_t err_ref[kErrSseMaxRows * WIENERNS_STATS_BATCH];
  int16_t err_tst[kErrSseMaxRows * WIENERNS_STATS_BATCH];
  uint16_t src[kErrSseMaxRows * kErrSseStride];
  uint16_t dgd[kErrSseMaxRows * kErrSseStride];

  for (int num_rows = 1; num_rows <= kErrSseMaxRows; ++num_rows) {
    for (int iter = 0; iter < 1000 && !HasFatalFailure(); ++iter) {
      memset(err_ref, 0, sizeof(err_ref));
      memset(err_tst, 0, sizeof(err_tst));
      for (int i = 0; i < kErrSseMaxRows * kErrSseStride; ++i) {
        src[i] = static_cast<uint16_t>(rnd_.Rand16() % 4096);
        dgd[i] = static_cast<uint16_t>(rnd_.Rand16() % 4096);
      }

      const int64_t sse_ref = av2_wienerns_compute_err_and_sse_mx16_c(
          err_ref, src, kErrSseStride, dgd, kErrSseStride, num_rows);
      const int64_t sse_tst = target_func_(err_tst, src, kErrSseStride, dgd,
                                           kErrSseStride, num_rows);

      ASSERT_EQ(sse_ref, sse_tst) << "sse mismatch, num_rows=" << num_rows;
      for (int l = 0; l < kErrSseMaxRows * WIENERNS_STATS_BATCH; ++l) {
        ASSERT_EQ(err_ref[l], err_tst[l])
            << "err mismatch at index " << l << ", num_rows=" << num_rows;
      }
    }
  }
}

void WienernsComputeErrSseTest::RunExtremeTest() {
  int16_t err_ref[kErrSseMaxRows * WIENERNS_STATS_BATCH];
  int16_t err_tst[kErrSseMaxRows * WIENERNS_STATS_BATCH];
  uint16_t src[kErrSseMaxRows * kErrSseStride];
  uint16_t dgd[kErrSseMaxRows * kErrSseStride];

  const uint16_t extreme_pixels[] = { 0, 1, 2047, 4095 };
  const int num_extreme = sizeof(extreme_pixels) / sizeof(extreme_pixels[0]);

  for (int num_rows = 1; num_rows <= kErrSseMaxRows; ++num_rows) {
    for (int e1 = 0; e1 < num_extreme; ++e1) {
      for (int e2 = 0; e2 < num_extreme; ++e2) {
        memset(err_ref, 0, sizeof(err_ref));
        memset(err_tst, 0, sizeof(err_tst));
        for (int i = 0; i < kErrSseMaxRows * kErrSseStride; ++i) {
          src[i] = extreme_pixels[e1];
          dgd[i] = extreme_pixels[e2];
        }

        const int64_t sse_ref = av2_wienerns_compute_err_and_sse_mx16_c(
            err_ref, src, kErrSseStride, dgd, kErrSseStride, num_rows);
        const int64_t sse_tst = target_func_(err_tst, src, kErrSseStride, dgd,
                                             kErrSseStride, num_rows);

        ASSERT_EQ(sse_ref, sse_tst)
            << "Extreme sse mismatch, num_rows=" << num_rows;
        for (int l = 0; l < kErrSseMaxRows * WIENERNS_STATS_BATCH; ++l) {
          ASSERT_EQ(err_ref[l], err_tst[l])
              << "Extreme err mismatch at index " << l
              << ", num_rows=" << num_rows;
        }
      }
    }
  }
}

void WienernsComputeErrSseTest::RunSpeedTest(int run_times) {
  int16_t err_ref[kErrSseMaxRows * WIENERNS_STATS_BATCH];
  int16_t err_tst[kErrSseMaxRows * WIENERNS_STATS_BATCH];
  uint16_t src[kErrSseMaxRows * kErrSseStride];
  uint16_t dgd[kErrSseMaxRows * kErrSseStride];
  for (int i = 0; i < kErrSseMaxRows * kErrSseStride; ++i) {
    src[i] = static_cast<uint16_t>(rnd_.Rand16() % 4096);
    dgd[i] = static_cast<uint16_t>(rnd_.Rand16() % 4096);
  }

  avm_usec_timer timer;
  avm_usec_timer_start(&timer);
  for (int i = 0; i < run_times; ++i) {
    av2_wienerns_compute_err_and_sse_mx16_c(err_ref, src, kErrSseStride, dgd,
                                            kErrSseStride, kErrSseMaxRows);
  }
  avm_usec_timer_mark(&timer);
  const double time_c = static_cast<double>(avm_usec_timer_elapsed(&timer));

  avm_usec_timer_start(&timer);
  for (int i = 0; i < run_times; ++i) {
    target_func_(err_tst, src, kErrSseStride, dgd, kErrSseStride,
                 kErrSseMaxRows);
  }
  avm_usec_timer_mark(&timer);
  const double time_simd = static_cast<double>(avm_usec_timer_elapsed(&timer));

  printf("compute_err_sse_4x16: C %7.2fus / SIMD %7.2fus (speedup: %3.2fx)\n",
         time_c, time_simd, time_c / time_simd);
}

TEST_P(WienernsComputeErrSseTest, RandomValues) { RunRandomTest(); }
TEST_P(WienernsComputeErrSseTest, ExtremeValues) { RunExtremeTest(); }
TEST_P(WienernsComputeErrSseTest, DISABLED_Speed) { RunSpeedTest(200000); }

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, WienernsComputeErrSseTest,
    ::testing::Values(av2_wienerns_compute_err_and_sse_mx16_avx2));
#endif  // HAVE_AVX2

}  // namespace
