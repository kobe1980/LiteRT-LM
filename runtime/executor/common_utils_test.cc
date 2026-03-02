// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/executor/common_utils.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm::executor::utils {
namespace {

using ::testing::status::StatusIs;

TEST(CommonUtilsTest, ExpandBufferMiddleDim) {
  std::vector<float> src_data = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> src_shape = {2, 2, 2};
  std::vector<int> dst_shape = {2, 4, 2};
  std::vector<float> dst_data(16);
  ASSERT_OK(ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()),
                         src_shape, reinterpret_cast<uint8_t*>(dst_data.data()),
                         dst_shape, sizeof(float)));
  EXPECT_THAT(dst_data, testing::ElementsAreArray(
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
}

TEST(CommonUtilsTest, ExpandBufferLastDim) {
  std::vector<int> src_data = {1, 2, 3, 4};
  std::vector<int> src_shape = {2, 2};
  std::vector<int> dst_shape = {2, 4};
  std::vector<int> dst_data(8);
  ASSERT_OK(ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()),
                         src_shape, reinterpret_cast<uint8_t*>(dst_data.data()),
                         dst_shape, sizeof(int)));
  EXPECT_THAT(dst_data, testing::ElementsAreArray({1, 2, 0, 0, 3, 4, 0, 0}));
}

TEST(CommonUtilsTest, ExpandBufferFirstDim) {
  std::vector<int> src_data = {1, 2, 3, 4};
  std::vector<int> src_shape = {2, 2};
  std::vector<int> dst_shape = {4, 2};
  std::vector<int> dst_data(8);
  ASSERT_OK(ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()),
                         src_shape, reinterpret_cast<uint8_t*>(dst_data.data()),
                         dst_shape, sizeof(int)));
  EXPECT_THAT(dst_data, testing::ElementsAreArray({1, 2, 3, 4, 0, 0, 0, 0}));
}

TEST(CommonUtilsTest, ExpandBufferDifferentRank) {
  std::vector<int> src_data = {1, 2};
  std::vector<int> src_shape = {2};
  std::vector<int> dst_shape = {2, 1};
  std::vector<int> dst_data(2);
  EXPECT_THAT(
      ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()), src_shape,
                   reinterpret_cast<uint8_t*>(dst_data.data()), dst_shape,
                   sizeof(int)),
      StatusIs(absl::StatusCode::kInternal));
}

TEST(CommonUtilsTest, ExpandBufferMultipleDifferentDims) {
  std::vector<int> src_data = {1, 2};
  std::vector<int> src_shape = {1, 2};
  std::vector<int> dst_shape = {2, 3};
  std::vector<int> dst_data(6);
  EXPECT_THAT(
      ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()), src_shape,
                   reinterpret_cast<uint8_t*>(dst_data.data()), dst_shape,
                   sizeof(int)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Tensors differ in more than one dimension."));
}

TEST(CommonUtilsTest, ExpandBufferSmallerDestDim) {
  std::vector<int> src_data = {1, 2, 3, 4};
  std::vector<int> src_shape = {2, 2};
  std::vector<int> dst_shape = {1, 2};
  std::vector<int> dst_data(2);
  EXPECT_THAT(
      ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()), src_shape,
                   reinterpret_cast<uint8_t*>(dst_data.data()), dst_shape,
                   sizeof(int)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Destination tensor dimension is smaller than source "
               "along an axis."));
}

TEST(CommonUtilsTest, ExpandBufferNoExpansionAxis) {
  std::vector<int> src_data = {1, 2, 3, 4};
  std::vector<int> src_shape = {2, 2};
  std::vector<int> dst_shape = {2, 2};
  std::vector<int> dst_data(4);
  EXPECT_THAT(
      ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()), src_shape,
                   reinterpret_cast<uint8_t*>(dst_data.data()), dst_shape,
                   sizeof(int)),
      StatusIs(absl::StatusCode::kInvalidArgument, "No expansion axis found."));
}
}  // namespace
}  // namespace litert::lm::executor::utils
