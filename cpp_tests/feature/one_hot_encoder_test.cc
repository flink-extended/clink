/*
 * Copyright 2021 The Clink Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fstream"
#include <cstdlib>
#include <string>
#include <sys/stat.h>

#include "clink/cpp_tests/test_util.h"
#include "clink/feature/one_hot_encoder.h"
#include "nlohmann/json.hpp"
#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(OneHotEncoderTest, Param) {
  clink::OneHotEncoderModel model;
  model.setDropLast(false);
  EXPECT_EQ(model.getDropLast(), false);
  model.setDropLast(true);
  EXPECT_EQ(model.getDropLast(), true);
}

TEST(OneHotEncoderTest, Transform) {
  clink::OneHotEncoderModelDataProto model_data;
  model_data.add_featuresizes(2);
  model_data.add_featuresizes(3);
  std::string model_data_str;
  model_data.SerializeToString(&model_data_str);

  clink::OneHotEncoderModel model;
  model.setDropLast(false);
  llvm::Error err = model.setModelData(model_data_str);
  EXPECT_EQ(!err, true);

  auto vector = model.transform(1, 0);
  EXPECT_EQ(!vector.takeError(), true);
  EXPECT_EQ(vector->get(1).get(), 1.0);

  auto invalid_value_vector = model.transform(1, 5);
  EXPECT_EQ(!invalid_value_vector.takeError(), false);

  auto invalid_index_vector = model.transform(5, 0);
  EXPECT_EQ(!invalid_index_vector.takeError(), false);
}

TEST(OneHotEncoderTest, Load) {
  std::string dir_name = clink::test::createTemporaryFolder();

  nlohmann::json params;
  params["paramMap"]["dropLast"] = "false";

  std::ofstream params_output(dir_name + "/metadata");
  params_output << params;
  params_output.close();

  clink::OneHotEncoderModelDataProto model_data;
  model_data.add_featuresizes(2);
  model_data.add_featuresizes(3);

  mkdir((dir_name + "/data").c_str(), S_IRWXU);
  std::ofstream model_data_output(dir_name + "/data/" +
                                  clink::test::generateRandomString());
  model_data.SerializeToOstream(&model_data_output);
  model_data_output.close();

  auto model = clink::OneHotEncoderModel::load(dir_name);
  EXPECT_EQ(!model.takeError(), true);

  auto vector = model->transform(1, 0);
  EXPECT_EQ(!vector.takeError(), true);
  EXPECT_EQ(vector->get(1).get(), 1.0);

  clink::test::deleteFolderRecursively(dir_name);
}

} // namespace
} // namespace tfrt
