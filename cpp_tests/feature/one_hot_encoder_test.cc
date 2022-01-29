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

#include "clink/feature/one_hot_encoder.h"

#include "clink/cpp_tests/test_util.h"
#include "gtest/gtest.h"

namespace clink {

namespace {

class OneHotEncoderTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    assert(host_context == nullptr);
    host_context =
        CreateHostContext("mstd", tfrt::HostAllocatorType::kLeakCheckMalloc)
            .release();
    assert(mlir_context == nullptr);
    mlir_context = new MLIRContext();
    mlir_context->allowUnregisteredDialects();
    mlir_context->printOpOnDiagnostic(true);
    mlir::DialectRegistry registry;
    registry.insert<clink::ClinkDialect>();
    registry.insert<tfrt::compiler::TFRTDialect>();
    mlir_context->appendDialectRegistry(registry);
  }

  static void TearDownTestSuite() {
    delete host_context;
    host_context = nullptr;
    delete mlir_context;
    mlir_context = nullptr;
  }

  static tfrt::HostContext *host_context;
  static MLIRContext *mlir_context;
};

tfrt::HostContext *OneHotEncoderTest::host_context = nullptr;

MLIRContext *OneHotEncoderTest::mlir_context = nullptr;

TEST_F(OneHotEncoderTest, Param) {
  RCReference<OneHotEncoderModel> model =
      tfrt::TakeRef(host_context->Construct<OneHotEncoderModel>(host_context));
  model->setDropLast(false);
  EXPECT_FALSE(model->getDropLast());
  model->setDropLast(true);
  EXPECT_TRUE(model->getDropLast());
}

TEST_F(OneHotEncoderTest, Transform) {
  OneHotEncoderModelDataProto model_data;
  model_data.add_featuresizes(2);
  model_data.add_featuresizes(3);
  std::string model_data_str;
  model_data.SerializeToString(&model_data_str);

  RCReference<OneHotEncoderModel> model =
      tfrt::TakeRef(host_context->Construct<OneHotEncoderModel>(host_context));
  model->setDropLast(false);
  llvm::Error err = model->setModelData(std::move(model_data_str));
  EXPECT_FALSE(err);

  SparseVector expected_vector(2);
  expected_vector.set(1, 1.0);
  auto actual_vector = model->transform(1, 0);
  EXPECT_EQ(actual_vector.get(), expected_vector);

  auto invalid_value_vector = model->transform(1, 5);
  EXPECT_TRUE(invalid_value_vector.IsError());

  auto invalid_index_vector = model->transform(5, 0);
  EXPECT_TRUE(invalid_index_vector.IsError());
}

TEST_F(OneHotEncoderTest, Load) {
  test::TemporaryFolder tmp_folder;

  nlohmann::json params;
  params["paramMap"]["dropLast"] = "false";

  OneHotEncoderModelDataProto model_data;
  model_data.add_featuresizes(2);
  model_data.add_featuresizes(3);

  test::saveMetaDataModelData(tmp_folder.getAbsolutePath(), params, model_data);

  auto model =
      OneHotEncoderModel::load(tmp_folder.getAbsolutePath(), host_context);
  EXPECT_FALSE((bool)model.takeError());

  SparseVector expected_vector(2);
  expected_vector.set(1, 1.0);
  auto actual_vector = model.get()->transform(1, 0);
  EXPECT_EQ(actual_vector.get(), expected_vector);
}

TEST_F(OneHotEncoderTest, Mlir) {
  test::TemporaryFolder tmp_folder;

  nlohmann::json params;
  params["paramMap"]["dropLast"] = "false";

  OneHotEncoderModelDataProto model_data;
  model_data.add_featuresizes(2);
  model_data.add_featuresizes(3);

  test::saveMetaDataModelData(tmp_folder.getAbsolutePath(), params, model_data);

  // TODO: Separate the load process that is triggered only once and the
  // repeatedly triggered transform process into different scripts.
  auto mlir_script = R"mlir(
    func @main(%path: !tfrt.string, %value: i32, %column_index: i32) -> !clink.vector {
      %model = clink.onehotencoder_load %path
      %vector = clink.onehotencoder_transform %model, %value, %column_index
      tfrt.return %vector : !clink.vector
    }
  )mlir";

  llvm::SmallVector<RCReference<AsyncValue>, 4> inputs;
  inputs.push_back(tfrt::MakeAvailableAsyncValueRef<std::string>(
      tmp_folder.getAbsolutePath()));
  inputs.push_back(tfrt::MakeAvailableAsyncValueRef<int32_t>(1));
  inputs.push_back(tfrt::MakeAvailableAsyncValueRef<int32_t>(0));

  auto results =
      test::runMlirScript(host_context, mlir_context, mlir_script, inputs);
  EXPECT_EQ(results.size(), 1);
  host_context->Await(results);
  SparseVector &actual_vector = results[0]->get<SparseVector>();
  SparseVector expected_vector(2);
  expected_vector.set(1, 1.0);
  EXPECT_EQ(actual_vector, expected_vector);

  llvm::SmallVector<RCReference<AsyncValue>, 4> invalid_inputs;
  invalid_inputs.push_back(tfrt::MakeAvailableAsyncValueRef<std::string>(
      tmp_folder.getAbsolutePath()));
  invalid_inputs.push_back(tfrt::MakeAvailableAsyncValueRef<int32_t>(5));
  invalid_inputs.push_back(tfrt::MakeAvailableAsyncValueRef<int32_t>(5));

  auto invalid_results = test::runMlirScript(host_context, mlir_context,
                                             mlir_script, invalid_inputs);
  EXPECT_EQ(invalid_results.size(), 1);
  host_context->Await(invalid_results);
  EXPECT_TRUE(invalid_results[0]->IsError());
}

}  // namespace
}  // namespace clink
