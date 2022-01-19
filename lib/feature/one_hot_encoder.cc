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

#include <fstream>

#include "clink/utils/clink_utils.h"
#include "nlohmann/json.hpp"

namespace clink {

tfrt::AsyncValueRef<SparseVector>
OneHotEncoderModel::transform(const int value, const int column_index) const {
  if (column_index >= model_data_.featuresizes_size()) {
    return tfrt::MakeErrorAsyncValueRef("Column index out of range.");
  }

  int len = model_data_.featuresizes(column_index);
  if (value >= len) {
    return tfrt::MakeErrorAsyncValueRef("Value out of range.");
  }
  if (getDropLast()) {
    len -= 1;
  }

  tfrt::AsyncValueRef<SparseVector> vector =
      tfrt::MakeAvailableAsyncValueRef<SparseVector>(len);
  if (value < len) {
    vector->set(value, 1.0);
  }
  return vector;
}

void OneHotEncoderModel::setDropLast(const bool is_droplast) {
  params_.is_droplast = is_droplast;
}

bool OneHotEncoderModel::getDropLast() const { return params_.is_droplast; }

llvm::Error
OneHotEncoderModel::setModelData(const std::string &model_data_str) {
  OneHotEncoderModelDataProto model_data;

  if (!model_data.ParseFromString(model_data_str)) {
    return tfrt::MakeStringError("Failed to parse modeldata.");
  }

  for (int i = 0; i < model_data.featuresizes_size(); i++) {
    if (model_data.featuresizes(i) <= 0) {
      return tfrt::MakeStringError(
          "Model data feature size value must be positive.");
    }
  }

  model_data_ = std::move(model_data);

  return llvm::Error::success();
}

llvm::Expected<tfrt::RCReference<OneHotEncoderModel>>
OneHotEncoderModel::load(const std::string &path, tfrt::HostContext *host) {
  tfrt::RCReference<OneHotEncoderModel> model =
      TakeRef(host->Construct<OneHotEncoderModel>(host));

  std::ifstream params_input(path + "/metadata");
  nlohmann::json params;
  params << params_input;
  std::string is_droplast = params["paramMap"]["dropLast"].get<std::string>();
  model->setDropLast(is_droplast != "false");
  params_input.close();

  std::string model_data_filename = getOnlyFileInDirectory(path + "/data");
  if (model_data_filename == "") {
    return tfrt::MakeStringError(
        "Failed to load OneHotEncoderModel: model data directory " + path +
        "/data does not exist, or it has zero or more than one file.");
  }

  std::ifstream model_data_input(path + "/data/" + model_data_filename);
  std::string model_data_str((std::istreambuf_iterator<char>(model_data_input)),
                             std::istreambuf_iterator<char>());
  llvm::Error err = model->setModelData(std::move(model_data_str));
  model_data_input.close();

  if (err) {
    return tfrt::MakeStringError(
        "Failed to load OneHotEncoderModel: invalid model data file " + path +
        "/data/" + model_data_filename);
  }

  return model;
}

} // namespace clink
