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
#include "sys/stat.h"
#include <dirent.h>
#include <string>

#include "clink/feature/one_hot_encoder.h"
#include "nlohmann/json.hpp"
#include "tfrt/support/error_util.h"

using namespace clink;
using namespace std;
using namespace nlohmann;

llvm::Expected<SparseVector>
OneHotEncoderModel::transform(const int value, const int columnIndex) {
  if (columnIndex >= model_data_.featuresizes_size()) {
    return tfrt::MakeStringError("Column index out of range.");
  }

  int len = model_data_.featuresizes(columnIndex);
  if (value >= len) {
    return tfrt::MakeStringError("Value out of range.");
  }
  if (getDropLast()) {
    len -= 1;
  }

  SparseVector vector(len);
  if (value < len) {
    vector.set(value, 1.0);
  }
  return vector;
}

void OneHotEncoderModel::setDropLast(const bool is_droplast) {
  params_.is_droplast = is_droplast;
}

bool OneHotEncoderModel::getDropLast() { return params_.is_droplast; }

llvm::Error OneHotEncoderModel::setModelData(const std::string model_data_str) {
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

namespace {
/*
 * Given a directory path, gets the single file in the directory.
 *
 * Flink ML saves model data in a file whose name is unknown to C++. This
 * function helps to locate that file.
 *
 * This function returns empty string if the directory does not exist, or there
 * is zero or more than one file in the directory.
 */
std::string getOnlyFileInDirectory(std::string path) {
  std::string result = "";
  struct dirent *entry;
  struct stat st;
  DIR *dir = opendir(path.c_str());

  if (dir == NULL) {
    return "";
  }
  while ((entry = readdir(dir)) != NULL) {
    const string full_file_name = path + "/" + entry->d_name;
    if (stat(full_file_name.c_str(), &st) == -1)
      continue;
    bool is_directory = (st.st_mode & S_IFDIR) != 0;
    if (!is_directory) {
      if (result.compare("")) {
        return "";
      }
      result = std::string(entry->d_name);
    }
  }
  closedir(dir);
  return result;
}
} // anonymous namespace

llvm::Expected<OneHotEncoderModel>
OneHotEncoderModel::load(const std::string path) {
  OneHotEncoderModel model;

  ifstream params_input(path + "/metadata");
  json params;
  params << params_input;
  std::string is_droplast = params["paramMap"]["dropLast"].get<std::string>();
  model.setDropLast(is_droplast.compare("false"));
  params_input.close();

  std::string model_data_filename = getOnlyFileInDirectory(path + "/data");
  if (!model_data_filename.compare("")) {
    return tfrt::MakeStringError(
        "Failed to load OneHotEncoderModel: model data directory " + path +
        "/data does not exist, or it has zero or more than one file.");
  }

  ifstream model_data_input(path + "/data/" + model_data_filename);
  std::string model_data_str((std::istreambuf_iterator<char>(model_data_input)),
                             std::istreambuf_iterator<char>());
  llvm::Error err = model.setModelData(model_data_str);
  model_data_input.close();

  if (err) {
    return tfrt::MakeStringError(
        "Failed to load OneHotEncoderModel: invalid model data file " + path +
        "/data/" + model_data_filename);
  }

  return model;
}
