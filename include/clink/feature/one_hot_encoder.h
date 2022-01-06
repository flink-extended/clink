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

#ifndef CLINK_FEATURE_ONE_HOT_ENCODER_H_
#define CLINK_FEATURE_ONE_HOT_ENCODER_H_

#include "clink/feature/proto/one_hot_encoder.pb.h"
#include "clink/linalg/sparse_vector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

namespace clink {

// A Model which encodes data into one-hot format.
class OneHotEncoderModel {
public:
  // Default constructor.
  OneHotEncoderModel() {}

  // Move operations are supported.
  OneHotEncoderModel(OneHotEncoderModel &&other) = default;
  OneHotEncoderModel &operator=(OneHotEncoderModel &&other) = default;

  // This class is not copyable or assignable.
  OneHotEncoderModel(const OneHotEncoderModel &other) = delete;
  OneHotEncoderModel &operator=(const OneHotEncoderModel &) = delete;

  // Converts the provided data to a SparseVector.
  llvm::Expected<SparseVector> transform(const int value,
                                         const int columnIndex);

  // Loads a OneHotEncoderModel from given path. The path should be a directory
  // containing params and model data saved through
  // org.clink.feature.onehotencoder.ClinkOneHotEnoderModel::save(...).
  static llvm::Expected<OneHotEncoderModel> load(const std::string path);

  void setDropLast(const bool is_droplast);

  bool getDropLast();

  llvm::Error setModelData(const std::string modelDataProtobuf);

private:
  // Params of OneHotEncoderModel.
  struct Params {
    // Whether to drop the last category.
    bool is_droplast;
  };

  // Params of OneHotEncoderModel.
  Params params_;

  // Model data of OneHotEncoderModel.
  OneHotEncoderModelDataProto model_data_;
};

} // namespace clink

#endif // CLINK_FEATURE_ONE_HOT_ENCODER_H_
