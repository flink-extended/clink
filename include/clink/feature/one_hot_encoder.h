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

#include "clink/api/model.h"
#include "clink/feature/proto/one_hot_encoder.pb.h"
#include "clink/linalg/sparse_vector.h"

namespace clink {

// A Model which encodes data into one-hot format.
class OneHotEncoderModel : public Model {
 public:
  // Default constructor.
  OneHotEncoderModel(tfrt::HostContext *host) : allocator_(host->allocator()) {}

  // Move operations are supported.
  OneHotEncoderModel(OneHotEncoderModel &&other) = default;
  OneHotEncoderModel &operator=(OneHotEncoderModel &&other) = default;

  // This class is not copyable or assignable.
  OneHotEncoderModel(const OneHotEncoderModel &other) = delete;
  OneHotEncoderModel &operator=(const OneHotEncoderModel &) = delete;

  llvm::SmallVector<tfrt::RCReference<tfrt::AsyncValue>, 4> transform(
      llvm::ArrayRef<tfrt::AsyncValue *> inputs,
      const tfrt::ExecutionContext &exec_ctx) override;

  // Loads a OneHotEncoderModel from given path. The path should be a directory
  // containing params and model data saved through
  // org.clink.feature.onehotencoder.ClinkOneHotEnoderModel::save(...).
  static llvm::Expected<tfrt::RCReference<OneHotEncoderModel>> load(
      const std::string &path, tfrt::HostContext *host);

  void setDropLast(const bool is_droplast);

  bool getDropLast() const;

 private:
  void Destroy() override {
    Model::DestroyImpl<OneHotEncoderModel>(this, allocator_);
  }

  // Params of OneHotEncoderModel.
  struct Params {
    // Whether to drop the last category.
    bool is_droplast;
  };

  // Params of OneHotEncoderModel.
  Params params_;

  // Model data of OneHotEncoderModel.
  OneHotEncoderModelDataProto model_data_;

  tfrt::HostAllocator *allocator_;
};

}  // namespace clink

#endif  // CLINK_FEATURE_ONE_HOT_ENCODER_H_
