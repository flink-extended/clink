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

#ifndef CLINK_FEATURE_OPDEFS_TYPES_H_
#define CLINK_FEATURE_OPDEFS_TYPES_H_

#include "mlir/IR/Types.h"

namespace clink {

class ModelType
    : public mlir::Type::TypeBase<ModelType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

class VectorType
    : public mlir::Type::TypeBase<VectorType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

} // namespace clink

#endif // CLINK_FEATURE_OPDEFS_TYPES_H_
