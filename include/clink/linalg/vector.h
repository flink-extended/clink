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

#ifndef CLINK_LINALG_VECTOR_H_
#define CLINK_LINALG_VECTOR_H_

#include "tfrt/host_context/chain.h"

namespace clink {

// A vector of double values.
class Vector {
 public:
  // Sets the value at a certain index of the vector.
  virtual llvm::Error set(const int index, const double value) = 0;

  // Gets the value at a certain index of the vector.
  virtual llvm::Expected<double> get(const int index) const = 0;

  // Gets the total number of dimensions of the vector.
  virtual int size() const = 0;

 protected:
  // Move operations are supported.
  Vector(Vector &&other) = default;
  Vector &operator=(Vector &&other) = default;

  // This class is not copyable or assignable.
  Vector(const Vector &other) = delete;
  Vector &operator=(const Vector &) = delete;

  Vector() = default;

  virtual ~Vector() {}
};

}  // namespace clink

#endif  // CLINK_LINALG_VECTOR_H_
