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

#ifndef CLINK_LINALG_SPARSE_VECTOR_H_
#define CLINK_LINALG_SPARSE_VECTOR_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

namespace clink {

// A sparse vector of double values.
class SparseVector {
public:
  // Constructor for SparseVector.
  // `n` stands for the number of dimensions of the vector.
  SparseVector(const int n) : n_(n) {}

  // Move operations are supported.
  SparseVector(SparseVector &&other) = default;
  SparseVector &operator=(SparseVector &&other) = default;

  // This class is not copyable or assignable.
  SparseVector(const SparseVector &other) = delete;
  SparseVector &operator=(const SparseVector &) = delete;

  // Sets the value at a certain index of the vector.
  llvm::Error set(const int index, const double value);

  // Gets the value at a certain index of the vector.
  llvm::Expected<double> get(const int index);

  // Gets the total number of dimensions of the vector.
  int size();

private:
  const int n_;
  llvm::SmallVector<int, 4> indices_;
  llvm::SmallVector<double, 4> values_;
};

} // namespace clink

#endif // CLINK_LINALG_SPARSE_VECTOR_H_
