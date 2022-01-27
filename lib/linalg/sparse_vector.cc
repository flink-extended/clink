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

#include "clink/linalg/sparse_vector.h"

namespace clink {

llvm::Expected<double> SparseVector::get(const int index) const {
  if (index >= n_ || index < 0) {
    return tfrt::MakeStringError("Index out of range.");
  }

  for (int i = 0; i < indices_.size(); i++) {
    if (indices_[i] == index) {
      return values_[i];
    }
  }
  return 0.0;
}

llvm::Error SparseVector::set(const int index, const double value) {
  if (index >= n_) {
    return tfrt::MakeStringError("Index out of range.");
  }

  for (int i = 0; i < indices_.size(); i++) {
    if (indices_[i] == index) {
      values_[i] = value;
      return llvm::Error::success();
    }
  }

  indices_.push_back(index);
  values_.push_back(value);
  return llvm::Error::success();
}

int SparseVector::size() const { return n_; }

bool SparseVector::operator==(const SparseVector &other) const {
  if (n_ != other.n_) {
    return false;
  }
  for (int i = 0; i < n_; i++) {
    if (this->get(i).get() != other.get(i).get()) {
      return false;
    }
  }
  return true;
}

}  // namespace clink
