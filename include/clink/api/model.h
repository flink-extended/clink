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

#ifndef CLINK_API_MODEL_H_
#define CLINK_API_MODEL_H_

#include "tfrt/host_context/host_allocator.h"
#include "tfrt/support/ref_count.h"

namespace clink {

// Basic interface for Clink operators that provides feature processing
// function.
class Model : public tfrt::ReferenceCounted<Model> {
 public:
  virtual ~Model() {}

  template <typename SubClass>
  void DestroyImpl(SubClass *ptr, tfrt::HostAllocator *allocator) {
    ptr->~SubClass();
    allocator->DeallocateBytes(ptr, sizeof(SubClass));
  }

 private:
  // For access to Destroy().
  friend class ReferenceCounted<Model>;

  virtual void Destroy() = 0;
};

}  // namespace clink

#endif  // CLINK_API_MODEL_H_
