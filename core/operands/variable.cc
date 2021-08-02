/* Copyright (c) 2021, Qihoo, Inc.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/

#include "core/operands/variable.h"

#include <utility>

namespace clink {

Variable::Variable(std::string token) {
  hash_token_ = MAKE_HASH(token);
  token_ = std::move(token);
}

Variable::Variable() {}

const Feature* Variable::Evaluate(Context* context) {
  return context->Get(hash_token_);
}

const std::string& Variable::GetKey() { return token_; }

const std::string* Variable::GetOperationName() { return &token_; }

}  // namespace clink
