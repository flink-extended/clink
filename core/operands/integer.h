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

#ifndef CORE_OPERANDS_INTEGER_H_
#define CORE_OPERANDS_INTEGER_H_
#include <memory>
#include <sstream>

#include "core/common/common.h"
#include "core/operands/operand.h"
namespace perception_feature {
class Integer : public Operand {
 public:
  explicit Integer(int64_t = 0);
  int Evaluate(const FeatureMap&, std::shared_ptr<Feature>&);

 protected:
  Integer(const Integer&);

 private:
  int64_t value_;
};
}  // namespace perception_feature

#endif  // CORE_OPERANDS_INTEGER_H_
