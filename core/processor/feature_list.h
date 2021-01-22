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

#ifndef CORE_PROCESSOR_FEATURE_LIST_H_
#define CORE_PROCESSOR_FEATURE_LIST_H_
#include <iostream>
#include <memory>
#include <string>
#include <vector>
namespace perception_feature {
namespace proto {
class Record;
}
using Feature = proto::Record;

class FeatureList {
 public:
  FeatureList();
  virtual ~FeatureList();
  void AddValue(const double& value);
  void AddValue(const float& value);
  void AddValue(const int& value);
  void AddValue(const std::string& value);
  void AddValue(const std::vector<float>& value);
  void AddValue(const std::vector<double>& value);
  void AddEmptyValue();

  inline bool Empty() const { return feature_list.empty(); }
  inline int Size() const { return feature_list.size(); }
  inline const std::shared_ptr<Feature>& GetValue(const int& index) const {
    return feature_list.at(index);
  }

  int GetValue(const int& index, std::shared_ptr<Feature>& feature);

 private:
  std::vector<std::shared_ptr<Feature>> feature_list;
};
}  // namespace perception_feature

#endif  // CORE_PROCESSOR_FEATURE_LIST_H_
