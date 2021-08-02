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

#ifndef COMMON_CLINK_SAMPLE_LIST_H_
#define COMMON_CLINK_SAMPLE_LIST_H_
#include <google/protobuf/arena.h>

#include <memory>
#include <string>
#include <vector>

#include "core/utils/util.h"

namespace clink {
namespace proto {
class SampleRecord;
class SampleMap;
class Record;
}  // namespace proto

using SampleRecord = proto::SampleRecord;
using SampleMap = proto::SampleMap;
using Feature = proto::Record;
class Sample {
 public:
  Sample(google::protobuf::Arena& arena);

  Sample(const int& feature_count, google::protobuf::Arena& arena);
  virtual ~Sample();

  void AddFeature(const double& value);

  void AddFeature(const float& value);

  void AddFeature(const int& value);

  void AddFeature(const int64_t& value);

  void AddFeature(const std::string& value);

  void AddFeature(const std::vector<float>& value);

  void AddFeature(const std::vector<double>& value);

  void AddEmptyValue();

  bool Empty() const;

  int Size() const;

  void ToString(std::string* output);

  const Feature* Get(const int32_t& feature_index) const;

  SampleRecord* GetSampleRecord();

 private:
  SampleRecord* sample_record_;

  int feature_index_;
};

class SampleList {
 public:
  SampleList();

  virtual ~SampleList();

  void AddSample(std::unique_ptr<Sample>& sample);

  void ClearSample();

  int Size();

  SampleMap* GetSampleRecord();

  std::unique_ptr<Sample> CreateSample();

  std::unique_ptr<Sample> CreateSample(const int& feature_count);

 private:
  // std::unique_ptr<SampleMap> sample_record_map_;
  SampleMap* sample_record_map_;
  google::protobuf::Arena arena_;
  int sample_index_;
};

}  // namespace clink

#endif  // COMMON_CLINK_SAMPLE_LIST_H_
