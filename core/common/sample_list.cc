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

#include "core/common/sample_list.h"

#include <butil/logging.h>
#include <google/protobuf/arena.h>

#include <string>

#include "core/common/common.h"
#include "core/utils/feature_internal.h"

namespace clink {

Sample::Sample(google::protobuf::Arena& arena) : feature_index_(-1) {
  sample_record_ = google::protobuf::Arena::CreateMessage<SampleRecord>(&arena);
}

Sample::Sample(const int& feature_count, google::protobuf::Arena& arena)
    : feature_index_(-1) {
  sample_record_ = google::protobuf::Arena::CreateMessage<SampleRecord>(&arena);
}

Sample::~Sample() { sample_record_->clear_features(); }

void Sample::AddFeature(const double& value) {
  GetFeatureValues<double>(
      &(*sample_record_->mutable_features())[++feature_index_])
      ->Add(value);
}

void Sample::AddFeature(const float& value) {
  GetFeatureValues<float>(
      &(*sample_record_->mutable_features())[++feature_index_])
      ->Add(value);
}

void Sample::AddFeature(const int& value) {
  GetFeatureValues<int>(
      &(*sample_record_->mutable_features())[++feature_index_])
      ->Add(value);
}

void Sample::AddFeature(const int64_t& value) {
  GetFeatureValues<int64_t>(
      &(*sample_record_->mutable_features())[++feature_index_])
      ->Add(value);
}

void Sample::AddFeature(const std::string& value) {
  *GetFeatureValues<std::string>(
       &(*sample_record_->mutable_features())[++feature_index_])
       ->Add() = value;

  // *GetFeatureValues<std::string>(sample_record_->mutable_features()->Add())
  //      ->Add() = value;
}

void Sample::AddFeature(const std::vector<float>& value) {
  AppendFeatureValues(value,
                      &(*sample_record_->mutable_features())[++feature_index_]);
}

void Sample::AddFeature(const std::vector<double>& value) {
  AppendFeatureValues(value,
                      &(*sample_record_->mutable_features())[++feature_index_]);
}
void Sample::AddEmptyValue() {
  *GetFeatureValues<std::string>(
       &(*sample_record_->mutable_features())[++feature_index_])
       ->Add() = "";
}

const Feature* Sample::Get(const int32_t& feature_index) const {
  if (sample_record_->features().contains(feature_index_)) {
    return &sample_record_->features().at(feature_index);
  } else {
    return nullptr;
  }
}

int Sample::Size() const { return sample_record_->features_size(); }

SampleRecord* Sample::GetSampleRecord() { return sample_record_; }

bool Sample::Empty() const { return sample_record_->features_size() == 0; }

void Sample::ToString(std::string* output) {
  // for (int i = 0; i < sample_record_->features_size(); i++) {
  //   const Feature& feature = sample_record_->features(i);
  //   switch (feature.kind_case()) {
  //     case Feature::kBytesList: {
  //       // for (int j=0;j<feature.) {
  //       //   std::cout << it << std::endl;
  //       // }
  //       auto& s = GetFeatureValues<std::string>(feature);
  //       for (auto& s1 : s) {
  //         output->append(s1 + "|");
  //       }
  //       output->append("\n");
  //       break;
  //     }
  //     case Feature::kBoolList: {
  //       // auto& s = GetFeatureValues<bool>(feature);
  //       // for (auto s1 : s) {
  //       //   std::cout << s1 << " ";
  //       // }
  //       // std::cout << std::endl;
  //       break;
  //     }
  //     case Feature::kIntList: {
  //       auto& s = GetFeatureValues<int>(feature);
  //       for (auto& s1 : s) {
  //         output->append(std::to_string(s1) + "|");
  //       }
  //       output->append("\n");
  //       break;
  //     }
  //     case Feature::kInt64List: {
  //       auto& s = GetFeatureValues<int64_t>(feature);
  //       for (auto& s1 : s) {
  //         output->append(std::to_string(s1) + "|");
  //       }
  //       output->append("\n");
  //       break;
  //     }
  //     case Feature::kFloatList: {
  //       auto& s = GetFeatureValues<float>(feature);
  //       for (auto& s1 : s) {
  //         output->append(std::to_string(s1) + "|");
  //       }
  //       output->append("\n");
  //       break;
  //     }
  //     case Feature::kDoubleList: {
  //       auto& s = GetFeatureValues<double>(feature);
  //       for (auto& s1 : s) {
  //         output->append(std::to_string(s1) + "|");
  //       }
  //       output->append("\n");
  //       break;
  //     }
  //     case Feature::kIvList: {
  //       break;
  //     }
  //     case Feature::KIND_NOT_SET: {
  //       break;
  //     }
  //   }
  // }
}

std::unique_ptr<Sample> SampleList::CreateSample() {
  return std::make_unique<Sample>(arena_);
}

std::unique_ptr<Sample> SampleList::CreateSample(const int& feature_count) {
  return std::make_unique<Sample>(feature_count, arena_);
}

void SampleList::AddSample(std::unique_ptr<Sample>& sample) {
  (*sample_record_map_->mutable_sample_map())[sample_index_].Swap(
      sample->GetSampleRecord());
  sample_index_++;
}

SampleList::SampleList() : sample_index_(0) {
  // sample_record_map_ = std::make_unique<SampleMap>();
  sample_record_map_ =
      google::protobuf::Arena::CreateMessage<SampleMap>(&arena_);
}

SampleList::~SampleList() { arena_.Reset(); }

SampleMap* SampleList::GetSampleRecord() { return sample_record_map_; }

void SampleList::ClearSample() {
  sample_record_map_->Clear();
  arena_.Reset();
}

}  // namespace clink
