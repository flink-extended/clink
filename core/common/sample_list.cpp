// Copyright (c) 2020 Qihoo Inc. All rights reserved.
// File  sample_list.cpp
// Author 360
// Date   2020/12/2 下午1:51
// Brief

#include "processor/sample_list.h"

#include "common/common.h"
#include "utils/feature_internal.h"
namespace perception_feature {
Sample::Sample() { sample_record_ = std::make_shared<SampleRecord>(); }
Sample::~Sample() { sample_record_->clear_feature_list(); }
void Sample::AddValue(const double& value) {
  GetFeatureValues<double>(sample_record_->mutable_feature_list()->Add())
      ->Add(value);
}
void Sample::AddValue(const float& value) {
  GetFeatureValues<float>(sample_record_->mutable_feature_list()->Add())
      ->Add(value);
}
void Sample::AddValue(const int& value) {
  GetFeatureValues<int>(sample_record_->mutable_feature_list()->Add())
      ->Add(value);
}
void Sample::AddValue(const std::string& value) {
  *GetFeatureValues<std::string>(sample_record_->mutable_feature_list()->Add())
       ->Add() = value;
}
void Sample::AddValue(const std::vector<float>& value) {
  AppendFeatureValues(value, sample_record_->mutable_feature_list()->Add());
}

void Sample::AddValue(const std::vector<double>& value) {
  AppendFeatureValues(value, sample_record_->mutable_feature_list()->Add());
}
void Sample::AddEmptyValue() {
  *GetFeatureValues<std::string>(sample_record_->mutable_feature_list()->Add())
       ->Add() = "";
}
const Feature& Sample::GetValue(const int& index) const {
  return sample_record_->feature_list().Get(index);
}
int Sample::Size() const { return sample_record_->feature_list_size(); }
bool Sample::Empty() const { return sample_record_->feature_list_size() <= 0; }
std::shared_ptr<SampleRecord>& Sample::GetSampleRecord() {
  return sample_record_;
}
void SampleList::AddSample(Sample& feature_list) {
  sample_record_list_->mutable_sample_list()->Add()->Swap(
      feature_list.GetSampleRecord().get());
}
SampleList::SampleList() {
  sample_record_list_ = std::make_shared<SampleRecordList>();
}
SampleList::~SampleList() { sample_record_list_->clear_sample_list(); }
std::shared_ptr<SampleRecordList>& SampleList::GetSampleRecord() {
  return sample_record_list_;
}
void SampleList::ClearSample() { sample_record_list_->clear_sample_list(); }
}  // namespace perception_feature
