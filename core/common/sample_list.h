// Copyright (c) 2020 Qihoo Inc. All rights reserved.
// File  sample_list.h
// Author 360
// Date   2020/12/2 下午1:51
// Brief

#ifndef PERCEPTION_FEATURE_PLUGIN_SRC_COMMON_SAMPLE_LIST_H_
#define PERCEPTION_FEATURE_PLUGIN_SRC_COMMON_SAMPLE_LIST_H_
#include <iostream>
#include <memory>
#include <vector>
namespace perception_feature {
namespace proto {
class Record;
class SampleRecord;
class SampleRecordList;
}  // namespace proto
using Feature = proto::Record;
using SampleRecord = proto::SampleRecord;
using SampleRecordList = proto::SampleRecordList;
class Sample {
 public:
  Sample();
  virtual ~Sample();
  void AddValue(const double& value);
  void AddValue(const float& value);
  void AddValue(const int& value);
  void AddValue(const std::string& value);
  void AddValue(const std::vector<float>& value);
  void AddValue(const std::vector<double>& value);
  void AddEmptyValue();
  bool Empty() const;
  int Size() const;
  const Feature& GetValue(const int& index) const;
  std::shared_ptr<SampleRecord>& GetSampleRecord();

 private:
  std::shared_ptr<SampleRecord> sample_record_;
};
class SampleList {
 public:
  SampleList();
  virtual ~SampleList();
  void AddSample(Sample& sample);
  void ClearSample();
  std::shared_ptr<SampleRecordList>& GetSampleRecord();

 private:
  std::shared_ptr<SampleRecordList> sample_record_list_;
};

}  // namespace perception_feature

#endif  // PERCEPTION_FEATURE_PLUGIN_SRC_COMMON_FEATURE_LIST_H_
