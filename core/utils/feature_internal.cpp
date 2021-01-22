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

#include "core/utils/feature_internal.h"
namespace perception_feature {

template <>
const google::protobuf::RepeatedField<int64_t>& GetFeatureValues<int64_t>(
    const Feature& feature) {
  return feature.int64_list().value();
}

template <>
google::protobuf::RepeatedField<int64_t>* GetFeatureValues<int64_t>(
    Feature* feature) {
  return feature->mutable_int64_list()->mutable_value();
}
template <>
const google::protobuf::RepeatedField<int32_t>& GetFeatureValues<int32_t>(
    const Feature& feature) {
  return feature.int_list().value();
}

template <>
google::protobuf::RepeatedField<int32_t>* GetFeatureValues<int32_t>(
    Feature* feature) {
  return feature->mutable_int_list()->mutable_value();
}

template <>
const google::protobuf::RepeatedField<float>& GetFeatureValues<float>(
    const Feature& feature) {
  return feature.float_list().value();
}

template <>
google::protobuf::RepeatedField<float>* GetFeatureValues<float>(
    Feature* feature) {
  return feature->mutable_float_list()->mutable_value();
}

template <>
const google::protobuf::RepeatedField<double>& GetFeatureValues<double>(
    const Feature& feature) {
  return feature.double_list().value();
}

template <>
google::protobuf::RepeatedField<double>* GetFeatureValues<double>(
    Feature* feature) {
  return feature->mutable_double_list()->mutable_value();
}

// template <>
// const google::protobuf::RepeatedField<IVRecordEntry>&
// GetFeatureValues<IVRecordEntry>(
//    const Feature& feature) {
//  return feature.iv_list().iv_record();
//}

// template <>
// google::protobuf::RepeatedField<float>* GetFeatureValues<float>(Feature*
// feature) {
//  return feature->mutable_float_list()->mutable_value();
//}

template <>
const google::protobuf::RepeatedPtrField<std::string>&
GetFeatureValues<std::string>(const Feature& feature) {
  return feature.bytes_list().value();
}
template <>
google::protobuf::RepeatedPtrField<std::string>* GetFeatureValues<std::string>(
    Feature* feature) {
  return feature->mutable_bytes_list()->mutable_value();
}

template <>
void ClearFeatureValues<int64_t>(Feature* feature) {
  feature->mutable_int64_list()->Clear();
}

template <>
void ClearFeatureValues<float>(Feature* feature) {
  feature->mutable_float_list()->Clear();
}

template <>
void ClearFeatureValues<std::string>(Feature* feature) {
  feature->mutable_bytes_list()->Clear();
}

template <>
const google::protobuf::RepeatedField<int64_t>& GetFeatureValues<int64_t>(
    const Feature& feature);

template <>
google::protobuf::RepeatedField<int64_t>* GetFeatureValues<int64_t>(
    Feature* feature);

template <>
const google::protobuf::RepeatedField<float>& GetFeatureValues<float>(
    const Feature& feature);

template <>
google::protobuf::RepeatedField<float>* GetFeatureValues<float>(
    Feature* feature);

template <>
const google::protobuf::RepeatedPtrField<std::string>&
GetFeatureValues<std::string>(const Feature& feature);

template <>
google::protobuf::RepeatedPtrField<std::string>* GetFeatureValues<std::string>(
    Feature* feature);

}  // namespace perception_feature
