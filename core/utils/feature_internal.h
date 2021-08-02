/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef COMMON_CLINK_FEATURE_INTERNAL_H_
#define COMMON_CLINK_FEATURE_INTERNAL_H_
#include <algorithm>
#include <string>

#include "core/common/common.h"

namespace clink {

namespace internal {

template <typename FeatureType>
struct RepeatedFieldTrait;
template <typename FeatureType>
const typename internal::RepeatedFieldTrait<FeatureType>::Type&
GetFeatureValues(const Feature& feature);

template <>
struct RepeatedFieldTrait<int64_t> {
  using Type = google::protobuf::RepeatedField<int64_t>;
};

template <>
struct RepeatedFieldTrait<int32_t> {
  using Type = google::protobuf::RepeatedField<int32_t>;
};

template <>
struct RepeatedFieldTrait<float> {
  using Type = google::protobuf::RepeatedField<float>;
};

template <>
struct RepeatedFieldTrait<double> {
  using Type = google::protobuf::RepeatedField<double>;
};

template <>
struct RepeatedFieldTrait<std::string> {
  using Type = google::protobuf::RepeatedPtrField<std::string>;
};

template <>
struct RepeatedFieldTrait<IVRecordEntry> {
  using Type = google::protobuf::RepeatedPtrField<IVRecordEntry>;
};

// Specializations of FeatureTrait define a type of feature corresponding to a
// selected value type.
template <typename ValueType, class Enable = void>
struct FeatureTrait;
template <typename ValueType>
struct FeatureTrait<ValueType, typename std::enable_if<
                                   std::is_integral<ValueType>::value>::type> {
  using Type = int;
};

template <typename ValueType>
struct FeatureTrait<
    ValueType,
    typename std::enable_if<std::is_floating_point<ValueType>::value>::type> {
  using Type = float;
};

template <typename T>
struct is_string
    : public std::integral_constant<
          bool,
          std::is_same<char*, typename std::decay<T>::type>::value ||
              std::is_same<const char*, typename std::decay<T>::type>::value> {
};

template <>
struct is_string<std::string> : std::true_type {};
template <typename ValueType>
struct FeatureTrait<
    ValueType, typename std::enable_if<is_string<ValueType>::value>::type> {
  using Type = std::string;
};

template <typename T>
struct is_iv_record
    : public std::integral_constant<
          bool,
          std::is_same<char*, typename std::decay<T>::type>::value ||
              std::is_same<const char*, typename std::decay<T>::type>::value> {
};

template <>
struct is_iv_record<IVRecordEntry> : std::true_type {};
}  //  namespace internal

template <typename T>
struct TypeHasFeatures : std::false_type {};

template <>
struct TypeHasFeatures<SampleRecord*> : std::true_type {};

// A family of template functions to return mutable Features proto from a
// container proto. Supported ProtoTypes: SampleRecord.
template <typename ProtoType>
typename std::enable_if<TypeHasFeatures<ProtoType>::value, SampleRecord*>::type
GetSampleRecord(ProtoType* proto);

template <typename ProtoType>
typename std::enable_if<TypeHasFeatures<ProtoType>::value,
                        const SampleRecord&>::type
GetSampleRecord(const ProtoType& proto);

template <typename FeatureType>
const typename internal::RepeatedFieldTrait<FeatureType>::Type&
GetFeatureValues(const Feature& feature);

// Returns a mutable repeated field of a feature values.
template <typename FeatureType>
typename internal::RepeatedFieldTrait<FeatureType>::Type* GetFeatureValues(
    Feature* feature);

// Returns a read only repeated field corresponding to a feature with the
// specified name and FeatureType. Supported ProtoTypes:  SampleRecord.
template <typename FeatureType, typename ProtoType>
const typename internal::RepeatedFieldTrait<FeatureType>::Type&
GetFeatureValues(const int& key, const ProtoType& proto) {
  return GetFeatureValues<FeatureType>(
      GetSampleRecord(proto).features().at(key));
}

// Returns a read-only Feature proto for the specified key, throws
// std::out_of_range if the key is not found. Supported types for the proto:
// Example, Features.
template <typename ProtoType>
const Feature& GetFeature(const int& key, const ProtoType& proto) {
  return GetSampleRecord(proto).features().at(key);
}

// Returns a mutable Feature proto for the specified key, creates a new if
// necessary. Supported types for the proto: Example, Features.
template <typename ProtoType>
Feature* GetFeature(const int& key, ProtoType* proto) {
  return &(*GetSampleRecord(proto)->mutable_features())[key];
}

template <typename IteratorType>
void AppendFeatureValues(IteratorType first, IteratorType last,
                         Feature* feature) {
  using FeatureType = typename internal::FeatureTrait<
      typename std::iterator_traits<IteratorType>::value_type>::Type;
  std::copy(first, last,
            google::protobuf::RepeatedFieldBackInserter(
                GetFeatureValues<FeatureType>(feature)));
}

template <typename ValueType>
void AppendFeatureValues(std::initializer_list<ValueType> container,
                         Feature* feature) {
  AppendFeatureValues(container.begin(), container.end(), feature);
}

template <typename ContainerType>
void AppendFeatureValues(const ContainerType& container, Feature* feature) {
  using IteratorType = typename ContainerType::const_iterator;
  AppendFeatureValues<IteratorType>(container.begin(), container.end(),
                                    feature);
}

// Copies elements from the range, defined by [first, last) into the feature
// obtainable from the (proto, key) combination.
template <typename IteratorType, typename ProtoType>
void AppendFeatureValues(IteratorType first, IteratorType last, const int& key,
                         ProtoType* proto) {
  AppendFeatureValues(first, last, GetFeature(key, GetSampleRecord(proto)));
}

// Copies all elements from the container into a feature.
template <typename ContainerType, typename ProtoType>
void AppendFeatureValues(const ContainerType& container, const int& key,
                         ProtoType* proto) {
  using IteratorType = typename ContainerType::const_iterator;
  AppendFeatureValues<IteratorType>(container.begin(), container.end(), key,
                                    proto);
}
// Copies all elements from the initializer list into a Feature contained by
// Features or Example proto.
template <typename ValueType, typename ProtoType>
void AppendFeatureValues(std::initializer_list<ValueType> container,
                         const int& key, ProtoType* proto) {
  using IteratorType =
      typename std::initializer_list<ValueType>::const_iterator;
  AppendFeatureValues<IteratorType>(container.begin(), container.end(), key,
                                    proto);
}

// Clears the feature's repeated field (int64, float, or string).
template <typename... FeatureType>
void ClearFeatureValues(Feature* feature);
// Clears the feature's repeated field (int64, float, or string). Copies
// elements from the range, defined by [first, last) into the feature's repeated
// field.
template <typename IteratorType>
void SetFeatureValues(IteratorType first, IteratorType last, Feature* feature) {
  using FeatureType = typename internal::FeatureTrait<
      typename std::iterator_traits<IteratorType>::value_type>::Type;
  ClearFeatureValues<FeatureType>(feature);
  AppendFeatureValues(first, last, feature);
}
// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the initializer list into the feature's repeated field.
template <typename ValueType>
void SetFeatureValues(std::initializer_list<ValueType> container,
                      Feature* feature) {
  SetFeatureValues(container.begin(), container.end(), feature);
}
// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the container into the feature's repeated field.
template <typename ContainerType>
void SetFeatureValues(const ContainerType& container, Feature* feature) {
  using IteratorType = typename ContainerType::const_iterator;
  SetFeatureValues<IteratorType>(container.begin(), container.end(), feature);
}
// Clears the feature's repeated field (int64, float, or string). Copies
// elements from the range, defined by [first, last) into the feature's repeated
// field.
template <typename IteratorType, typename ProtoType>
void SetFeatureValues(IteratorType first, IteratorType last,
                      const std::string& key, ProtoType* proto) {
  SetFeatureValues(first, last, GetFeature(key, GetFeatures(proto)));
}

// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the container into the feature's repeated field.
template <typename ContainerType, typename ProtoType>
void SetFeatureValues(const ContainerType& container, const std::string& key,
                      ProtoType* proto) {
  using IteratorType = typename ContainerType::const_iterator;
  SetFeatureValues<IteratorType>(container.begin(), container.end(), key,
                                 proto);
}

// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the initializer list into the feature's repeated field.
template <typename ValueType, typename ProtoType>
void SetFeatureValues(std::initializer_list<ValueType> container,
                      const std::string& key, ProtoType* proto) {
  using IteratorType =
      typename std::initializer_list<ValueType>::const_iterator;
  SetFeatureValues<IteratorType>(container.begin(), container.end(), key,
                                 proto);
}

}  // namespace clink
#endif  // CORE_UTILS_FEATURE_INTERNAL_H_