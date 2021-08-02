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

#include "core/operators/bucket.h"

#include <butil/logging.h>

#include <iostream>
#include <string>
#include <vector>

#include "core/utils/convert_util.h"
#include "core/utils/feature_internal.h"
#include "core/utils/feature_util.h"
namespace clink {
Bucket::Bucket() : BaseOperator(1) {}

Bucket::Bucket(const std::string& feature_name, const OpParamMap& param_map)
    : BaseOperator(1) {
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  if (!ParseParam(feature_name, param_map)) {
    init_status_ = false;
  }
  init_status_ = true;
}

const Feature* Bucket::Evaluate(Context* context) {
  Feature* output = context->CreateMessage();
  if (!init_status_ || opa_num_ != 1 || children_.empty() ||
      children_[0] == nullptr) {
    LOG(ERROR) << "Bucket Evaluate failed, not init";
    return nullptr;
  }
  auto child = children_[0]->Evaluate(context);
  if (child == nullptr) {
    LOG(ERROR) << "Bucket Evaluate failed";
    return nullptr;
  }
  // auto var = children_[0]->GetOperationName();
  // if (var == nullptr) {
  //   LOG(ERROR) << "Bucket Evaluate failed";
  //   return nullptr;
  // }
  int index;
  GetBucketIndex(*child, &index);
  GetFeatureValues<int>(output)->Add(index);
  return output;
}

int Bucket::GetBucketIndex(const Feature& child) {
  double value;
  ConvertUtil::ToDouble(child, value);
  int vec_size = boundaries_.size();
  int index = vec_size;
  if (value < boundaries_.at(vec_size - 1)) {
    index = std::upper_bound(boundaries_.begin(), boundaries_.end(), value) -
            boundaries_.begin();
  }
  return index;
}

std::shared_ptr<BaseOperator> Bucket::Clone() const {
  return std::make_shared<Bucket>();
}

bool Bucket::ParseParamMap(const std::string& feature_name,
                           const OpParamMap& param_map) {
  return ParseParam(feature_name, param_map);
}

bool Bucket::ParseParam(const std::string& feature_name,
                        const OpParamMap& param_map) {
  std::string key = feature_name + boundary_postfix;
  OpParam bucket_param;
  if (!FeatureUtil::GetParam(param_map, key, bucket_param)) {
    return false;
  }
  ParseBucketBoundaries(bucket_param);
  return true;
}

void Bucket::ParseBucketBoundaries(const OpParam& bucket_param) {
  RecordType type = FeatureUtil::GetType(bucket_param);
  switch (type) {
    case RECORD_TYPE_FLOAT: {
      for (int i = 0; i < bucket_param.float_list().value_size(); ++i) {
        boundaries_.emplace_back(bucket_param.float_list().value(i));
      }
      break;
    }
    case RECORD_TYPE_DOUBLE: {
      for (int i = 0; i < bucket_param.double_list().value_size(); ++i) {
        boundaries_.emplace_back(bucket_param.double_list().value(i));
      }
      break;
    }
    case RECORD_TYPE_INT: {
      for (int i = 0; i < bucket_param.int_list().value_size(); ++i) {
        boundaries_.emplace_back(bucket_param.int_list().value(i));
      }
      break;
    }
    case RECORD_TYPE_INT64: {
      for (int i = 0; i < bucket_param.int64_list().value_size(); ++i) {
        boundaries_.emplace_back(bucket_param.int64_list().value(i));
      }
      break;
    }
    default:
      break;
  }
}

bool Bucket::ParseParamMap(const std::vector<std::string>& variables,
                           const OpParamMap& param_map) {
  for (auto& item : variables) {
    ParseParam(item, param_map);
  }
  return true;
}

}  // namespace clink
