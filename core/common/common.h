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
#ifndef CORE_COMMON_COMMON_H_
#define CORE_COMMON_COMMON_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "core/protos/common.pb.h"
#include "core/protos/datasource.pb.h"
#include "core/protos/interface.pb.h"
#include "core/protos/operations.pb.h"
#include "core/utils/murmurhash.h"
namespace clink {
#define MAKE_HASH(key) MurmurHash64A(key.c_str(), key.size(), 0)
using Feature = proto::Record;
using OpParam = proto::Record;
using FeatureMap = std::unordered_map<int64_t, Feature*>;
using OpParamMap = std::unordered_map<std::string, proto::Record>;
using OperationList = proto::OperationList;
using Operation = proto::Operation;
using Transform = proto::Transform;
using DataSource = proto::DataSource;
using DataSourceList = proto::DataSourceList;
using CsvDataConfig = proto::CsvDataConfig;
using CsvDataConfigList = proto::CsvDataConfigList;
using OutputFromat = proto::OutputFormat;
using FeatureType = proto::FeatureType;
using IVRecordEntry = proto::IVRecordEntry;
using IVRecordList = proto::IVRecordList;
using SampleRecord = proto::SampleRecord;
using DinResultRecord = proto::DinResultRecord;
typedef enum {
  STATUS_OK = 0,
  ERR_NOT_INIT = 1,
  ERR_CONFIG_EMPTY = 2,
  ERR_INVALID_JSON = 3,
  ERR_INVALID_TYPE = 4,
  ERR_INVALID_PB = 5,
  ERR_OPERATION_EMPTY = 6,
  ERR_DATASOURCE_EMPTY = 7,
  ERR_DATASOURCE_CONFIG = 8,
  ERR_PARSE_INPUT_DATA = 9,
  ERR_UNKNOWN_FIELD = 10,
  ERR_PARSE_TRANSFORM = 11,
  ERR_DUPLICATE_FEATURE = 12,
  ERR_TOP_SORT = 13,
  ERR_EMPTY_INPUT = 14,
  ERR_INVALID_CONFIG = 15,
  STATUS_PARSE_INPUT_ERROR,
  STATUS_FILE_LIST_EMPTY = 16,
  ERR_MISSING_FEATURE = 17,
  ERR_MISSING_PARAM = 18,
  ERR_OP_NOT_INIT = 19,
  ERR_OP_STATUS_FAILED = 20,
  ERROR_EMPTY_EXPRESSION = 21,
  ERR_DOWNLOAD_FILE = 22,
  ERR_EXTRACT_FILE = 23,
  ERR_INVALID_VEC_SIZE = 24,
  ERR_INVALID_VALUE = 25,
  ERR_MD5_CHECK = 26,
  ERR_READ_FILE = 27,
  ERR_INDEX_OUT_BOUNDARY = 28,
  ERR_INDEX_VALUE_UNEQUAL = 29,
} StatusCode;

const char digits[] = "0123456789";
const char integer_chars[] = "0123456789-eE";
const char real_chars[] = "0123456789-eE.";
}  // namespace clink
#endif  // CORE_COMMON_COMMON_H_
