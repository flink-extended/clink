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

#include "parser_factory.h"

#include <butil/logging.h>

#include <fstream>

#include "core/common/common.h"
#include "core/source/csv_source_parser.h"
#include "core/source/source_parser_base.h"
#include "core/utils/proto_json.h"

namespace clink {
static const char SOURCE_CONFIG[] = "datasource.conf";

std::shared_ptr<SourceParserBase> SourceFactory::CreateSourceParser(
    const std::string& config_path) {
  std::string config_file = config_path + "/" + SOURCE_CONFIG;
  std::ifstream ifs;
  std::string json_str;
  ifs.open(config_file, std::ios::in | std::ios::binary);
  json_str.assign(std::istreambuf_iterator<char>(ifs),
                  std::istreambuf_iterator<char>());
  ifs.close();
  proto::DataSourceList data_source_list;
  int status = ProtoJson::json2pb(json_str, data_source_list, true);
  if (status != STATUS_OK) {
    LOG(ERROR) << "Parse config file  error, conf:" << config_file;
    return nullptr;
  }
  if (!data_source_list.IsInitialized() ||
      data_source_list.data_source_size() <= 0) {
    return nullptr;
  }
  auto& it = data_source_list.data_source(0);
  std::shared_ptr<SourceParserBase> source_parser = nullptr;
  if (it.data_type() == proto::CSV_DATA && !it.data_conf().empty()) {
    source_parser = std::make_shared<CsvSourceParser>(it);
    status = source_parser->LoadConfig(config_path);
    if (status != STATUS_OK) {
      LOG(ERROR) << "failed to load data data_source, parse error";
      return nullptr;
    }
  }
  return source_parser;
}
}  // namespace clink