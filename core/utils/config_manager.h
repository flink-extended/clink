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

#ifndef CORE_UTILS_CONFIG_MANAGER_H_
#define CORE_UTILS_CONFIG_MANAGER_H_
#include <iostream>
#include <string>
namespace clink {
class ConfigManager {
 public:
  static int FetchConfig(const std::string& remote_url,
                         const std::string& archive_file);
  static int ExtractConfig(const std::string& archive_file);
  static int FetchAndExtractConfig(const std::string& remote_url,
                                   const std::string& archive_file);
  static bool Md5Checksum(const std::string& archive_file,
                          const std::string& md5);
};
}  // namespace clink

#endif  // CORE_UTILS_CONFIG_MANAGER_H_
