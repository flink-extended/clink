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

#ifndef CORE_UTILS_FILE_UTILS_H_
#define CORE_UTILS_FILE_UTILS_H_
#include <glog/logging.h>
#include <sys/dir.h>
#include <sys/stat.h>
#include <unistd.h>

#include <list>
#include <queue>
#include <string>
namespace perception_feature {
class FileUtils {
 public:
  static void ListFiles(std::list<std::string>& files, const std::string& path,
                        const std::string& extension, bool recursive);
  //  static bool DownLoadFile(const std::string& remote_url,
  //                           const std::string& local_path,
  //                           const std::string& archive_file);
  static inline bool FileExists(const std::string& file_name) {
    if (access(file_name.c_str(), R_OK | W_OK | F_OK) != -1) {
      return true;
    }
    return false;
  }
  static int MkDir(const std::string& pszDir);
  static std::string& GetFileName(const std::string& file_path,
                                  std::string& file_name);
  static int ExtractFile(const std::string& local_path,
                         const std::string& archive_file, bool need_parent_dir);
  static int DownLoadFile(const std::string& remote_url,
                          const std::string& local_file_name);
  static int ReadFile(const std::string& file_path, std::string& content);

  static inline std::string PathWithExtension(const std::string& file_path,
                                              const std::string& ext) {
    size_t last_dash = file_path.find_last_of('/');
    last_dash = last_dash == std::string::npos ? 0 : (last_dash + 1);
    size_t last_dot = file_path.find_last_of('.');
    if (last_dot < last_dash || last_dot == std::string::npos) {
      return file_path + ext;
    }
    return file_path;
  }
};
}  // namespace perception_feature

#endif  // CORE_UTILS_FILE_UTILS_H_
