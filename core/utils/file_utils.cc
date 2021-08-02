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

#include "core/utils/file_utils.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <cstdio>
#include <fstream>

#include "archive.hpp"
#include "core/common/common.h"
#include "curl.hpp"
namespace clink {

void FileUtils::ListFiles(std::list<std::string>& files,
                          const std::string& path, const std::string& extension,
                          bool recursive) {
  DIR* dir = nullptr;
  DIR* sub_dir = nullptr;
  struct dirent* ent;
  dir = opendir(path.c_str());
  if (dir == nullptr) {
    LOG(ERROR) << "Could not open directory:" << path;
    return;
  } else {
    closedir(dir);
  }

  std::queue<std::string> paths;
  paths.push(path);

  while (!paths.empty()) {
    std::string one_path = paths.front();
    paths.pop();
    dir = opendir(one_path.c_str());
    if (dir == nullptr) {
      continue;
    }

    while ((ent = readdir(dir)) != nullptr) {
      std::string name(ent->d_name);
      if (name.compare(".") == 0 || name.compare("..") == 0) {
        continue;
      }
      // add current_path to the file name
      std::string current_path = one_path;
      current_path.append("/");
      current_path.append(name);
      // check if it's a folder by trying to open it
      sub_dir = opendir(current_path.c_str());
      if (sub_dir != NULL) {  // it's a folder
        closedir(sub_dir);
        if (recursive) {
          paths.push(current_path);
        }
      } else {  // it's a file
        if (extension.empty()) {
          files.emplace_back(current_path);
        } else {  // check file extension
          if (name.substr(name.find_last_of('.')).compare(extension) == 0) {
            files.emplace_back(current_path);
          }
        }
      }
    }
    closedir(dir);
  }
}
// bool FileUtils::FileExists(const std::string& file_name) {
//  if (access(file_name.c_str(), R_OK | W_OK | F_OK) != -1) {
//    return true;
//  }
//  return false;
//}
int FileUtils::MkDir(const std::string& path) {
  size_t pos = 0, last_pose = 1, ret = 0, len;
  std::string sub_path = path;

  if (sub_path.at(path.length() - 1) != '/') {
    sub_path.append("/");
  }
  len = sub_path.length();
  for (int i = 1; i < len; i++) {
    if (sub_path[i] == '/') {
      sub_path[i] = '\0';
      //如果不存在,创建
      ret = access(sub_path.c_str(), 0);
      if (ret != 0) {
        ret = mkdir(sub_path.c_str(), 0755);
        if (ret != 0) {
          return -1;
        }
      }
      // 支持linux,将所有\换成/
      sub_path[i] = '/';
    }
  }
  return 0;
}
int FileUtils::DownLoadFile(const std::string& remote_url,
                            const std::string& local_file_path) {
  int err_code;
  if ((err_code = Curl::Download(remote_url, local_file_path)) != 0) {
    LOG(WARNING) << "Download plugin url:" << remote_url
                 << " failed, errno:" << err_code;
    return ERR_DOWNLOAD_FILE;
  }
  return STATUS_OK;
}

std::string& FileUtils::GetFileName(const std::string& file_path,
                                    std::string& file_name) {
  size_t last_dash = file_path.find_last_of("/");
  if (last_dash == std::string::npos) {
    last_dash = 0;
  } else {
    ++last_dash;
  }
  file_name = file_path.substr(last_dash, file_path.length());
  return file_name;
}
int FileUtils::ExtractFile(const std::string& local_path,
                           const std::string& archive_file,
                           bool need_parent_dir) {
  std::string extract_path = local_path;
  if (need_parent_dir) {
    extract_path += +"/conf/";
  }
  if (!FileExists(extract_path)) {
    LOG(INFO) << "Make directory:" << extract_path;
    FileUtils::MkDir(extract_path.c_str());
  }

  if (!FileUtils::FileExists(archive_file)) {
    LOG(WARNING) << "Can not find archive file:" << archive_file;
    return ERR_EXTRACT_FILE;
  }

  int err_code;
  if ((err_code = Archive::Extract(archive_file, extract_path)) != 0) {
    LOG(WARNING) << "Extract archive file:" << archive_file
                 << " failed, errno:" << err_code;
    return ERR_EXTRACT_FILE;
  }
  return STATUS_OK;
}
int FileUtils::ReadFile(const std::string& file_path, std::string& content) {
  std::ifstream ifs;
  ifs.open(file_path, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    LOG(ERROR) << "error in open file " << file_path << std::endl;
    return ERR_READ_FILE;
  }
  content.assign(std::istreambuf_iterator<char>(ifs),
                 std::istreambuf_iterator<char>());
  ifs.close();
  return STATUS_OK;
}

}  // namespace clink
