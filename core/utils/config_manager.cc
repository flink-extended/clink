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

#include "core/utils/config_manager.h"

#include <string>

#include "core/common/common.h"
#include "core/utils/file_utils.h"
#include "core/utils/md5_util.h"
namespace clink {
int ConfigManager::FetchConfig(const std::string &remote_url,
                               const std::string &archive_file) {
  std::string parent_dir = ".";
  if (archive_file.find_last_of('/') != std::string::npos) {
    parent_dir = archive_file.substr(0, archive_file.find_last_of('/'));
  }

  int err_code;
  if ((err_code = FileUtils::MkDir(parent_dir)) != 0) {
    std::cerr << "Making parent directory:" << parent_dir << " of "
              << archive_file << " failed!";
    return err_code;
  }
  if ((err_code = FileUtils::DownLoadFile(remote_url, archive_file)) != 0) {
    std::cerr << "Download config url:" << remote_url
              << " failed, errno:" << err_code;
  }
  return err_code;
}

int ConfigManager::ExtractConfig(const std::string &archive_file) {
  if (!FileUtils::FileExists(archive_file)) {
    std::cerr << "Can not find archive file:" << archive_file;
    return ERR_READ_FILE;
  }
  std::string extract_path = ".";
  if (archive_file.find_last_of('/') != std::string::npos) {
    extract_path = archive_file.substr(0, archive_file.find_last_of('/'));
  }
  int err_code;
  if ((err_code = FileUtils::ExtractFile(extract_path, archive_file, false)) !=
      0) {
    std::cerr << "Extract archive file:" << archive_file
              << " failed, errno:" << err_code;
  }
  if (remove(archive_file.c_str()) == 0) {
    std::cout << "删除本地压缩文件" << archive_file << std::endl;
  } else {
    std::cerr << "删除失败" << archive_file << std::endl;
  }
  return err_code;
}

int ConfigManager::FetchAndExtractConfig(const std::string &remote_url,
                                         const std::string &local_path) {
  std::string archive_file = local_path + "/";
  std::string file_name;
  FileUtils::GetFileName(remote_url, file_name);
  archive_file += file_name;
  int err_code;
  if ((err_code = FetchConfig(remote_url, archive_file)) != 0) return err_code;
  if ((err_code = ExtractConfig(archive_file)) != 0) return err_code;
  return 0;
}

bool ConfigManager::Md5Checksum(const std::string &archive_file,
                                const std::string &md5) {
  if (md5.empty() || !FileUtils::FileExists(archive_file)) {
    return false;
  }
  std::string file_md5;
  MD5Util::MD5File(archive_file, &file_md5);
  return file_md5.compare(md5);
}

}  // namespace clink
