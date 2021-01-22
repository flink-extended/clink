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

#ifndef CORE_UTILS_ARCHIVE_HPP_
#define CORE_UTILS_ARCHIVE_HPP_
#include <archive.h>
#include <archive_entry.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace perception_feature {
class Archive {
  static void errmsg(struct archive *handler) {
    std::cerr << archive_error_string(handler) << std::endl;
  }

  static int extract_file(struct archive *archive_read,
                          struct archive *archive_write) {
    int r;
    const void *buf;
    size_t size;
    int64_t offset;

    for (;;) {
      r = archive_read_data_block(archive_read, &buf, &size, &offset);
      if (r == ARCHIVE_EOF) {
        return ARCHIVE_OK;
      } else if (r != ARCHIVE_OK) {
        errmsg(archive_read);
        return r;
      }
      r = archive_write_data_block(archive_write, buf, size, offset);
      if (r != ARCHIVE_OK) {
        errmsg(archive_write);
        return r;
      }
    }
  }

 public:
  static int Extract(const std::string &filename,
                     const std::string &extract_dir) {
    struct archive *handler = archive_read_new();
    archive_read_support_format_all(handler);
    archive_read_support_filter_all(handler);

    struct archive *extract = archive_write_disk_new();
    int flags = ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM |
                ARCHIVE_EXTRACT_ACL | ARCHIVE_EXTRACT_FFLAGS;
    archive_write_disk_set_options(extract, flags);
    archive_write_disk_set_standard_lookup(extract);

    int r = archive_read_open_filename(handler, filename.c_str(), 16384);
    if (r != ARCHIVE_OK) errmsg(handler);

    for (; r == ARCHIVE_OK;) {
      struct archive_entry *entry;
      r = archive_read_next_header(handler, &entry);

      if (r == ARCHIVE_EOF) {
        r = ARCHIVE_OK;
        break;
      } else if (r != ARCHIVE_OK) {
        errmsg(handler);
        break;
      }
      std::string new_file(extract_dir + "/" + archive_entry_pathname(entry));
      archive_entry_set_pathname(entry, new_file.c_str());

      r = archive_write_header(extract, entry);
      if (r != ARCHIVE_OK) {
        errmsg(extract);
        break;
      } else if (archive_entry_size(entry) > 0) {
        r = extract_file(handler, extract);
        if (r != ARCHIVE_OK) break;
      }
      r = archive_write_finish_entry(extract);
      if (r != ARCHIVE_OK) {
        errmsg(extract);
        break;
      }
    }

    archive_read_close(handler);
    archive_read_free(handler);
    archive_write_close(extract);
    archive_write_free(extract);
    return r;
  }
};
}  // namespace perception_feature
#endif  // CORE_UTILS_ARCHIVE_HPP_
