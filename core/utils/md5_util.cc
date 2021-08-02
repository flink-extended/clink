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

#include "md5_util.h"

#include <openssl/md5.h>
#include <stdio.h>

#include <iostream>


namespace clink {

void MD5Util::MD5(const std::string& str, std::string* md5) {
  MD5(str.c_str(), str.size(), md5);
}

void MD5Util::MD5(const void* key, size_t len, std::string* md5) {
  unsigned char results[MD5_DIGEST_LENGTH];
  MD5_CTX my_md5;
  MD5_Init(&my_md5);
  MD5_Update(&my_md5, (const unsigned char*)key, len);
  MD5_Final(results, &my_md5);

  char tmp[3] = {'\0'};
  *md5 = "";

  for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
    snprintf(tmp, sizeof(tmp), "%02x", results[i]);
    *md5 += tmp;
  }
}

int MD5Util::MD5File(const std::string& filename, std::string* md5) {
  FILE* f = fopen(filename.c_str(), "rb");
  if (f == NULL) {
    std::cerr << "MD5 file failed, open file error, filename:" << filename
              << std::endl;
    return -1;
  }

  int bytes;
  unsigned char data[1024];

  MD5_CTX ctx;
  MD5_Init(&ctx);

  while ((bytes = fread(data, 1, 1024, f)) != 0) {
    MD5_Update(&ctx, data, bytes);
  }

  unsigned char results[MD5_DIGEST_LENGTH];
  MD5_Final(results, &ctx);

  fclose(f);

  char tmp[3] = {'\0'};
  *md5 = "";

  for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
    snprintf(tmp, sizeof(tmp), "%02x", results[i]);
    *md5 += tmp;
  }
  return 0;
}

}  // namespace clink
