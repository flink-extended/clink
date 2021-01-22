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

#ifndef CORE_UTILS_CURL_HPP_
#define CORE_UTILS_CURL_HPP_
#include <curl/curl.h>

#include <iostream>
#include <string>

namespace perception_feature {
class Curl {
 public:
  static int Download(const std::string &url, const std::string &filename) {
    CURL *curl = curl_easy_init();
    FILE *fp;
    if (curl) {
      fp = fopen(filename.c_str(), "wb");
      if (fp == NULL) {
        std::cerr << "Open file error, file:" << filename << std::endl;
        return -1;
      }
      curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
      curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
      CURLcode r = curl_easy_perform(curl);
      curl_easy_cleanup(curl);
      fclose(fp);
      return r;
    }
    return -1;
  }
};
}  // namespace perception_feature

#endif  // CORE_UTILS_CURL_HPP_
