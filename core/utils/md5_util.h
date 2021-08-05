/* Copyright (C) 1991-2, RSA Data Security, Inc. Created 1991. All
 rights reserved.
 License to copy and use this software is granted provided that it
 is identified as the "RSA Data Security, Inc. MD5 Message-Digest
 Algorithm" in all material mentioning or referencing this software
 or this function.
 License is also granted to make and use derivative works provided
 that such works are identified as "derived from the RSA Data
 Security, Inc. MD5 Message-Digest Algorithm" in all material
 mentioning or referencing the derived work.
 RSA Data Security, Inc. makes no representations concerning either
 the merchantability of this software or the suitability of this
 software for any particular purpose. It is provided "as is"
 without express or implied warranty of any kind.
 These notices must be retained in any copies of any part of this
 documentation and/or software.
 */

#ifndef CORE_UTILS_MD5_UTIL_H_
#define CORE_UTILS_MD5_UTIL_H_
#include <string>

namespace clink {

typedef unsigned int size_type;  // must be 32bit

typedef unsigned int MD5_U32;
const int MD5_BLOCK = 16;
const int MD5_DIGEST_LEN = 16;

struct MD5Ctx {
  uint32_t state[4];        /* state (ABCD) */
  uint32_t count[2];        /* number of bits, modulo 2^64 (lsb first) */
  unsigned char buffer[64]; /* input buffer */
};

class MD5Util {
 public:
  // 计算字符串的md5
  static void MD5(const std::string& str, std::string* md5);

  static void MD5(const void* key, size_t len, std::string* md5);

  // 计算文件中内容的md5
  static int MD5File(const std::string& filename, std::string* md5);

 private:
  static void MD5Init(MD5Ctx* c);

  static void MD5Update(const unsigned char* input, const size_t& input_len,
                        MD5Ctx* ctx);

  static void MD5Final(MD5Ctx* context, unsigned char* output);

  static void MD5Transform(const unsigned char* block, MD5Ctx* ctx);

  static void Encode(const uint32_t* input, const unsigned int& len,
                     unsigned char* output);

  static void Decode(const unsigned char* input, const unsigned int& len,
                     uint32_t* output);
};  // MD5Util
}  // namespace clink

#endif  // CORE_UTILS_MD5_UTIL_H_
