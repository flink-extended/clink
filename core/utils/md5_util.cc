/*
 * Copyright 1995-2020 The OpenSSL Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License 2.0 (the "License").  You may not use
 * this file except in compliance with the License.  You can obtain a copy
 * in the file LICENSE in the source distribution or at
 * https://www.openssl.org/source/license.html
 */

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

#include <cstring>
#include <iostream>

/* F, G, H and I are basic MD5 functions.
 */
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))
/* ROTATE_LEFT rotates x left n bits.
 */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4.
 Rotation is separate from addition to prevent recomputation.
 */
#define FF(a, b, c, d, x, s, ac)                    \
  {                                                 \
    (a) += F((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = ROTATE_LEFT((a), (s));                    \
    (a) += (b);                                     \
  }
#define GG(a, b, c, d, x, s, ac)                    \
  {                                                 \
    (a) += G((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = ROTATE_LEFT((a), (s));                    \
    (a) += (b);                                     \
  }
#define HH(a, b, c, d, x, s, ac)                    \
  {                                                 \
    (a) += H((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = ROTATE_LEFT((a), (s));                    \
    (a) += (b);                                     \
  }
#define II(a, b, c, d, x, s, ac)                    \
  {                                                 \
    (a) += I((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = ROTATE_LEFT((a), (s));                    \
    (a) += (b);                                     \
  }

namespace clink {
// Constants for MD5Transform routine.
const uint32_t S11 = 7;
const uint32_t S12 = 12;
const uint32_t S13 = 17;
const uint32_t S14 = 22;
const uint32_t S21 = 5;
const uint32_t S22 = 9;
const uint32_t S23 = 14;
const uint32_t S24 = 20;
const uint32_t S31 = 4;
const uint32_t S32 = 11;
const uint32_t S33 = 16;
const uint32_t S34 = 23;
const uint32_t S41 = 6;
const uint32_t S42 = 10;
const uint32_t S43 = 15;
const uint32_t S44 = 21;

void MD5Util::MD5(const std::string& str, std::string* md5) {
  MD5(str.c_str(), str.size(), md5);
}

void MD5Util::MD5(const void* key, size_t len, std::string* md5) {
  unsigned char results[MD5_DIGEST_LEN];
  clink::MD5Ctx my_md5;
  MD5Init(&my_md5);
  MD5Update((const unsigned char*)key, len, &my_md5);
  MD5Final(&my_md5, results);

  char tmp[3] = {'\0'};
  *md5 = "";

  for (int i = 0; i < MD5_DIGEST_LEN; i++) {
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

  MD5Ctx ctx;
  MD5Init(&ctx);

  while ((bytes = fread(data, 1, 1024, f)) != 0) {
    MD5Update(data, bytes, &ctx);
  }

  unsigned char results[MD5_DIGEST_LEN];
  MD5Final(&ctx, results);

  fclose(f);

  char tmp[3] = {'\0'};
  *md5 = "";

  for (int i = 0; i < MD5_DIGEST_LEN; i++) {
    snprintf(tmp, sizeof(tmp), "%02x", results[i]);
    *md5 += tmp;
  }
  return 0;
}

void MD5Util::MD5Init(MD5Ctx* ctx) {
  ctx->count[0] = ctx->count[1] = 0;
  /* Load magic initialization constants.
   */
  ctx->state[0] = 0x67452301;
  ctx->state[1] = 0xefcdab89;
  ctx->state[2] = 0x98badcfe;
  ctx->state[3] = 0x10325476;
}

void MD5Util::MD5Update(const unsigned char* input, const size_t& input_len,
                        MD5Ctx* context) {
  unsigned int i, index, part_len;
  /* Compute number of bytes mod 64 */
  index = (unsigned int)((context->count[0] >> 3) & 0x3F);
  /* Update number of bits */
  if ((context->count[0] += ((uint32_t)input_len << 3)) <
      ((uint32_t)input_len << 3)) {
    context->count[1]++;
  }
  context->count[1] += ((uint32_t)input_len >> 29);
  part_len = 64 - index;
  /* Transform as many times as possible.
   */
  if (input_len >= part_len) {
    memcpy(&context->buffer[index], input, part_len);
    MD5Transform(context->buffer, context);
    for (i = part_len; i + 63 < input_len; i += 64) {
      MD5Transform(&input[i], context);
    }
    index = 0;
  } else {
    i = 0;
  }
  /* Buffer remaining input */
  memcpy(&context->buffer[index], &input[i], input_len - i);
}

void MD5Util::MD5Final(MD5Ctx* context, unsigned char* output) {
  unsigned char bits[8];
  unsigned int index, pad_len;
  static unsigned char padding[64] = {
      0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  /* Save number of bits */
  Encode(context->count, 8, bits);
  /* Pad out to 56 mod 64.
   */
  index = (unsigned int)((context->count[0] >> 3) & 0x3f);
  pad_len = (index < 56) ? (56 - index) : (120 - index);
  MD5Update(padding, pad_len, context);
  /* Append length (after padding) */
  MD5Update(bits, 8, context);
  /* Store state in digest */
  Encode(context->state, 16, output);
  /* Zeroize sensitive information.
   */
  memset(context, 0, sizeof(*context));
}

void MD5Util::MD5Transform(const unsigned char* block, MD5Ctx* ctx) {
  uint32_t a = ctx->state[0];
  uint32_t b = ctx->state[1];
  uint32_t c = ctx->state[2];
  uint32_t d = ctx->state[3];
  uint32_t x[16];
  Decode(block, 64, x);
  /* Round 1 */
  FF(a, b, c, d, x[0], S11, 0xd76aa478);  /* 1 */
  FF(d, a, b, c, x[1], S12, 0xe8c7b756);  /* 2 */
  FF(c, d, a, b, x[2], S13, 0x242070db);  /* 3 */
  FF(b, c, d, a, x[3], S14, 0xc1bdceee);  /* 4 */
  FF(a, b, c, d, x[4], S11, 0xf57c0faf);  /* 5 */
  FF(d, a, b, c, x[5], S12, 0x4787c62a);  /* 6 */
  FF(c, d, a, b, x[6], S13, 0xa8304613);  /* 7 */
  FF(b, c, d, a, x[7], S14, 0xfd469501);  /* 8 */
  FF(a, b, c, d, x[8], S11, 0x698098d8);  /* 9 */
  FF(d, a, b, c, x[9], S12, 0x8b44f7af);  /* 10 */
  FF(c, d, a, b, x[10], S13, 0xffff5bb1); /* 11 */
  FF(b, c, d, a, x[11], S14, 0x895cd7be); /* 12 */
  FF(a, b, c, d, x[12], S11, 0x6b901122); /* 13 */
  FF(d, a, b, c, x[13], S12, 0xfd987193); /* 14 */
  FF(c, d, a, b, x[14], S13, 0xa679438e); /* 15 */
  FF(b, c, d, a, x[15], S14, 0x49b40821); /* 16 */
  /* Round 2 */
  GG(a, b, c, d, x[1], S21, 0xf61e2562);  /* 17 */
  GG(d, a, b, c, x[6], S22, 0xc040b340);  /* 18 */
  GG(c, d, a, b, x[11], S23, 0x265e5a51); /* 19 */
  GG(b, c, d, a, x[0], S24, 0xe9b6c7aa);  /* 20 */
  GG(a, b, c, d, x[5], S21, 0xd62f105d);  /* 21 */
  GG(d, a, b, c, x[10], S22, 0x2441453);  /* 22 */
  GG(c, d, a, b, x[15], S23, 0xd8a1e681); /* 23 */
  GG(b, c, d, a, x[4], S24, 0xe7d3fbc8);  /* 24 */
  GG(a, b, c, d, x[9], S21, 0x21e1cde6);  /* 25 */
  GG(d, a, b, c, x[14], S22, 0xc33707d6); /* 26 */
  GG(c, d, a, b, x[3], S23, 0xf4d50d87);  /* 27 */
  GG(b, c, d, a, x[8], S24, 0x455a14ed);  /* 28 */
  GG(a, b, c, d, x[13], S21, 0xa9e3e905); /* 29 */
  GG(d, a, b, c, x[2], S22, 0xfcefa3f8);  /* 30 */
  GG(c, d, a, b, x[7], S23, 0x676f02d9);  /* 31 */
  GG(b, c, d, a, x[12], S24, 0x8d2a4c8a); /* 32 */
  /* Round 3 */
  HH(a, b, c, d, x[5], S31, 0xfffa3942);  /* 33 */
  HH(d, a, b, c, x[8], S32, 0x8771f681);  /* 34 */
  HH(c, d, a, b, x[11], S33, 0x6d9d6122); /* 35 */
  HH(b, c, d, a, x[14], S34, 0xfde5380c); /* 36 */
  HH(a, b, c, d, x[1], S31, 0xa4beea44);  /* 37 */
  HH(d, a, b, c, x[4], S32, 0x4bdecfa9);  /* 38 */
  HH(c, d, a, b, x[7], S33, 0xf6bb4b60);  /* 39 */
  HH(b, c, d, a, x[10], S34, 0xbebfbc70); /* 40 */
  HH(a, b, c, d, x[13], S31, 0x289b7ec6); /* 41 */
  HH(d, a, b, c, x[0], S32, 0xeaa127fa);  /* 42 */
  HH(c, d, a, b, x[3], S33, 0xd4ef3085);  /* 43 */
  HH(b, c, d, a, x[6], S34, 0x4881d05);   /* 44 */
  HH(a, b, c, d, x[9], S31, 0xd9d4d039);  /* 45 */
  HH(d, a, b, c, x[12], S32, 0xe6db99e5); /* 46 */
  HH(c, d, a, b, x[15], S33, 0x1fa27cf8); /* 47 */
  HH(b, c, d, a, x[2], S34, 0xc4ac5665);  /* 48 */
  /* Round 4 */
  II(a, b, c, d, x[0], S41, 0xf4292244);  /* 49 */
  II(d, a, b, c, x[7], S42, 0x432aff97);  /* 50 */
  II(c, d, a, b, x[14], S43, 0xab9423a7); /* 51 */
  II(b, c, d, a, x[5], S44, 0xfc93a039);  /* 52 */
  II(a, b, c, d, x[12], S41, 0x655b59c3); /* 53 */
  II(d, a, b, c, x[3], S42, 0x8f0ccc92);  /* 54 */
  II(c, d, a, b, x[10], S43, 0xffeff47d); /* 55 */
  II(b, c, d, a, x[1], S44, 0x85845dd1);  /* 56 */
  II(a, b, c, d, x[8], S41, 0x6fa87e4f);  /* 57 */
  II(d, a, b, c, x[15], S42, 0xfe2ce6e0); /* 58 */
  II(c, d, a, b, x[6], S43, 0xa3014314);  /* 59 */
  II(b, c, d, a, x[13], S44, 0x4e0811a1); /* 60 */
  II(a, b, c, d, x[4], S41, 0xf7537e82);  /* 61 */
  II(d, a, b, c, x[11], S42, 0xbd3af235); /* 62 */
  II(c, d, a, b, x[2], S43, 0x2ad7d2bb);  /* 63 */
  II(b, c, d, a, x[9], S44, 0xeb86d391);  /* 64 */
  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  /* Zeroize sensitive information.
   */
  memset(x, 0, sizeof(x));
}

void MD5Util::Encode(const uint32_t* input, const unsigned int& len,
                     unsigned char* output) {
  unsigned int i, j;
  for (i = 0, j = 0; j < len; i += 1, j += 4) {
    output[j] = (unsigned char)(input[i] & 0xff);
    output[j + 1] = (unsigned char)((input[i] >> 8) & 0xff);
    output[j + 2] = (unsigned char)((input[i] >> 16) & 0xff);
    output[j + 3] = (unsigned char)((input[i] >> 24) & 0xff);
  }
}

void MD5Util::Decode(const unsigned char* input, const unsigned int& len,
                     uint32_t* output) {
  unsigned int i, j;
  for (i = 0, j = 0; j < len; i++, j += 4) {
    output[i] = ((uint32_t)input[j]) | (((uint32_t)input[j + 1]) << 8) |
                (((uint32_t)input[j + 2]) << 16) |
                (((uint32_t)input[j + 3]) << 24);
  }
}
}  // namespace clink
