/*
 The MIT License (MIT)
Copyright (c) 2013 shafreeck renenglish at gmail dot com
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef CORE_UTILS_PROTO_JSON_H_
#define CORE_UTILS_PROTO_JSON_H_
#include <google/protobuf/message.h>
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <memory>
#include <string>

#include "core/common/common.h"
#include "core/utils/bin2ascii.h"
namespace clink {
class ProtoJson {
  using Message = google::protobuf::Message;
  using Descriptor = google::protobuf::Descriptor;
  using FieldDescriptor = google::protobuf::FieldDescriptor;
  using Reflection = google::protobuf::Reflection;
  using EnumValueDescriptor = google::protobuf::EnumValueDescriptor;
  using EnumDescriptor = google::protobuf::EnumDescriptor;

 public:
  static void pb2json(const Message& msg, std::string& str);

  static int JsonToProto(const std::string& json, Message& msg,
                     const bool& base64_encode);

 private:
  static std::shared_ptr<rapidjson::Value> parse_message(
      const Message& msg, rapidjson::Value::AllocatorType& allocator);
  //   static void hex_encode(const std::string& input, std::string& output);

  static std::shared_ptr<rapidjson::Value> field2json(
      const Message& msg, const FieldDescriptor* field,
      rapidjson::Value::AllocatorType& allocator);

  static void json2string(const std::shared_ptr<rapidjson::Value>& json,
                          std::string& str);

  static int parse_json(const rapidjson::Value& json, Message& msg,
                        const bool& base64_encode);

  static int json2field(const rapidjson::Value& json, Message& msg,
                        const FieldDescriptor* field,
                        const bool& base64_encode);
};
}  // namespace clink
#endif  // CORE_UTILS_PROTO_JSON_H_