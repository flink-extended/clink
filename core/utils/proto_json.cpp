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

#include "core/utils/proto_json.h"

#include <glog/logging.h>

#include <iostream>
namespace perception_feature {
void ProtoJson::hex_encode(const std::string& input, std::string& output) {
  static const char* const lut = "0123456789abcdef";
  size_t len = input.length();

  output.reserve(2 * len);
  for (size_t i = 0; i < len; ++i) {
    const unsigned char c = input[i];
    output.push_back(lut[c >> 4]);
    output.push_back(lut[c & 15]);
  }
  return;
}

std::shared_ptr<rapidjson::Value> ProtoJson::field2json(
    const Message& msg, const FieldDescriptor* field,
    rapidjson::Value::AllocatorType& allocator) {
  const Reflection* ref = msg.GetReflection();
  const bool repeated = field->is_repeated();

  size_t array_size = 0;
  if (repeated) {
    array_size = ref->FieldSize(msg, field);
  }
  std::shared_ptr<rapidjson::Value> json = nullptr;
  if (repeated) {
    json = std::make_shared<rapidjson::Value>(rapidjson::kArrayType);
  }
  switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_DOUBLE:
      if (repeated) {
        for (size_t i = 0; i != array_size; ++i) {
          double value = ref->GetRepeatedDouble(msg, field, i);
          rapidjson::Value v(value);
          json->PushBack(v, allocator);
        }
      } else {
        json = std::make_shared<rapidjson::Value>(ref->GetDouble(msg, field));
      }
      break;
    case FieldDescriptor::CPPTYPE_FLOAT:
      if (repeated) {
        for (size_t i = 0; i != array_size; ++i) {
          float value = ref->GetRepeatedFloat(msg, field, i);
          rapidjson::Value v(value);
          json->PushBack(v, allocator);
        }
      } else {
        json = std::make_shared<rapidjson::Value>(ref->GetFloat(msg, field));
      }
      break;
    case FieldDescriptor::CPPTYPE_INT64:
      if (repeated) {
        for (size_t i = 0; i != array_size; ++i) {
          int64_t value = ref->GetRepeatedInt64(msg, field, i);
          rapidjson::Value v(value);
          json->PushBack(v, allocator);
        }
      } else {
        json = std::make_shared<rapidjson::Value>(
            static_cast<int64_t>(ref->GetInt64(msg, field)));
      }
      break;
    case FieldDescriptor::CPPTYPE_UINT64:
      if (repeated) {
        for (size_t i = 0; i != array_size; ++i) {
          uint64_t value = ref->GetRepeatedUInt64(msg, field, i);
          rapidjson::Value v(value);
          json->PushBack(v, allocator);
        }
      } else {
        json = std::make_shared<rapidjson::Value>(
            static_cast<uint64_t>(ref->GetUInt64(msg, field)));
      }
      break;
    case FieldDescriptor::CPPTYPE_INT32:
      if (repeated) {
        for (size_t i = 0; i != array_size; ++i) {
          int32_t value = ref->GetRepeatedInt32(msg, field, i);
          rapidjson::Value v(value);
          json->PushBack(v, allocator);
        }
      } else {
        json = std::make_shared<rapidjson::Value>(ref->GetInt32(msg, field));
      }
      break;
    case FieldDescriptor::CPPTYPE_UINT32:
      if (repeated) {
        for (size_t i = 0; i != array_size; ++i) {
          uint32_t value = ref->GetRepeatedUInt32(msg, field, i);
          rapidjson::Value v(value);
          json->PushBack(v, allocator);
        }
      } else {
        json = std::make_shared<rapidjson::Value>(ref->GetUInt32(msg, field));
      }
      break;
    case FieldDescriptor::CPPTYPE_BOOL:
      if (repeated) {
        for (size_t i = 0; i != array_size; ++i) {
          bool value = ref->GetRepeatedBool(msg, field, i);
          rapidjson::Value v(value);
          json->PushBack(v, allocator);
        }
      } else {
        json = std::make_shared<rapidjson::Value>(ref->GetBool(msg, field));
      }
      break;
    case FieldDescriptor::CPPTYPE_STRING: {
      bool is_binary = field->type() == FieldDescriptor::TYPE_BYTES;
      if (repeated) {
        for (size_t i = 0; i != array_size; ++i) {
          std::string output;
          if (is_binary) {
            b64_encode(ref->GetRepeatedString(msg, field, i), output);
          } else {
            output = ref->GetRepeatedString(msg, field, i);
          }
          rapidjson::Value v(output.c_str(),
                             static_cast<rapidjson::SizeType>(output.size()),
                             allocator);
          json->PushBack(v, allocator);
        }
      } else {
        std::string output;
        if (is_binary) {
          b64_encode(ref->GetString(msg, field), output);
        } else {
          output = ref->GetString(msg, field);
        }
        json = std::make_shared<rapidjson::Value>(output.c_str(), output.size(),
                                                  allocator);
      }
      break;
    }
    case FieldDescriptor::CPPTYPE_MESSAGE:
      if (repeated) {
        for (size_t i = 0; i != array_size; ++i) {
          const Message& value = ref->GetRepeatedMessage(msg, field, i);
          std::shared_ptr<rapidjson::Value> v = parse_message(value, allocator);
          json->PushBack(*v, allocator);
        }
      } else {
        const Message* value = &(ref->GetMessage(msg, field));
        json = parse_message(*value, allocator);
      }
      break;
    case FieldDescriptor::CPPTYPE_ENUM:
      if (repeated) {
        for (size_t i = 0; i != array_size; ++i) {
          const EnumValueDescriptor* value =
              ref->GetRepeatedEnum(msg, field, i);
          rapidjson::Value v(value->number());
          json->PushBack(v, allocator);
        }
      } else {
        json = std::make_shared<rapidjson::Value>(
            ref->GetEnum(msg, field)->number());
      }
      break;
    default:
      break;
  }
  return json;
}

std::shared_ptr<rapidjson::Value> ProtoJson::parse_message(
    const Message& msg, rapidjson::Value::AllocatorType& allocator) {
  const Descriptor* d = msg.GetDescriptor();
  if (!d) return nullptr;
  size_t count = d->field_count();

  std::shared_ptr<rapidjson::Value> root =
      std::make_shared<rapidjson::Value>(rapidjson::kObjectType);
  if (!root) return nullptr;
  for (size_t i = 0; i != count; ++i) {
    const FieldDescriptor* field = d->field(i);
    if (!field) {
      return nullptr;
    }

    const Reflection* ref = msg.GetReflection();
    if (!ref) {
      return nullptr;
    }
    if (field->is_optional() && !ref->HasField(msg, field)) {
      // do nothing
    } else {
      std::shared_ptr<rapidjson::Value> field_json =
          field2json(msg, field, allocator);
      rapidjson::Value field_name(field->name().c_str(), field->name().size());
      root->AddMember(field_name, *field_json, allocator);
    }
  }
  return root;
}

void ProtoJson::json2string(const std::shared_ptr<rapidjson::Value>& json,
                            std::string& str) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  json->Accept(writer);
  str.append(buffer.GetString(), buffer.GetSize());
}

int ProtoJson::json2pb(const std::string& json, Message& msg,
                       const bool& base64_encode) {
  rapidjson::Document root;
  root.Parse(json.c_str());
  if (root.HasParseError()) {
    return ERR_INVALID_JSON;
  }
  return parse_json(root, msg, base64_encode);
}

int ProtoJson::parse_json(const rapidjson::Value& root, Message& msg,
                          const bool& base64_encode) {
  if (!root.IsObject()) {
    return ERR_INVALID_TYPE;
  }
  const Descriptor* descriptor = msg.GetDescriptor();
  const Reflection* reflection = msg.GetReflection();
  if (!descriptor || !reflection) {
    return ERR_INVALID_PB;
  }
  for (rapidjson::Value::ConstMemberIterator iter = root.MemberBegin();
       iter != root.MemberEnd(); ++iter) {
    const char* name = iter->name.GetString();
    const FieldDescriptor* field = descriptor->FindFieldByName(name);
    if (!field) {
      reflection->FindKnownExtensionByName(name);
    }
    if (!field) {
      LOG(ERROR) << "unknown field " << name;
      return ERR_UNKNOWN_FIELD;
    }
    if (field->is_repeated()) {
      if (iter->value.GetType() != rapidjson::kArrayType) {
        return ERR_INVALID_JSON;
      } else {
        for (rapidjson::Value::ConstValueIterator ait = iter->value.Begin();
             ait != iter->value.End(); ++ait) {
          int ret = json2field(*ait, msg, field, base64_encode);
          if (ret != 0) {
            return ret;
          }
        }
      }
    } else {
      int ret = json2field(iter->value, msg, field, base64_encode);
      if (ret != 0) {
        return ret;
      }
    }
  }
  return STATUS_OK;
}

int ProtoJson::json2field(const rapidjson::Value& json, Message& msg,
                          const FieldDescriptor* field,
                          const bool& base64_encode) {
  const Reflection* ref = msg.GetReflection();
  const bool repeated = field->is_repeated();
  switch (field->cpp_type()) {
#define _SET_OR_ADD(sfunc, afunc, value) \
  do {                                   \
    if (repeated)                        \
      ref->afunc(&msg, field, value);    \
    else                                 \
      ref->sfunc(&msg, field, value);    \
  } while (0)

    case FieldDescriptor::CPPTYPE_STRING: {
      if (!json.IsString()) return ERR_INVALID_JSON;
      std::string value = json.GetString();
      if (field->type() == FieldDescriptor::TYPE_BYTES) {
        if (base64_encode) {
          _SET_OR_ADD(SetString, AddString, b64_decode(value));
        } else {
          _SET_OR_ADD(SetString, AddString, value);
        }
      } else {
        _SET_OR_ADD(SetString, AddString, value);
      }
      break;
    }
    case FieldDescriptor::CPPTYPE_INT32: {
      if (!json.IsInt()) return ERR_INVALID_JSON;

      _SET_OR_ADD(SetInt32, AddInt32, json.GetInt());
      break;
    }
    case FieldDescriptor::CPPTYPE_UINT32: {
      if (!json.IsUint()) return ERR_INVALID_JSON;

      _SET_OR_ADD(SetUInt32, AddUInt32, json.GetUint());
      break;
    }
    case FieldDescriptor::CPPTYPE_INT64: {
      if (!json.IsInt64()) return ERR_INVALID_JSON;
      _SET_OR_ADD(SetInt64, AddInt64, json.GetInt64());
      break;
    }
    case FieldDescriptor::CPPTYPE_UINT64: {
      if (!json.IsUint64()) return ERR_INVALID_JSON;
      _SET_OR_ADD(SetUInt64, AddUInt64, json.GetInt64());
      break;
    }
    case FieldDescriptor::CPPTYPE_DOUBLE: {
      if (!json.IsDouble()) return ERR_INVALID_JSON;
      _SET_OR_ADD(SetDouble, AddDouble, json.GetDouble());
      break;
    }
    case FieldDescriptor::CPPTYPE_FLOAT: {
      if (!json.IsDouble()) return ERR_INVALID_JSON;
      _SET_OR_ADD(SetFloat, AddFloat, json.GetDouble());
      break;
    }
    case FieldDescriptor::CPPTYPE_BOOL: {
      if (!json.IsBool()) return ERR_INVALID_JSON;
      _SET_OR_ADD(SetBool, AddBool, json.GetBool());
      break;
    }
    case FieldDescriptor::CPPTYPE_MESSAGE: {
      Message* mf = (repeated) ? ref->AddMessage(&msg, field)
                               : ref->MutableMessage(&msg, field);
      parse_json(json, *mf, base64_encode);
      break;
    }
    case FieldDescriptor::CPPTYPE_ENUM: {
      const EnumDescriptor* ed = field->enum_type();
      const EnumValueDescriptor* ev = 0;
      if (json.GetType() == rapidjson::kNumberType) {
        ev = ed->FindValueByNumber(json.GetInt());
      } else if (json.GetType() == rapidjson::kStringType) {
        ev = ed->FindValueByName(json.GetString());
      } else {
        return ERR_INVALID_JSON;
      }
      if (!ev) {
        return ERR_INVALID_JSON;
      }
      _SET_OR_ADD(SetEnum, AddEnum, ev);
      break;
    }
    default:
      break;
  }
  return STATUS_OK;
}

void ProtoJson::pb2json(const ProtoJson::Message& msg, std::string& str) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  rapidjson::Value::AllocatorType allocator;
  std::shared_ptr<rapidjson::Value> root = parse_message(msg, allocator);
  json2string(root, str);
}

}  // namespace perception_feature
