/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>

#include "clink/feature/one_hot_encoder.h"
#include "clink/kernels/clink_kernels.h"
#include "clink/utils/clink_utils.h"
#include "nlohmann/json.hpp"
#include "tfrt/support/logging.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#ifdef __cplusplus
extern "C" {
#endif

// Clink JNA programs handle exception in Java. C++ part of the programs only
// prints out the information.
#define CLINK_JNA_HANDLE_ERROR(ERR)                                            \
  do {                                                                         \
    if (auto err = ERR) {                                                      \
      llvm::Error unknown = llvm::handleErrors(                                \
          std::move(err), [&](const llvm::StringError &err) {                  \
            TFRT_LOG(INFO) << err.getMessage() << "\n";                        \
          });                                                                  \
      assert(!unknown && "Unknown error type");                                \
      errno = -1;                                                              \
    }                                                                          \
  } while (0);

double SquareAdd(double x, double y) {
  std::unique_ptr<HostContext> host_context =
      clink::CreateHostContext("s", tfrt::HostAllocatorType::kLeakCheckMalloc);
  ExecutionContext exec_ctx(
      *tfrt::RequestContextBuilder(host_context.get(), nullptr).build());

  AsyncValueRef<double> x_async = MakeAvailableAsyncValueRef<double>(x);
  Argument<double> x_arg(x_async.GetAsyncValue());

  AsyncValueRef<double> y_async = MakeAvailableAsyncValueRef<double>(y);
  Argument<double> y_arg(y_async.GetAsyncValue());

  AsyncValueRef<double> result_async = clink::SquareAdd(x_arg, y_arg, exec_ctx);

  exec_ctx.host()->Await(result_async.CopyRCRef());
  return result_async.get();
}

double Square(double x) { return clink::Square(x); }

// Struct representation of clink::SparseVector. It is only used for JNA to
// transmit data between Java and C++.
typedef struct SparseVectorJNA {
  SparseVectorJNA(clink::SparseVector &);
  ~SparseVectorJNA();

  // Total dimensions of the sparse vector.
  int n;
  int *indices;
  double *values;
  // Length of indices and values array.
  int length;
} SparseVectorJNA;

SparseVectorJNA::SparseVectorJNA(clink::SparseVector &sparse_vector) {
  this->n = sparse_vector.size();
  this->length = 0;
  for (int i = 0; i < this->n; i++) {
    if (sparse_vector.get(i).get() != 0.0) {
      this->length++;
    }
  }

  this->indices = new int[this->length];
  this->values = new double[this->length];
  int offset = 0;
  for (int i = 0; i < this->n; i++) {
    if (sparse_vector.get(i).get() != 0.0) {
      this->indices[offset] = i;
      this->values[offset] = sparse_vector.get(i).get();
      offset++;
    }
  }
}

SparseVectorJNA::~SparseVectorJNA() {
  delete[] this->indices;
  this->indices = NULL;
  delete[] this->values;
  this->values = NULL;
}

void SparseVector_delete(SparseVectorJNA *vector) { delete vector; }

SparseVectorJNA *OneHotEncoderModel_transform(clink::OneHotEncoderModel *model,
                                              const int value,
                                              const int columnIndex) {
  auto sparse_vector = model->transform(value, columnIndex);
  CLINK_JNA_HANDLE_ERROR(sparse_vector.takeError())
  return new SparseVectorJNA(sparse_vector.get());
}

clink::OneHotEncoderModel *
OneHotEncoderModel_loadFromMemory(const char *params_str,
                                  const char *model_data_str,
                                  const int model_data_str_len) {
  clink::OneHotEncoderModel *model = new clink::OneHotEncoderModel();

  nlohmann::json params = nlohmann::json::parse(params_str);
  std::string is_droplast = params["dropLast"].get<std::string>();
  model->setDropLast(is_droplast.compare("false"));
  CLINK_JNA_HANDLE_ERROR(
      model->setModelData(std::string(model_data_str, model_data_str_len)))

  return model;
}

void OneHotEncoderModel_delete(const clink::OneHotEncoderModel *model) {
  delete model;
}

#ifdef __cplusplus
}
#endif
