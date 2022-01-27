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
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/support/logging.h"

#ifdef __cplusplus
extern "C" {
#endif

// Handles llvm::Error generated in Clink JNA methods. This function prints
// corresponding error message to std::err and set errno to 1, which causes Java
// codes to throw LastErrorException.
#define CLINK_JNA_HANDLE_ERROR(ERR, RET_VALUE)      \
  do {                                              \
    if (auto err = ERR) {                           \
      TFRT_LOG(ERROR) << tfrt::StrCat(err) << "\n"; \
      errno = -1;                                   \
      return RET_VALUE;                             \
    }                                               \
  } while (0);

// Handles tfrt::RCReference<tfrt::ErrorAsyncValue> generated in Clink JNA
// methods. This function prints corresponding error message to std::err and set
// errno to 1, which causes Java codes to throw LastErrorException.
#define CLINK_JNA_HANDLE_ASYNC_ERROR(ERR, RET_VALUE)            \
  do {                                                          \
    if (ERR->IsError()) {                                       \
      TFRT_LOG(ERROR) << tfrt::StrCat(ERR->GetError()) << "\n"; \
      errno = -1;                                               \
      return RET_VALUE;                                         \
    }                                                           \
  } while (0);

namespace {
inline tfrt::HostContext *getJnaHostContext() {
#ifndef NDEBUG
  static tfrt::HostAllocatorType allocator_type =
      tfrt::HostAllocatorType::kLeakCheckMalloc;
#else
  static tfrt::HostAllocatorType allocator_type =
      tfrt::HostAllocatorType::kMalloc;
#endif
  static tfrt::HostContext *jna_host_context =
      clink::CreateHostContext("mstd", allocator_type).release();
  return jna_host_context;
}

inline ExecutionContext &getJnaExecutionContext() {
  static ExecutionContext exec_ctx(
      *tfrt::RequestContextBuilder(getJnaHostContext(), nullptr).build());
  return exec_ctx;
}

}  // namespace

double SquareAdd(double x, double y) {
  ExecutionContext &exec_ctx = getJnaExecutionContext();

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
  SparseVectorJNA(const clink::SparseVector &vector,
                  tfrt::HostContext *host_context);
  ~SparseVectorJNA();

  // Total dimensions of the sparse vector.
  int n;
  int *indices;
  double *values;
  // Length of indices and values array.
  int length;

  tfrt::HostContext *host_;
} SparseVectorJNA;

SparseVectorJNA::SparseVectorJNA(const clink::SparseVector &sparse_vector,
                                 tfrt::HostContext *host_context)
    : host_(host_context) {
  this->n = sparse_vector.size();
  this->length = 0;
  for (int i = 0; i < this->n; i++) {
    if (sparse_vector.get(i).get() != 0.0) {
      this->length++;
    }
  }

  this->indices = host_->Allocate<int>(this->length);
  this->values = host_->Allocate<double>(this->length);
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
  host_->Deallocate(this->indices, this->length);
  host_->Deallocate(this->values, this->length);
}

void SparseVector_delete(SparseVectorJNA *vector) {
  getJnaHostContext()->Destruct(vector);
}

SparseVectorJNA *OneHotEncoderModel_transform(clink::OneHotEncoderModel *model,
                                              const int value,
                                              const int column_index) {
  tfrt::AsyncValueRef<int> value_ref = MakeAvailableAsyncValueRef<int>(value);
  tfrt::AsyncValueRef<int> colum_index_ref =
      MakeAvailableAsyncValueRef<int>(column_index);
  llvm::SmallVector<tfrt::AsyncValue *, 4> inputs{
      value_ref.GetAsyncValue(), colum_index_ref.GetAsyncValue()};

  auto output = model->transform(inputs, getJnaExecutionContext())[0];
  getJnaHostContext()->Await(output);
  CLINK_JNA_HANDLE_ASYNC_ERROR(output, NULL);
  clink::SparseVector &actual_vector = output->get<clink::SparseVector>();

  return getJnaHostContext()->Construct<SparseVectorJNA>(actual_vector,
                                                         getJnaHostContext());
}

clink::OneHotEncoderModel *OneHotEncoderModel_load(const char *path) {
  auto model = clink::OneHotEncoderModel::load(path, getJnaHostContext());
  CLINK_JNA_HANDLE_ERROR(model.takeError(), NULL);
  return model->release();
}

void OneHotEncoderModel_delete(clink::OneHotEncoderModel *model) {
  model->DropRef();
}

#ifdef __cplusplus
}
#endif
