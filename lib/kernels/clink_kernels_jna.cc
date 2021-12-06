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

#include "clink/kernels/clink_kernels.h"
#include "clink/utils/clink_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif
