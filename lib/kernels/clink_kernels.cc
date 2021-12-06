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

#include "clink/kernels/clink_kernels.h"

#include "tfrt/host_context/async_dispatch.h"

using namespace tfrt;

namespace clink {

AsyncValueRef<double> SquareAdd(Argument<double> x, Argument<double> y,
                                const ExecutionContext &exec_ctx) {
  HostContext *host = exec_ctx.host();
  // Submit a subtask to compute x^2.
  AsyncValueRef<double> x_square =
      EnqueueWork(exec_ctx, [x = x.ValueRef()] { return x.get() * x.get(); });

  // Submit a subtask to compute y^2.
  AsyncValueRef<double> y_square =
      EnqueueWork(exec_ctx, [y = y.ValueRef()] { return y.get() * y.get(); });

  SmallVector<AsyncValue *, 2> async_values;
  async_values.push_back(x_square.GetAsyncValue());
  async_values.push_back(y_square.GetAsyncValue());

  // Submit a subtask to compute x^2 + y^2 once the previous two subtasks have
  // completed.
  auto output = MakeUnconstructedAsyncValueRef<double>(host);
  RunWhenReady(async_values,
               [x_square = std::move(x_square), y_square = std::move(y_square),
                output = output.CopyRef(), exec_ctx]() {
                 output.emplace(x_square.get() + y_square.get());
               });

  return output;
}

double Square(double x) { return x * x; }

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void RegisterClinkKernels(tfrt::KernelRegistry *registry) {
  registry->AddKernel("clink.square_add.f64", TFRT_KERNEL(SquareAdd));
  registry->AddKernel("clink.square.f64", TFRT_KERNEL(Square));
}

} // namespace clink
