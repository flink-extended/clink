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

#ifndef CLINK_KERNELS_CLINK_KERNELS_H_
#define CLINK_KERNELS_CLINK_KERNELS_H_

#include "tfrt/host_context/kernel_utils.h"

using namespace tfrt;

namespace clink {

AsyncValueRef<double> SquareAdd(Argument<double> x, Argument<double> y,
                                const ExecutionContext &exec_ctx);

double Square(double x);

void RegisterClinkKernels(tfrt::KernelRegistry *registry);

}  // namespace clink

#endif  // CLINK_KERNELS_CLINK_KERNELS_H_
