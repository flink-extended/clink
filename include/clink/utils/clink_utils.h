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

#ifndef CLINK_UTILS_CLINK_UTILS_H_
#define CLINK_UTILS_CLINK_UTILS_H_

#include "tfrt/bef_executor_driver/bef_executor_driver.h"

using namespace tfrt;

namespace clink {

std::unique_ptr<tfrt::HostContext>
CreateHostContext(string_view work_queue_type,
                  tfrt::HostAllocatorType host_allocator_type);

} // namespace clink

#endif // CLINK_UTILS_CLINK_UTILS_H_
