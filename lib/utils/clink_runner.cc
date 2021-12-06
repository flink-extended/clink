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

#include "clink/utils/clink_runner.h"

#include "mlir/Parser.h"
#include "tfrt/bef_converter/mlir_to_bef.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_context.h"

namespace clink {

ClinkRunner::Builder::Builder() {}

ClinkRunner ClinkRunner::Builder::Compile() {
  assert(!mlir_input_.empty() &&
         "mlir_input must be set before calling Compile.");
  assert(!fn_name_.empty() && "fn_name must be set before calling Compile.");
  assert(mlir_context_ && "MLIR context must be set before calling Compile.");

  mlir::OwningModuleRef module =
      mlir::parseSourceString(mlir_input_, mlir_context_);

  tfrt::BefBuffer bef_buffer =
      tfrt::ConvertMLIRToBEF(module.get(), /*disable_optional_sections=*/true);
  auto bef_file =
      BEFFile::Open(bef_buffer, host_context_->GetKernelRegistry(),
                    host_context_->diag_handler(), host_context_->allocator());
  return ClinkRunner(fn_name_, std::move(bef_buffer), host_context_);
}

ClinkRunner::ClinkRunner(const std::string &fn_name, BefBuffer bef_buffer,
                         HostContext *host_context)
    : fn_name_(fn_name), bef_buffer_(bef_buffer), host_context_(host_context),
      execution_context_(
          *tfrt::RequestContextBuilder(host_context_, resource_context_.get())
               .build()) {
  bef_file_ =
      BEFFile::Open(bef_buffer_, host_context_->GetKernelRegistry(),
                    host_context_->diag_handler(), host_context_->allocator());
  func_ = bef_file_->GetFunction(fn_name_);
}

llvm::SmallVector<RCReference<AsyncValue>>
ClinkRunner::Run(ArrayRef<RCReference<AsyncValue>> inputs) {
  assert((func_->num_arguments() == inputs.size()) &&
         "Incorrect number of arguments set.");

  std::vector<tfrt::AsyncValue *> input_ptrs;
  input_ptrs.resize(inputs.size());
  std::transform(inputs.begin(), inputs.end(), input_ptrs.begin(),
                 [](auto &value) { return value.get(); });

  llvm::SmallVector<RCReference<AsyncValue>, 4> results;
  results.resize(func_->result_types().size());
  func_->Execute(execution_context_, input_ptrs, results);
  return std::move(results);
}

} // namespace clink
