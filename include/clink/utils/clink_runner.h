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

#ifndef CLINK_UTILS_CLINK_RUNNER_H_
#define CLINK_UTILS_CLINK_RUNNER_H_

#include "mlir/IR/MLIRContext.h"
#include "tfrt/bef/bef_buffer.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/host_context/execution_context.h"

using namespace tfrt;

namespace clink {

// This class is a utility class that provides support for users to specify an
// MLIR function, supply inputs and then have it compiled and run through TFRT.
class ClinkRunner {
public:
  class Builder {
  public:
    Builder();

    // Sets the MLIR function string and returns the object to chain setters.
    // Does not perform validation, will be validated when Compile is called.
    Builder &set_mlir_input(string_view mlir_input) {
      assert(!mlir_input.empty() && "MLIR input must not be empty.");
      mlir_input_ = mlir_input.str();
      return *this;
    }

    // Sets the MLIR function name that will be compiled and run, returns the
    // object to chain setters.
    Builder &set_mlir_fn_name(string_view fn_name) {
      assert(!fn_name.empty() && "Function name must not be empty.");
      fn_name_ = fn_name.str();
      return *this;
    }

    // Sets the `host_context_` that should be used for opening the BefFile.
    // `host_context` must outlive ClinkRunner.
    Builder &set_host_context(HostContext *host_context) {
      assert(host_context && "HostContext must not be null.");
      host_context_ = host_context;
      return *this;
    }

    // Sets the `mlir_context` that should be used for compiling the MLIR code.
    // `mlir_context` must outlive ClinkRunner.
    Builder &set_mlir_context(mlir::MLIRContext *mlir_context) {
      assert(mlir_context && "MLIR context must not be null.");
      mlir_context_ = mlir_context;
      return *this;
    }

    // Compiles the MLIR function to BEF and returns a ClinkRunner
    // object that can be used to Run the MLIR function of interest on TFRT and
    // extract outputs. Assert fails if any of mlir_input, fn_name, mlir_context
    // are not set.
    ClinkRunner Compile();

  private:
    std::string mlir_input_;
    std::string fn_name_;
    mlir::MLIRContext *mlir_context_ = nullptr;
    HostContext *host_context_ = nullptr;
  };

  // Runs the MLIR function on TFRT and returns the outputs.
  llvm::SmallVector<RCReference<AsyncValue>>
  Run(llvm::ArrayRef<RCReference<AsyncValue>> inputs);

private:
  // Use ClinkRunner::Builder to get a ClinkRunner object.
  ClinkRunner(const std::string &fn_name, BefBuffer bef_buffer,
              HostContext *host_context);

  std::string fn_name_;
  BefBuffer bef_buffer_;
  HostContext *host_context_ = nullptr;
  RCReference<tfrt::BEFFile> bef_file_;
  const tfrt::Function *func_;
  std::unique_ptr<tfrt::ResourceContext> resource_context_ = nullptr;
  ExecutionContext execution_context_;
};

} // namespace clink

#endif // CLINK_UTILS_CLINK_RUNNER_H_
