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

#include "clink/kernels/opdefs/clink_kernels.h"
#include "clink/utils/clink_runner.h"
#include "clink/utils/clink_utils.h"
#include "mlir/Support/FileUtilities.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"
#include "tfrt/bef_executor_driver/bef_executor_driver.h"
#include "tfrt/host_context/host_context.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::list<std::string> cl_functions( // NOLINT
    "functions", llvm::cl::desc("Specify MLIR functions to run"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

// Enable ConcurrentWorkQueue types to be specified on the command line.
static llvm::cl::opt<std::string> cl_work_queue_type( // NOLINT
    "work_queue_type",
    llvm::cl::desc("Specify concurrent work queue type (s, mstd, ...):"),
    llvm::cl::init("s"));

// Enable HostAllocator types to be specified on the command line.
static llvm::cl::opt<tfrt::HostAllocatorType> cl_host_allocator_type( // NOLINT
    "host_allocator_type", llvm::cl::desc("Specify host allocator type:"),
    llvm::cl::values(
        clEnumValN(tfrt::HostAllocatorType::kMalloc, "malloc", "Malloc."),
        clEnumValN(tfrt::HostAllocatorType::kTestFixedSizeMalloc,
                   "test_fixed_size_1k",
                   "Fixed size (1 kB) Malloc for testing."),
        clEnumValN(tfrt::HostAllocatorType::kProfiledMalloc,
                   "profiled_allocator", "Malloc with metric profiling."),
        clEnumValN(tfrt::HostAllocatorType::kLeakCheckMalloc,
                   "leak_check_allocator", "Malloc with memory leak check.")),
    llvm::cl::init(tfrt::HostAllocatorType::kLeakCheckMalloc));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR translation driver\n");

  // Reads mlir source program.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  auto mlir_src =
      source_mgr.getMemoryBuffer(source_mgr.getMainFileID())->getBuffer();

  // Initializes MLIR context.
  MLIRContext context;
  context.allowUnregisteredDialects();
  context.printOpOnDiagnostic(true);
  mlir::DialectRegistry registry;
  registry.insert<clink::ClinkDialect>();
  registry.insert<tfrt::compiler::TFRTDialect>();
  context.appendDialectRegistry(registry);

  // Initializes HostContext.
  std::unique_ptr<HostContext> host_context =
      clink::CreateHostContext(cl_work_queue_type, cl_host_allocator_type);

  // Initializes ClinkRunner.
  clink::ClinkRunner::Builder builder;
  builder.set_mlir_fn_name("main")
      .set_mlir_input(mlir_src.data())
      .set_host_context(host_context.get())
      .set_mlir_context(&context);
  auto runner = builder.Compile();

  // Executes ClinkRunner.
  llvm::SmallVector<RCReference<AsyncValue>, 4> inputs;
  inputs.push_back(tfrt::MakeAvailableAsyncValueRef<double>(2.0));
  auto results = runner.Run(inputs);

  return 0;
}
