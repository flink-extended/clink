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

#include "clink/utils/clink_utils.h"

#include <dirent.h>
#include <sys/stat.h>

#include <string>

#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/profiled_allocator.h"

namespace clink {

std::unique_ptr<HostContext> CreateHostContext(
    string_view work_queue_type, tfrt::HostAllocatorType host_allocator_type) {
  auto decoded_diagnostic_handler = [&](const DecodedDiagnostic &diag) {
    TFRT_LOG(FATAL) << "Encountered error while executing, aborting: "
                    << diag.message;
  };
  std::unique_ptr<ConcurrentWorkQueue> work_queue =
      CreateWorkQueue(work_queue_type);

  std::unique_ptr<HostAllocator> host_allocator;
  switch (host_allocator_type) {
    case HostAllocatorType::kMalloc:
      host_allocator = CreateMallocAllocator();
      llvm::outs() << "Choosing malloc.\n";
      break;
    case HostAllocatorType::kTestFixedSizeMalloc:
      host_allocator = tfrt::CreateFixedSizeAllocator();
      llvm::outs() << "Choosing fixed size malloc.\n";
      break;
    case HostAllocatorType::kProfiledMalloc:
      host_allocator = CreateMallocAllocator();
      host_allocator = CreateProfiledAllocator(std::move(host_allocator));
      llvm::outs() << "Choosing profiled allocator based on malloc.\n";
      break;
    case HostAllocatorType::kLeakCheckMalloc:
      host_allocator = CreateMallocAllocator();
      host_allocator = CreateLeakCheckAllocator(std::move(host_allocator));
      llvm::outs() << "Choosing memory leak check allocator.\n";
  }
  llvm::outs().flush();

  auto host_ctx = std::make_unique<HostContext>(decoded_diagnostic_handler,
                                                std::move(host_allocator),
                                                std::move(work_queue));
  RegisterStaticKernels(host_ctx->GetMutableRegistry());
  return host_ctx;
}

std::string getOnlyFileInDirectory(std::string path) {
  std::string result = "";
  struct dirent *entry;
  struct stat st;
  DIR *dir = opendir(path.c_str());

  if (dir == NULL) {
    return "";
  }
  while ((entry = readdir(dir)) != NULL) {
    const std::string full_file_name = tfrt::StrCat(path, "/", entry->d_name);
    if (stat(full_file_name.c_str(), &st) == -1) continue;
    bool is_directory = (st.st_mode & S_IFDIR) != 0;
    if (!is_directory) {
      if (result != "") {
        return "";
      }
      result = std::string(entry->d_name);
    }
  }
  closedir(dir);
  return result;
}

}  // namespace clink
