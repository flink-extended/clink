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

// This file implements MLIR operation functions for the clink library.

#include "clink/kernels/opdefs/clink_kernels.h"

#include "mlir/IR/BuiltinOps.h"

namespace clink {

//===----------------------------------------------------------------------===//
// Clink Dialect
//===----------------------------------------------------------------------===//

ClinkDialect::ClinkDialect(MLIRContext *context)
    : Dialect(/*name=*/"clink", context, TypeID::get<ClinkDialect>()) {
  allowUnknownTypes();
  allowUnknownOperations();

  addOperations<
#define GET_OP_LIST
#include "clink/kernels/opdefs/clink_kernels.cpp.inc"
      >();
}

} // namespace clink

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clink/kernels/opdefs/clink_kernels.cpp.inc"
