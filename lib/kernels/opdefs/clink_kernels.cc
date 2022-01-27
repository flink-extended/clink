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

#include "clink/kernels/opdefs/types.h"
#include "mlir/IR/BuiltinOps.h"
#include "tfrt/basic_kernels/opdefs/types.h"

namespace clink {

//===----------------------------------------------------------------------===//
// Clink Dialect
//===----------------------------------------------------------------------===//

ClinkDialect::ClinkDialect(MLIRContext *context)
    : Dialect(/*name=*/"clink", context, TypeID::get<ClinkDialect>()) {
  allowUnknownTypes();
  allowUnknownOperations();

  addTypes<ModelType, VectorType>();

  addOperations<
#define GET_OP_LIST
#include "clink/kernels/opdefs/clink_kernels.cpp.inc"
      >();
}

mlir::Type ClinkDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (spec == "model") return ModelType::get(getContext());
  if (spec == "vector") return VectorType::get(getContext());

  if (auto type = mlir::Dialect::parseType(parser)) return type;

  mlir::Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  mlir::emitError(loc) << "unknown data type " << spec;
  return {};
}

void ClinkDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter &printer) const {
  if (type.isa<ModelType>()) {
    printer << "model";
    return;
  }

  if (type.isa<VectorType>()) {
    printer << "vector";
    return;
  }

  llvm_unreachable("unknown data type");
}

}  // namespace clink

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clink/kernels/opdefs/clink_kernels.cpp.inc"
