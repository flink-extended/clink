// Copyright 2021 The Clink Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- clink_kernels.td ----------------------------------------------------===//
//
// Operation definitions for clink_kernels.
//
//===----------------------------------------------------------------------===//

#ifdef CLINK_OPS
#else
#define CLINK_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// "clink" dialect
def Clink_Dialect : Dialect {
  let name = "clink";

  let description = [{
    This dialect contains common clink operations.
  }];

  let cppNamespace = "::clink";
}

// Base class for Clink dialect ops.
class Clink_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Clink_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {

  // Each registered op in the Clink namespace needs to provide a parser.
  let parser = [{ return clink::parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Clink ops
//===----------------------------------------------------------------------===//

def SquareAddF64Op: Clink_Op<"square_add.f64"> {
  let summary = "clink.square_add.f64 operation";
  let description = [{
     An operation that takes two inputs and returns their squared sum as the result.

     Example:
       %2 = clink.square_add.f64 %0, %1
  }];
  let arguments = (ins F64, F64);
  let results = (outs F64);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def SquareF64Op: Clink_Op<"square.f64"> {
  let summary = "clink.square.f64 operation";
  let description = [{
     An operation that returns the square of the input.

     Example:
       %1 = clink.square.f64 %0
  }];
  let arguments = (ins F64);
  let results = (outs F64);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

#endif  // CLINK_OPS
