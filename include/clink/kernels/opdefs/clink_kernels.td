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

include "tfrt/basic_kernels/opdefs/tfrt_base.td"

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
// Clink types
//===----------------------------------------------------------------------===//

def Clink_ModelType :
  Type<CPred<"$_self.isa<clink::ModelType>()">, "!clink.model type">,
  BuildableType<"$_builder.getType<clink::ModelType>()">;

def Clink_VectorType :
  Type<CPred<"$_self.isa<clink::VectorType>()">, "!clink.vector type">,
  BuildableType<"$_builder.getType<clink::VectorType>()">;

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

def OneHotEncoderLoadOp: Clink_Op<"onehotencoder_load"> {
  let summary = "onehotencoder_load operation";
  let description = [{
     An operation that loads a OneHotEncoderModel from a given path. The path should be a directory
     containing params and model data saved through
     org.clink.feature.onehotencoder.ClinkOneHotEnoderModel::save(...).

     Example:
       %1 = clink.onehotencoder_load %0
  }];
  let arguments = (ins TFRT_StringType:$path);
  let results = (outs Clink_ModelType:$model);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

// TODO: Change this to generic kernels for load and transform operations
def OneHotEncoderTransformOp: Clink_Op<"onehotencoder_transform"> {
  let summary = "onehotencoder_transform operation";
  let description = [{
     An operation that transforms data based on a OneHotEncoderModel. It takes a value
     and its index among all columns in a database table that needs to be one-hot
     encoded, and returns the resulting encoded sparse vector.

     Example:
       %3 = clink.onehotencoder_transform %0, %1, %2
  }];
  let arguments = (ins 
    Clink_ModelType:$model,
    I32:$value,
    I32:$column_index
  );
  let results = (outs Clink_VectorType:$vector);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

#endif  // CLINK_OPS
