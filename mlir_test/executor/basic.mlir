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

func @main(%arg_0: f64) {
  %ch0 = tfrt.new.chain

  %value_1 = clink.square.f64 %arg_0
  %result = clink.square_add.f64 %value_1, %arg_0

  %ch1 = tfrt.print.f64 %result, %ch0
  tfrt.return
}
