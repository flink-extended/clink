# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library", "tfrt_cc_test")

licenses(["notice"])

package(
    default_visibility = [":__subpackages__"],
)

tfrt_cc_library(
    name = "common",
    testonly = True,
    hdrs = [
        "include/clink/cpp_tests/test_util.h",
    ],
    visibility = [
        "//visibility:public",
    ],
)

tfrt_cc_test(
    name = "linalg/sparse_vector_test",
    srcs = ["linalg/sparse_vector_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@clink//:clink_kernels",
    ],
)

tfrt_cc_test(
    name = "feature/one_hot_encoder_test",
    srcs = ["feature/one_hot_encoder_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@clink//:clink_kernels",
    ],
)
