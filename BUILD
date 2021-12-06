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

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_binary", "tfrt_cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])

package_group(
    name = "friends",
    packages = [
        "//...",
    ],
)

tfrt_cc_binary(
    name = "executor",
    srcs = [
        "lib/executor/main.cc",
    ],
    deps = [
        ":clink_kernels_alwayslink",
        ":clink_kernels_opdefs",
        ":clink_utils",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:basic_kernels_alwayslink",
    ],
)

tfrt_cc_library(
    name = "clink_utils",
    srcs = [
        "lib/utils/clink_runner.cc",
        "lib/utils/clink_utils.cc",
    ],
    hdrs = [
        "include/clink/utils/clink_runner.h",
        "include/clink/utils/clink_utils.h",
    ],
    visibility = [":friends"],
    deps = [
        "@llvm-project//llvm:Support",
        "@tf_runtime//:bef_executor_driver",
        "@tf_runtime//:mlirtobef",
    ],
)

tfrt_cc_library(
    name = "clink_kernels",
    srcs = [
        "lib/kernels/clink_kernels.cc",
    ],
    hdrs = [
        "include/clink/kernels/clink_kernels.h",
    ],
    alwayslink_static_registration_src = "lib/kernels/static_registration.cc",
    visibility = [":friends"],
    deps = [
        "@tf_runtime//:hostcontext",
    ],
)

gentbl_cc_library(
    name = "clink_kernels_opdefs_inc_gen",
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/clink/kernels/opdefs/clink_kernels.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/clink/kernels/opdefs/clink_kernels.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/clink/kernels/opdefs/clink_kernels.td",
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
)

tfrt_cc_library(
    name = "clink_kernels_opdefs",
    srcs = [
        "lib/kernels/opdefs/clink_kernels.cc",
    ],
    hdrs = [
        "include/clink/kernels/opdefs/clink_kernels.h",
    ],
    visibility = [":friends"],
    deps = [
        ":clink_kernels_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)

tfrt_cc_binary(
    name = "clink_kernels_jna",
    srcs = [
        "lib/kernels/clink_kernels_jna.cc",
    ],
    linkshared = True,
    visibility = [":friends"],
    deps = [
        ":clink_kernels",
        ":clink_utils",
        "@tf_runtime//:hostcontext_alwayslink",
    ],
)

java_binary(
    name = "example",
    srcs = [
        "java-lib/src/main/java/org/clink/example/Main.java",
    ],
    visibility = [":friends"],
    deps = [
        "@maven//:net_java_dev_jna_jna",
        ":clink_kernels_jna",
    ],
    jvm_flags = [
        "-Djna.library.path=.",
    ],
    main_class = "org.clink.example.Main",
)


