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
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_java//java:defs.bzl", "java_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

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
        "lib/linalg/sparse_vector.cc",
        "lib/feature/one_hot_encoder.cc",
        "lib/kernels/clink_kernels.cc",
    ],
    hdrs = [
        "include/clink/linalg/sparse_vector.h",
        "include/clink/feature/one_hot_encoder.h",
        "include/clink/kernels/clink_kernels.h",
    ],
    alwayslink_static_registration_src = "lib/kernels/static_registration.cc",
    visibility = [":friends"],
    deps = [
        "@tf_runtime//:hostcontext",
        ":clink_cc_proto",
        "@com_github_nlohmann_json//:json",
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
    name = "clink_jna",
    srcs = [
        "lib/jna/clink_jna.cc",
    ],
    linkshared = True,
    visibility = [":friends"],
    deps = [
        ":clink_kernels",
        ":clink_utils",
        "@tf_runtime//:hostcontext_alwayslink",
    ],
)

FLINK_VERSION = "1_14_0"
SCALA_VERSION = "2_12"

java_library(
    name = "clink_kernels_java_deps",
    exports = [
        "@maven//:net_java_dev_jna_jna",
        "@maven//:org_apache_flink_flink_core",
        "@maven//:org_apache_flink_flink_connector_files",
        "@maven//:org_apache_flink_flink_streaming_java_%s" % SCALA_VERSION,
        "@maven//:org_apache_flink_flink_shaded_jackson",
        "@maven//:org_apache_flink_flink_table_api_java",
        "@maven//:org_apache_flink_flink_table_api_java_bridge_%s" % SCALA_VERSION,
        "@maven//:org_apache_flink_flink_clients_%s" % SCALA_VERSION,
        "@maven//:org_apache_flink_flink_table_planner_%s" % SCALA_VERSION,
        "@maven//:org_apache_flink_flink_table_runtime_%s" % SCALA_VERSION,
        "@maven//:org_apache_flink_flink_ml_core",
        "@maven//:org_apache_flink_flink_ml_iteration",
        "@maven//:org_apache_flink_flink_ml_lib_%s" % SCALA_VERSION,
        "@maven//:commons_collections_commons_collections_3_2_2",
        "@maven//:org_apache_commons_commons_lang3_3_3_2",
        "@maven//:org_apache_commons_commons_compress",
    ],
)

java_library(
    name = "clink_kernels_java_test_deps",
    exports = [
        "@maven//:org_apache_flink_flink_test_utils_junit_%s" % FLINK_VERSION,
        "@maven//:junit_junit",
    ],
)


java_library(
    name = "clink_kernels_java",
    srcs = glob(["java-lib/src/main/**/*.java"]),
    visibility = [":friends"],
    deps = [
        ":clink_kernels_java_deps",
        ":clink_jna",
        ":clink_java_proto",
    ],
)

java_test(
    name = "clink_kernels_java_test",
    srcs = glob(["java-lib/src/test/**/*.java"]),
    visibility = [":friends"],
    deps = [
        ":clink_kernels_java_deps",
        ":clink_kernels_java_test_deps",
        ":clink_kernels_java",
        ":clink_jna",
        ":clink_java_proto",
    ],
    jvm_flags = [
        "-Djna.library.path=.",
    ],
    test_class = "org.flinkextended.clink.util.AllTestsRunner",
)

java_proto_library( 
    name = "clink_java_proto", 
    deps = [":clink_proto"], 
)

cc_proto_library( 
    name = "clink_cc_proto", 
    deps = [":clink_proto"], 
)

proto_library( 
    name = "clink_proto", 
    srcs = ["include/clink/feature/proto/one_hot_encoder.proto"], 
)
