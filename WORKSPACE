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

workspace(name = "clink")

local_repository(
    name = "tf_runtime",
    path = "tfrt",
)

load("@tf_runtime//:dependencies.bzl", "tfrt_dependencies")

tfrt_dependencies()

load("@bazel_skylib//lib:versions.bzl", "versions")

versions.check(minimum_bazel_version = "4.0.0")

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure", "llvm_disable_optional_support_deps")

maybe(
    llvm_configure,
    name = "llvm-project",
)

llvm_disable_optional_support_deps()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

load("@rules_cc//cc:repositories.bzl", "rules_cc_toolchains")

rules_cc_toolchains()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_JVM_EXTERNAL_TAG = "4.2"
RULES_JVM_EXTERNAL_SHA = "cd1a77b7b02e8e008439ca76fd34f5b07aecb8c752961f9640dea15e9e5ba1ca"

http_archive(
    name = "rules_jvm_external",
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    sha256 = RULES_JVM_EXTERNAL_SHA,
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG,
)

http_archive(
    name = "com_github_nlohmann_json",
    build_file = "//third_party:json.BUILD", # see below
    sha256 = "4cf0df69731494668bdd6460ed8cb269b68de9c19ad8c27abc24cd72605b2d5b",
    strip_prefix = "json-3.9.1",
    urls = ["https://github.com/nlohmann/json/archive/v3.9.1.tar.gz"],
)

# io_bazel_rules_scala defines scala version.
rules_scala_version = "e7a948ad1948058a7a5ddfbd9d1629d6db839933"
http_archive(
    name = "io_bazel_rules_scala",
    sha256 = "76e1abb8a54f61ada974e6e9af689c59fd9f0518b49be6be7a631ce9fa45f236",
    strip_prefix = "rules_scala-%s" % rules_scala_version,
    type = "zip",
    url = "https://github.com/bazelbuild/rules_scala/archive/%s.zip" % rules_scala_version,
)

load("@rules_jvm_external//:defs.bzl", "maven_install")

load("@io_bazel_rules_scala//:scala_config.bzl", "scala_config")
scala_config(scala_version = "2.12.7")

load("@io_bazel_rules_scala//scala:scala.bzl", "scala_repositories")
scala_repositories()

FLINK_VERSION = "1.14.0"
FLINK_ML_VERSION = "2.0.0"
SCALA_VERSION = "2.12"

maven_install(
    artifacts = [
        "net.java.dev.jna:jna:5.6.0",
        "net.java.dev.jna:jna-platform:5.6.0",
        "org.apache.flink:flink-connector-files:%s" % FLINK_VERSION,
        "org.apache.flink:flink-core:%s" % FLINK_VERSION,
        "org.apache.flink:flink-streaming-java_%s:%s" % (SCALA_VERSION, FLINK_VERSION),
        "org.apache.flink:flink-shaded-jackson:2.12.4-14.0",
        "org.apache.flink:flink-table-api-java:%s" % FLINK_VERSION,
        "org.apache.flink:flink-table-api-java-bridge_%s:%s" % (SCALA_VERSION, FLINK_VERSION),
        "org.apache.flink:flink-clients_%s:%s" % (SCALA_VERSION, FLINK_VERSION),
        "org.apache.flink:flink-table-planner_%s:%s" % (SCALA_VERSION, FLINK_VERSION),
        "org.apache.flink:flink-table-runtime_%s:%s" % (SCALA_VERSION, FLINK_VERSION),
        "org.apache.flink:flink-test-utils-junit:%s" % FLINK_VERSION,
        "org.apache.flink:flink-ml-core:%s" % FLINK_ML_VERSION,
        "org.apache.flink:flink-ml-iteration:%s" % FLINK_ML_VERSION,
        "org.apache.flink:flink-ml-lib_%s:%s" % (SCALA_VERSION, FLINK_ML_VERSION),
        "org.apache.commons:commons-compress:1.21",
        "commons-collections:commons-collections:3.2.2",
        "org.apache.commons:commons-lang3:3.3.2",
        "junit:junit:4.12",
    ],
    repositories = [
        "https://maven.google.com",
        "https://repo1.maven.org/maven2",
        "http://packages.confluent.io/maven",
        "http://mvnrepo.alibaba-inc.com/mvn/repository",
        "https://repository.apache.org/content/repositories/orgapacheflink-1473",
    ],
    override_targets = {
        "org.scala-lang.scala-library": "@io_bazel_rules_scala_scala_library//:io_bazel_rules_scala_scala_library",
        "org.scala-lang.scala-reflect": "@io_bazel_rules_scala_scala_reflect//:io_bazel_rules_scala_scala_reflect",
        "org.scala-lang.scala-compiler": "@io_bazel_rules_scala_scala_compiler//:io_bazel_rules_scala_scala_compiler",
        "org.scala-lang.modules.scala-parser-combinators_2.11": "@io_bazel_rules_scala_scala_parser_combinators//:io_bazel_rules_scala_scala_parser_combinators",
        "org.scala-lang.modules.scala-xml_2.11": "@io_bazel_rules_scala_scala_xml//:io_bazel_rules_scala_scala_xml",
    },
)
