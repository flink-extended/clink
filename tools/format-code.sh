#!/usr/bin/env bash
################################################################################
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
################################################################################

# This script formats all codes in the Clink repository. It uses clang-format to
# format C++ code, diffplug/spotless to format Java code, and Buildifier to
# format Bazel code.

set -e

version_array=(
    "clang-format"  "11.1.0"    "clang-format --version | cut -d\" \" -f3"
    "bazel"         "4.0.0"     "bazel --version | cut -d\" \" -f2"
    "mvn"           "3.1.0"     "mvn --version | head -n1 | cut -d\" \" -f3"
)

# Checks whether required tools have been installed
for ((i = 0; i < ${#version_array[@]}; i += 3)); do
    cmd=${version_array[$i]}
    if ! command -v $cmd &> /dev/null
    then
        echo "$cmd: command not found"
        exit 1
    fi
    expected_version=${version_array[$i+1]}
    actual_version=`eval "${version_array[$i+2]}"`
    unsorted_versions="${expected_version}\n${actual_version}\n"
    sorted_versions=`printf ${unsorted_versions} | sort -V`
    unsorted_versions=`printf ${unsorted_versions}`
    if [ "${unsorted_versions}" != "${sorted_versions}" ]; then
        echo "$cmd $expected_version or a higher version is required, but found $actual_version"
        exit 1
    fi
done

# Formats C++ codes
find . \( -name "*.cc" -or -name "*.h" \) -not -path "./tfrt/*" -exec clang-format -i {} \;

# Formats Java codes
mvn -f java-lib spotless:apply

# Formats Bazel codes
bazel run //:buildifier
