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

# Checks whether required tools have been installed
for cmd in clang-format mvn
do
    if ! command -v $cmd &> /dev/null
    then
        echo "$cmd: command not found"
        exit
    fi
done

# TODO: change clang-format style to google
# Formats C++ codes
find . \( -name "*.cc" -or -name "*.h" \) -not -path "./tfrt/*" -exec clang-format -i -style=llvm {} \;

# Formats Java codes
mvn -f java-lib spotless:apply

# Formats Bazel codes
bazel run //:buildifier
