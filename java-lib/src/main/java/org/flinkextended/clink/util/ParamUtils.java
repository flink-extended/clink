/*
 * Copyright 2021 The Clink Authors
 *
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

package org.flinkextended.clink.util;

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ReadWriteUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** Utility methods for operator parameters. */
public class ParamUtils {
    /** Converts a parameter map to corresponding json string. */
    public static String jsonEncode(Map<Param<?>, Object> paramMap) throws IOException {
        Map<String, String> result = new HashMap<>(paramMap.size());
        for (Map.Entry<Param<?>, Object> entry : paramMap.entrySet()) {
            String json = jsonEncodeHelper(entry.getKey(), entry.getValue());
            result.put(entry.getKey().name, json);
        }
        return ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(result);
    }

    // A helper method that calls encodes the given parameter value to a json string. We can not
    // call param.jsonEncode(value) directly because Param::jsonEncode(...) needs the actual type
    // of the value.
    private static <T> String jsonEncodeHelper(Param<T> param, Object value) throws IOException {
        return param.jsonEncode((T) value);
    }
}
