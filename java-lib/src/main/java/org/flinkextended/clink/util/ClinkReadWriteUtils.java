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

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** Utility methods for reading and writing clink operators. */
public class ClinkReadWriteUtils {
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * Saves the metadata of the given stage and the extra metadata to a file named `metadata` under
     * the given path. The metadata of a stage includes the stage class name, parameter values etc.
     *
     * <p>Required: the metadata file under the given path should not exist.
     *
     * @param paramMap The parameter map of the stage.
     * @param modelClass The detailed class of the stage.
     * @param path The parent directory to save the stage metadata.
     */
    public static void saveMetadata(
            Map<Param<?>, Object> paramMap, Class<?> modelClass, String path) throws IOException {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("className", modelClass.getName());
        metadata.put("timestamp", System.currentTimeMillis());
        metadata.put("paramMap", jsonEncode(paramMap));
        String metadataStr = OBJECT_MAPPER.writeValueAsString(metadata);

        // Creates parent directories if not already created.
        new File(path).mkdirs();

        File metadataFile = new File(path, "metadata");
        if (!metadataFile.createNewFile()) {
            throw new IOException("File " + metadataFile.toString() + " already exists.");
        }
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(metadataFile))) {
            writer.write(metadataStr);
        }
    }

    /** Converts a parameter map to corresponding json string. */
    public static Map<String, String> jsonEncode(Map<Param<?>, Object> paramMap)
            throws IOException {
        Map<String, String> result = new HashMap<>(paramMap.size());
        for (Map.Entry<Param<?>, Object> entry : paramMap.entrySet()) {
            String json = jsonEncodeHelper(entry.getKey(), entry.getValue());
            result.put(entry.getKey().name, json);
        }
        return result;
    }

    // A helper method that calls encodes the given parameter value to a json string. We can not
    // call param.jsonEncode(value) directly because Param::jsonEncode(...) needs the actual type
    // of the value.
    private static <T> String jsonEncodeHelper(Param<T> param, Object value) throws IOException {
        return param.jsonEncode((T) value);
    }
}
