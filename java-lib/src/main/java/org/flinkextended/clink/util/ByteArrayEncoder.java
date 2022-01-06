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

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;

import java.io.IOException;
import java.io.OutputStream;

/** Data Encoder for byte array. */
public class ByteArrayEncoder implements Encoder<byte[]> {
    @Override
    public void encode(byte[] bytes, OutputStream outputStream) throws IOException {
        DataOutputViewStreamWrapper outputViewStreamWrapper =
                new DataOutputViewStreamWrapper(outputStream);
        outputViewStreamWrapper.writeInt(bytes.length);
        outputViewStreamWrapper.write(bytes);
        outputStream.flush();
    }
}
