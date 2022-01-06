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

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.util.Preconditions;

import java.io.EOFException;
import java.io.IOException;

/** Data Decoder for byte array. */
public class ByteArrayDecoder extends SimpleStreamFormat<byte[]> {
    @Override
    public Reader<byte[]> createReader(Configuration config, FSDataInputStream inputStream) {
        return new Reader<byte[]>() {
            final DataInputViewStreamWrapper inputViewStreamWrapper =
                    new DataInputViewStreamWrapper(inputStream);

            @Override
            public byte[] read() throws IOException {
                try {
                    int expectedLen = inputViewStreamWrapper.readInt();
                    byte[] bytes = new byte[expectedLen];
                    int actualLen = inputViewStreamWrapper.read(bytes);
                    Preconditions.checkArgument(expectedLen == actualLen);
                    return bytes;
                } catch (EOFException e) {
                    return null;
                }
            }

            @Override
            public void close() throws IOException {
                inputStream.close();
            }
        };
    }

    @Override
    public TypeInformation<byte[]> getProducedType() {
        return (TypeInformation<byte[]>) Types.PRIMITIVE_ARRAY(Types.BYTE);
    }
}
