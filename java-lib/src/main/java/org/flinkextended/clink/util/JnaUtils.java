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

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

/** Utility methods to be used when Java invokes C++ logic through JNA. */
public class JnaUtils {
    /**
     * Allocates a certain chunk of memory containing values in a byte array, and returns a pointer
     * to that memory chunk.
     */
    public static Pointer getByteArrayPointer(byte[] bytes) {
        Pointer pointer = new Memory((long) bytes.length * Native.getNativeSize(Byte.TYPE));
        pointer.write(0, bytes, 0, bytes.length);
        return pointer;
    }
}
