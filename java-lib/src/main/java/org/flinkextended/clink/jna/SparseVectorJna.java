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

package org.flinkextended.clink.jna;

import org.apache.flink.ml.linalg.SparseVector;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.Structure.FieldOrder;

/**
 * Class that corresponds to struct SparseVectorJNA in C++. It is only used by JNA to transmit data
 * between Java and C++.
 */
@FieldOrder({"n", "indices", "values", "length"})
public class SparseVectorJna extends Structure {
    public static class ByReference extends SparseVectorJna implements Structure.ByReference {}

    public int n;
    public Pointer indices;
    public Pointer values;
    public int length;

    /** Converts this class to {@link SparseVector}. */
    public SparseVector toSparseVector() {
        return new SparseVector(
                n, indices.getIntArray(0, length), values.getDoubleArray(0, length));
    }
}
