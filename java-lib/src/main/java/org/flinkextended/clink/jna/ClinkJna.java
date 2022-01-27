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

import com.sun.jna.LastErrorException;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

/** Utility methods that have implementations in C++. */
public interface ClinkJna extends Library {
    ClinkJna INSTANCE = Native.load("clink_jna", ClinkJna.class);

    double SquareAdd(double x, double y);

    double Square(double x);

    /**
     * Deletes a {@link SparseVectorJna} in order to avoid memory leak.
     *
     * @param vector A reference to the {@link SparseVectorJna} object.
     */
    // TODO: Automatically free C++ objects to avoid memory leak and improve usability.
    void SparseVector_delete(SparseVectorJna.ByReference vector);

    /**
     * Loads a {@link org.apache.flink.ml.feature.onehotencoder.OneHotEncoderModel} C++ operator
     * from given path. The path should be a directory containing params saved in json format and
     * model data saved in protobuf format.
     *
     * @return Pointer to the loaded C++ Operator
     */
    Pointer OneHotEncoderModel_load(String path) throws LastErrorException;

    /**
     * Converts an indexed integer to one-hot-encoded sparse vector, using the {@link
     * org.apache.flink.ml.feature.onehotencoder.OneHotEncoderModel} C++ operator.
     *
     * @param modelPointer Pointer to the OneHotEncoder C++ operator
     * @param value The indexed integer to be converted
     * @param columnIndex The column index which the indexed integer locates
     * @return A one-hot-encoded sparse vector
     */
    // TODO: Compare the performance of using ByReference v.s. ByValue and optimize accordingly.
    SparseVectorJna.ByReference OneHotEncoderModel_transform(
            Pointer modelPointer, int value, int columnIndex) throws LastErrorException;

    /**
     * Deletes a {@link org.apache.flink.ml.feature.onehotencoder.OneHotEncoderModel} C++ operator
     * in order to avoid memory leak.
     *
     * @param modelPointer Pointer to the OneHotEncoder C++ operator
     */
    void OneHotEncoderModel_delete(Pointer modelPointer);
}
