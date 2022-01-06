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

package org.flinkextended.clink.feature.onehotencoder;

import org.apache.flink.api.java.tuple.Tuple2;

import org.apache.commons.compress.utils.Lists;
import org.clink.feature.onehotencoder.OneHotEncoderModelDataProto;

import java.io.IOException;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

/** Utility functions for OneHotEncoder to convert from and to ProtoBuf data. */
public class OneHotEncoderProtobufUtils {
    /** Converts the mode data from Tuple2 iterables to protobuf-formatted byte array. */
    public static byte[] getModelDataByteArray(Iterable<Tuple2<Integer, Integer>> iterable) {
        List<Tuple2<Integer, Integer>> list = Lists.newArrayList(iterable.iterator());
        list.sort(Comparator.comparingInt(o -> o.f0));

        OneHotEncoderModelDataProto.Builder builder = OneHotEncoderModelDataProto.newBuilder();
        builder.addAllFeatureSizes(list.stream().map(x -> x.f1).collect(Collectors.toList()));

        return builder.build().toByteArray();
    }

    /** Converts the model data from protobuf-formatted byte array to Tuple2 iterables. */
    public static Iterable<Tuple2<Integer, Integer>> getModelDataIterable(byte[] bytes)
            throws IOException {
        OneHotEncoderModelDataProto modelDataProto = OneHotEncoderModelDataProto.parseFrom(bytes);
        return () -> new ModelDataIterator(modelDataProto);
    }

    private static class ModelDataIterator implements Iterator<Tuple2<Integer, Integer>> {
        private final OneHotEncoderModelDataProto modelDataProto;
        private int i = 0;

        private ModelDataIterator(OneHotEncoderModelDataProto modelDataProto) {
            this.modelDataProto = modelDataProto;
        }

        @Override
        public boolean hasNext() {
            return i < modelDataProto.getFeatureSizesCount();
        }

        @Override
        public Tuple2<Integer, Integer> next() {
            Tuple2<Integer, Integer> tuple2 = new Tuple2<>(i, modelDataProto.getFeatureSizes(i));
            i++;
            return tuple2;
        }
    }
}
