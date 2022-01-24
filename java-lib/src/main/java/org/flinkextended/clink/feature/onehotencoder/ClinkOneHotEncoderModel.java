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

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoderModelData;
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoderParams;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.table.runtime.typeutils.ExternalTypeInfo;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import com.sun.jna.Pointer;
import org.apache.commons.lang3.ArrayUtils;
import org.flinkextended.clink.jna.ClinkJna;
import org.flinkextended.clink.jna.SparseVectorJna;
import org.flinkextended.clink.util.ByteArrayDecoder;
import org.flinkextended.clink.util.ByteArrayEncoder;
import org.flinkextended.clink.util.JnaUtils;
import org.flinkextended.clink.util.ParamUtils;

import java.io.IOException;
import java.util.*;
import java.util.function.Function;

import static org.apache.flink.ml.util.ParamUtils.initializeMapWithDefaultValues;

/**
 * Wrapper class for Flink ML OneHotEncoderModel which calls equivalent C++ operator to transform.
 */
public class ClinkOneHotEncoderModel
        implements Model<ClinkOneHotEncoderModel>, OneHotEncoderParams<ClinkOneHotEncoderModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public ClinkOneHotEncoderModel() {
        initializeMapWithDefaultValues(this.paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        final String[] inputCols = getInputCols();
        final String[] outputCols = getOutputCols();
        final String broadcastModelKey = "OneHotModelStream";
        final Table modelDataTable = getModelData()[0];

        Preconditions.checkArgument(getHandleInvalid().equals(HasHandleInvalid.ERROR_INVALID));
        Preconditions.checkArgument(inputs.length == 1);
        Preconditions.checkArgument(inputCols.length == outputCols.length);

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(),
                                Collections.nCopies(
                                                outputCols.length,
                                                ExternalTypeInfo.of(Vector.class))
                                        .toArray(new TypeInformation[0])),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), outputCols));

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        DataStream<Row> input = tEnv.toDataStream(inputs[0]);
        DataStream<Tuple2<Integer, Integer>> modelStream =
                OneHotEncoderModelData.getModelDataStream(modelDataTable);

        GenerateOutputsFunction mapFunction =
                new GenerateOutputsFunction(getParamMap(), broadcastModelKey, inputCols);

        Function<List<DataStream<?>>, DataStream<Row>> function =
                dataStreams -> {
                    DataStream stream = dataStreams.get(0);
                    return stream.map(mapFunction, outputTypeInfo);
                };

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(input),
                        Collections.singletonMap(broadcastModelKey, modelStream),
                        function);

        Table outputTable = tEnv.fromDataStream(output);

        return new Table[] {outputTable};
    }

    @Override
    public ClinkOneHotEncoderModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    private static class GenerateOutputsFunction extends RichMapFunction<Row, Row> {
        private final Map<Param<?>, Object> paramMap;
        private final String broadcastModelKey;
        private final String[] inputCols;
        private Pointer modelPointer = null;

        private GenerateOutputsFunction(
                Map<Param<?>, Object> paramMap, String broadcastModelKey, String[] inputCols) {
            this.paramMap = new HashMap<>(paramMap);
            this.inputCols = inputCols;
            this.broadcastModelKey = broadcastModelKey;
        }

        @Override
        public Row map(Row row) throws IOException {
            if (modelPointer == null) {
                List<Tuple2<Integer, Integer>> modelDataList =
                        getRuntimeContext().getBroadcastVariable(broadcastModelKey);
                modelPointer = loadCppModel(paramMap, modelDataList);
            }

            Row resultRow = new Row(inputCols.length);
            for (int i = 0; i < inputCols.length; i++) {
                String inputCol = inputCols[i];
                int number = ((Number) row.getField(inputCol)).intValue();
                SparseVectorJna.ByReference jnaVector =
                        ClinkJna.INSTANCE.OneHotEncoderModel_transform(modelPointer, number, i);
                SparseVector vector = jnaVector.toSparseVector();
                ClinkJna.INSTANCE.SparseVector_delete(jnaVector);
                resultRow.setField(i, vector);
            }
            return Row.join(row, resultRow);
        }

        @Override
        public void close() throws Exception {
            super.close();
            if (modelPointer != null) {
                ClinkJna.INSTANCE.OneHotEncoderModel_delete(modelPointer);
                modelPointer = null;
            }
        }
    }

    private static Pointer loadCppModel(
            Map<Param<?>, Object> paramMap, List<Tuple2<Integer, Integer>> modelDataList)
            throws IOException {
        String paramString = ParamUtils.jsonEncode(paramMap);

        byte[] modelDataBytes = OneHotEncoderProtobufUtils.getModelDataByteArray(modelDataList);
        Pointer modelDataPointer = JnaUtils.getByteArrayPointer(modelDataBytes);

        return ClinkJna.INSTANCE.OneHotEncoderModel_loadFromMemory(
                paramString, modelDataPointer, modelDataBytes.length);
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);

        DataStream<byte[]> modelDataProtoBuf =
                DataStreamUtils.mapPartition(
                        OneHotEncoderModelData.getModelDataStream(getModelData()[0]),
                        new GenerateProtobufModelDataByteArrayFunction());

        ReadWriteUtils.saveModelData(modelDataProtoBuf, path, new ByteArrayEncoder());
    }

    private static class GenerateProtobufModelDataByteArrayFunction
            implements MapPartitionFunction<Tuple2<Integer, Integer>, byte[]> {
        @Override
        public void mapPartition(
                Iterable<Tuple2<Integer, Integer>> iterable, Collector<byte[]> collector) {
            collector.collect(OneHotEncoderProtobufUtils.getModelDataByteArray(iterable));
        }
    }

    public static ClinkOneHotEncoderModel load(StreamExecutionEnvironment env, String path)
            throws IOException {
        ClinkOneHotEncoderModel clinkModel =
                (ClinkOneHotEncoderModel) ReadWriteUtils.loadStageParam(path);

        DataStream<byte[]> modelDataProtobuf =
                ReadWriteUtils.loadModelData(env, path, new ByteArrayDecoder());
        DataStream<Tuple2<Integer, Integer>> modelData =
                modelDataProtobuf.flatMap(
                        new FlatMapFunction<byte[], Tuple2<Integer, Integer>>() {
                            @Override
                            public void flatMap(
                                    byte[] bytes, Collector<Tuple2<Integer, Integer>> collector)
                                    throws Exception {
                                for (Tuple2<Integer, Integer> tup2 :
                                        OneHotEncoderProtobufUtils.getModelDataIterable(bytes)) {
                                    collector.collect(tup2);
                                }
                            }
                        });

        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        clinkModel.setModelData(tEnv.fromDataStream(modelData));

        return clinkModel;
    }
}
