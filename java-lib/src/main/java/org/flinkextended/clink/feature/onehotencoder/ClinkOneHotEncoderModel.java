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
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoderModelData;
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoderParams;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
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
import org.apache.commons.compress.utils.Lists;
import org.apache.commons.lang3.ArrayUtils;
import org.clink.feature.onehotencoder.OneHotEncoderModelDataProto;
import org.flinkextended.clink.jna.ClinkJna;
import org.flinkextended.clink.jna.SparseVectorJna;
import org.flinkextended.clink.util.ByteArrayDecoder;
import org.flinkextended.clink.util.ByteArrayEncoder;
import org.flinkextended.clink.util.ClinkReadWriteUtils;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Wrapper class for Flink ML OneHotEncoderModel which calls equivalent C++ operator to transform.
 */
public class ClinkOneHotEncoderModel
        implements Model<ClinkOneHotEncoderModel>, OneHotEncoderParams<ClinkOneHotEncoderModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private String modelDataPath;
    private Table modelDataTable;

    public ClinkOneHotEncoderModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    /**
     * Sets model data using the given list of tables. Each table could be an unbounded stream of
     * model data changes.
     *
     * <p>This method should not be invoked if the {@link ClinkOneHotEncoderModel} is instantiated
     * through {@link ClinkOneHotEncoderModel#load(StreamExecutionEnvironment, String)}, which have
     * provided model data from persistent storage.
     *
     * @throws IllegalStateException if users attempt to modify the model data of a Model loaded
     *     from filesystem.
     * @param inputs a list of tables
     */
    @Override
    public ClinkOneHotEncoderModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        Preconditions.checkState(
                modelDataPath == null,
                "It is not allowed to modify the model data of a Model loaded from filesystem.");
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    /**
     * Applies the one-hot encoding process on the given input tables and returns the result tables.
     *
     * <p>The {@link ClinkOneHotEncoderModel} object from which this method is invoked must have had
     * its model data saved in persistent storage, i.e., must be instantiated through {@link
     * ClinkOneHotEncoderModel#load(StreamExecutionEnvironment, String)} method.
     *
     * @throws IllegalStateException if this {@link ClinkOneHotEncoderModel} object is not created
     *     through {@link ClinkOneHotEncoderModel#load(StreamExecutionEnvironment, String)} method.
     */
    @Override
    public Table[] transform(Table... inputs) {
        final String[] inputCols = getInputCols();
        final String[] outputCols = getOutputCols();

        Preconditions.checkArgument(getHandleInvalid().equals(HasHandleInvalid.ERROR_INVALID));
        Preconditions.checkArgument(inputs.length == 1);
        Preconditions.checkArgument(inputCols.length == outputCols.length);
        Preconditions.checkState(
                modelDataPath != null,
                "transform() can only be invoked after the model is loaded from a path "
                        + "containing the saved model data. Please use ClinkOneHotEncoderModel::save() "
                        + "to save the model data to a certain path in file system and use "
                        + "ClinkOneHotEncoderModel::load() to instantiate the model from that path.");

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
        DataStream<Row> output =
                input.map(new GenerateOutputsFunction(inputCols, modelDataPath), outputTypeInfo);
        Table outputTable = tEnv.fromDataStream(output);
        return new Table[] {outputTable};
    }

    private static class GenerateOutputsFunction extends RichMapFunction<Row, Row> {
        private final String[] inputCols;
        private final String modelDataPath;
        private Pointer modelPointer;

        private GenerateOutputsFunction(String[] inputCols, String modelDataPath) {
            this.inputCols = inputCols;
            this.modelDataPath = modelDataPath;
        }

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            modelPointer = ClinkJna.INSTANCE.OneHotEncoderModel_load(modelDataPath);
        }

        @Override
        public Row map(Row row) {
            Row resultRow = new Row(inputCols.length);
            for (int i = 0; i < inputCols.length; i++) {
                String inputCol = inputCols[i];
                int inputValue = ((Number) row.getField(inputCol)).intValue();
                SparseVectorJna.ByReference jnaVector =
                        ClinkJna.INSTANCE.OneHotEncoderModel_transform(modelPointer, inputValue, i);
                SparseVector vector = jnaVector.toSparseVector();
                ClinkJna.INSTANCE.SparseVector_delete(jnaVector);
                resultRow.setField(i, vector);
            }
            return Row.join(row, resultRow);
        }

        @Override
        public void close() throws Exception {
            super.close();
            ClinkJna.INSTANCE.OneHotEncoderModel_delete(modelPointer);
            modelPointer = null;
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ClinkReadWriteUtils.saveMetadata(getParamMap(), getClass(), path);

        DataStream<byte[]> modelDataProtoBuf =
                DataStreamUtils.mapPartition(
                        OneHotEncoderModelData.getModelDataStream(getModelData()[0]),
                        new ModelDataAggregateFunction());
        modelDataProtoBuf.getTransformation().setParallelism(1);

        ReadWriteUtils.saveModelData(modelDataProtoBuf, path, new ByteArrayEncoder());
    }

    private static class ModelDataAggregateFunction
            implements MapPartitionFunction<Tuple2<Integer, Integer>, byte[]> {
        @Override
        public void mapPartition(
                Iterable<Tuple2<Integer, Integer>> iterable, Collector<byte[]> collector) {
            List<Tuple2<Integer, Integer>> list = Lists.newArrayList(iterable.iterator());
            list.sort(Comparator.comparingInt(o -> o.f0));

            OneHotEncoderModelDataProto.Builder builder = OneHotEncoderModelDataProto.newBuilder();
            builder.addAllFeatureSizes(list.stream().map(x -> x.f1).collect(Collectors.toList()));

            collector.collect(builder.build().toByteArray());
        }
    }

    public static ClinkOneHotEncoderModel load(StreamExecutionEnvironment env, String path)
            throws IOException {
        ClinkOneHotEncoderModel clinkModel = ReadWriteUtils.loadStageParam(path);

        DataStream<byte[]> modelDataProtobuf =
                ReadWriteUtils.loadModelData(env, path, new ByteArrayDecoder());
        DataStream<Tuple2<Integer, Integer>> modelData =
                modelDataProtobuf.flatMap(new ModelDataFlatMapFunction());

        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        clinkModel.setModelData(tEnv.fromDataStream(modelData));
        clinkModel.modelDataPath = path;

        return clinkModel;
    }

    private static class ModelDataFlatMapFunction
            implements FlatMapFunction<byte[], Tuple2<Integer, Integer>> {
        @Override
        public void flatMap(byte[] bytes, Collector<Tuple2<Integer, Integer>> collector)
                throws Exception {
            OneHotEncoderModelDataProto modelDataProto =
                    OneHotEncoderModelDataProto.parseFrom(bytes);
            for (int i = 0; i < modelDataProto.getFeatureSizesCount(); i++) {
                collector.collect(new Tuple2<>(i, modelDataProto.getFeatureSizes(i)));
            }
        }
    }
}
