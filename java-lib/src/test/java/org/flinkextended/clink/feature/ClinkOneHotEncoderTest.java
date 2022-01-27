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

package org.flinkextended.clink.feature;

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoderModelData;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import com.sun.jna.LastErrorException;
import org.flinkextended.clink.feature.onehotencoder.ClinkOneHotEncoder;
import org.flinkextended.clink.feature.onehotencoder.ClinkOneHotEncoderModel;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import static org.junit.Assert.assertEquals;

/** Tests Java wrapped C++ OneHotEncoder Estimator and Model Operator. */
public class ClinkOneHotEncoderTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private ClinkOneHotEncoder estimator;
    private String savePath;
    private static final Row[] trainInput = new Row[] {Row.of(0, 1), Row.of(2, 3)};
    private static final Row predictInput = Row.of(0, 1);
    private static final Row expectedOutput =
            Row.of(
                    0,
                    1,
                    Vectors.sparse(2, new int[] {0}, new double[] {1.0}),
                    Vectors.sparse(3, new int[] {1}, new double[] {1.0}));

    @Before
    public void before() throws Exception {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);

        estimator =
                new ClinkOneHotEncoder()
                        .setInputCols("a", "b")
                        .setOutputCols("c", "d")
                        .setDropLast(false);

        savePath = tempFolder.newFolder().getAbsolutePath();
    }

    @Test
    public void testFitAndPredict() throws Exception {
        DataStream<Row> trainStream = env.fromElements(trainInput);
        Table trainTable = tEnv.fromDataStream(trainStream).as("a", "b");
        DataStream<Row> predictStream = env.fromElements(predictInput);
        Table predictTable = tEnv.fromDataStream(predictStream).as("a", "b");

        estimator.fit(trainTable).save(savePath);
        env.execute();
        ClinkOneHotEncoderModel model = ClinkOneHotEncoderModel.load(env, savePath);

        Table outputTable = model.transform(predictTable)[0];

        Row actual = outputTable.execute().collect().next();
        assertEquals(expectedOutput, actual);
    }

    @Test
    public void testInvalidInput() throws Exception {
        DataStream<Row> trainStream = env.fromElements(trainInput);
        Table trainTable = tEnv.fromDataStream(trainStream).as("a", "b");
        DataStream<Row> predictStream = env.fromElements(Row.of(5, 6));
        Table predictTable = tEnv.fromDataStream(predictStream).as("a", "b");

        estimator.fit(trainTable).save(savePath);
        env.execute();
        ClinkOneHotEncoderModel model = ClinkOneHotEncoderModel.load(env, savePath);

        Table outputTable = model.transform(predictTable)[0];

        try {
            outputTable.execute().collect().next();
            Assert.fail("Expected LastErrorException");
        } catch (Exception e) {
            Throwable exception = e;
            while (exception.getCause() != null) {
                exception = exception.getCause();
            }
            assertEquals(LastErrorException.class, exception.getClass());
        }
    }

    @Test
    public void testGetModelData() throws Exception {
        estimator.setInputCols("a").setOutputCols("c");
        DataStream<Row> trainStream = env.fromElements(Row.of(0), Row.of(1), Row.of(2));
        Table trainTable = tEnv.fromDataStream(trainStream).as("a");

        ClinkOneHotEncoderModel model = estimator.fit(trainTable);
        Tuple2<Integer, Integer> expected = new Tuple2<>(0, 2);
        Tuple2<Integer, Integer> actual =
                OneHotEncoderModelData.getModelDataStream(model.getModelData()[0])
                        .executeAndCollect()
                        .next();
        assertEquals(expected, actual);
    }

    @Test
    public void testSetModelData() throws Exception {
        DataStream<Row> trainStream = env.fromElements(trainInput);
        Table trainTable = tEnv.fromDataStream(trainStream).as("a", "b");
        DataStream<Row> predictStream = env.fromElements(predictInput);
        Table predictTable = tEnv.fromDataStream(predictStream).as("a", "b");

        ClinkOneHotEncoderModel modelA = estimator.fit(trainTable);

        Table modelData = modelA.getModelData()[0];
        ClinkOneHotEncoderModel modelB = new ClinkOneHotEncoderModel().setModelData(modelData);
        ReadWriteUtils.updateExistingParams(modelB, modelA.getParamMap());
        modelB.save(savePath);
        env.execute();
        ClinkOneHotEncoderModel modelC = ClinkOneHotEncoderModel.load(env, savePath);

        Table outputTable = modelC.transform(predictTable)[0];

        Row actual = outputTable.execute().collect().next();
        Row expected =
                Row.of(
                        0,
                        1,
                        Vectors.sparse(2, new int[] {0}, new double[] {1.0}),
                        Vectors.sparse(3, new int[] {1}, new double[] {1.0}));
        assertEquals(expected, actual);
    }
}
