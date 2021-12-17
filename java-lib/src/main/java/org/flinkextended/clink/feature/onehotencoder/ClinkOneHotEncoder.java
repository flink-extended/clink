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

import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoder;
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoderModel;
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoderParams;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;

import java.io.IOException;
import java.util.Map;

public class ClinkOneHotEncoder
        implements Estimator<ClinkOneHotEncoder, ClinkOneHotEncoderModel>,
                OneHotEncoderParams<ClinkOneHotEncoder> {
    private final OneHotEncoder estimator;

    public ClinkOneHotEncoder() {
        this(new OneHotEncoder());
    }

    private ClinkOneHotEncoder(OneHotEncoder estimator) {
        this.estimator = estimator;
    }

    @Override
    public ClinkOneHotEncoderModel fit(Table... inputs) {
        OneHotEncoderModel model = estimator.fit(inputs);
        ClinkOneHotEncoderModel clinkModel = new ClinkOneHotEncoderModel();
        ReadWriteUtils.updateExistingParams(clinkModel, model.getParamMap());
        clinkModel.setModelData(model.getModelData());
        return clinkModel;
    }

    @Override
    public void save(String path) throws IOException {
        estimator.save(path);
    }

    public static ClinkOneHotEncoder load(StreamExecutionEnvironment env, String path)
            throws IOException {
        OneHotEncoder estimator = OneHotEncoder.load(env, path);
        return new ClinkOneHotEncoder(estimator);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return estimator.getParamMap();
    }
}
