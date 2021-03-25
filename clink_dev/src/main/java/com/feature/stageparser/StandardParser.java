package com.feature.stageparser;

import com.alibaba.alink.operator.common.dataproc.StandardScalerModelDataConverter;
import com.alibaba.alink.pipeline.PipelineStageBase;
import com.alibaba.alink.pipeline.dataproc.StandardScaler;
import com.alibaba.fastjson.JSONObject;
import com.feature.protoparser.BaseOperatorBuilder;
import com.feature.protoparser.OperationBuilder;
import com.feature.protoparser.StdBuilder;
import com.googlecode.protobuf.format.JsonFormat;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;
import perception_feature.proto.Common;

import java.util.HashMap;
import java.util.List;

public class StandardParser extends BaseStageParser {
    @Override
    public PipelineStageBase parseJsonToPipelineStage(JSONObject obj) {
        String params = obj.get("params").toString();
        String extraParams = obj.getOrDefault("extraParams", new HashMap<>()).toString();
        Params opParams = new Params().fromJson(params);
        opParams.merge(new Params().fromJson(extraParams));
        return new StandardScaler(opParams);
    }

    @Override
    public String serializeModelToJson(Tuple3<PipelineStageBase<?>, TableSchema, List<Row>> t3) {
        Tuple4<Boolean, Boolean, double[], double[]> modelData =
                new StandardScalerModelDataConverter().load(t3.f2);
        Params params = (t3.f0).getParams();

        String inputCol = params.getStringArray("selectedCols")[0];
        String outputCol = params.getStringArray("outputCols")[0];
        OperationBuilder operation = new OperationBuilder(outputCol, 1, 1);
        operation.addInputFeatures(inputCol);

        BaseOperatorBuilder operator = new StdBuilder(inputCol);
        Common.RecordEntry.Builder mean = Common.RecordEntry.newBuilder();
        Common.RecordEntry.Builder std = Common.RecordEntry.newBuilder();

        mean.setKey(String.format("%s_mean", inputCol))
                .setValue(
                        Common.Record.newBuilder()
                                .setDoubleList(
                                        Common.DoubleList.newBuilder().addValue(modelData.f2[0])));
        std.setKey(String.format("%s_std", inputCol))
                .setValue(
                        Common.Record.newBuilder()
                                .setDoubleList(
                                        Common.DoubleList.newBuilder().addValue(modelData.f3[0])));
        operator.addParam(mean.build());
        operator.addParam(std.build());
        operation.addOperator(operator.getBuiltOperator());

        return new JsonFormat().printToString(operation.getBuiltOperation());
    }
}
