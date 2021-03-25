package com.feature.stageparser;

import com.alibaba.alink.operator.common.feature.OneHotModelData;
import com.alibaba.alink.operator.common.feature.OneHotModelDataConverter;
import com.alibaba.alink.pipeline.PipelineStageBase;
import com.alibaba.alink.pipeline.feature.OneHotEncoder;
import com.alibaba.fastjson.JSONObject;
import com.feature.protoparser.BaseOperatorBuilder;
import com.feature.protoparser.OneHotBuilder;
import com.feature.protoparser.OperationBuilder;
import com.google.common.base.Charsets;
import com.google.protobuf.ByteString;
import com.googlecode.protobuf.format.JsonFormat;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;
import perception_feature.proto.Common;

import java.util.*;

public class OneHotEncoderParser extends BaseStageParser {

    @Override
    public PipelineStageBase parseJsonToPipelineStage(JSONObject obj) {
        String params = obj.get("params").toString();
        String extraParams = obj.getOrDefault("extraParams", new HashMap<>()).toString();
        Params opParams = new Params().fromJson(params);
        opParams.merge(new Params().fromJson(extraParams));
        return new OneHotEncoder(opParams);
    }

    @Override
    public String serializeModelToJson(Tuple3<PipelineStageBase<?>, TableSchema, List<Row>> t3) {
        OneHotModelData modelData = new OneHotModelDataConverter().load(t3.f2);
        Params params = (t3.f0).getParams();

        String inputCol = params.getStringArray("selectedCols")[0];
        String outputCol = params.getStringArray("outputCols")[0];
        Map<Integer, Long> binMap = modelData.modelData.tokenNumber;
        Boolean indexOnly = params.getBool("indexOnly");
        OperationBuilder operation =
                new OperationBuilder(outputCol, indexOnly ? 1 : binMap.size(), indexOnly ? 1 : 2);
        operation.addInputFeatures(inputCol);
        Common.BytesList.Builder binList = Common.BytesList.newBuilder();
        Object[] tokenArray =
                binMap.entrySet().stream()
                        .sorted((o1, o2) -> o1.getKey() > o2.getKey() ? o1.getKey() : o2.getKey())
                        .map(i -> i.getValue().toString())
                        .toArray();
        Arrays.stream(tokenArray)
                .forEach(
                        i ->
                                binList.addValue(
                                        ByteString.copyFromUtf8(
                                                Base64.getEncoder()
                                                        .encodeToString(
                                                                i.toString()
                                                                        .getBytes(
                                                                                Charsets.UTF_8)))));
        BaseOperatorBuilder operator = new OneHotBuilder(inputCol);
        operator.addParam(
                Common.RecordEntry.newBuilder()
                        .setKey(String.format("%s_bins", inputCol))
                        .setValue(Common.Record.newBuilder().setBytesList(binList))
                        .build());
        operation.addOperator(operator.getBuiltOperator());

        return new JsonFormat().printToString(operation.getBuiltOperation());
    }
}
