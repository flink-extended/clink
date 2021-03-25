package com.feature.stageparser;

import com.alibaba.alink.pipeline.PipelineStageBase;
import com.alibaba.alink.pipeline.dataproc.vector.VectorAssembler;
import com.alibaba.fastjson.JSONObject;
import com.feature.protoparser.BaseOperatorBuilder;
import com.feature.protoparser.OperationBuilder;
import com.feature.protoparser.ToVectorBuilder;
import com.google.common.base.Charsets;
import com.google.protobuf.ByteString;
import com.googlecode.protobuf.format.JsonFormat;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;
import perception_feature.proto.Common;
import perception_feature.proto.Datasource;

import java.util.Base64;
import java.util.HashMap;
import java.util.List;

public class VectorAssemblerParser extends BaseStageParser {
    @Override
    public PipelineStageBase parseJsonToPipelineStage(JSONObject obj) {
        String params = obj.get("params").toString();
        String extraParams = obj.getOrDefault("extraParams", new HashMap<>()).toString();
        Params opParams = new Params().fromJson(params);
        opParams.merge(new Params().fromJson(extraParams));
        return new VectorAssembler(opParams);
    }

    @Override
    public String serializeModelToJson(Tuple3<PipelineStageBase<?>, TableSchema, List<Row>> t3) {
        Params params = (t3.f0).getParams();

        String inputCol = params.getStringArray("selectedCols")[0];
        String outputCol = params.getString("outputCol");
        Integer vectorSize = params.getInteger("vecSize");
        Common.RecordEntry.Builder vecSizeParam = Common.RecordEntry.newBuilder();
        vecSizeParam.setKey(String.format("%s_size", inputCol));
        vecSizeParam.setValue(
                Common.Record.newBuilder()
                        .setIntList(Common.IntList.newBuilder().addValue(vectorSize)));
        String innerDeli = params.getStringOrDefault("innerDeli", "|");
        Common.RecordEntry.Builder innerDeliParam = Common.RecordEntry.newBuilder();
        innerDeliParam.setKey(String.format("%s_deli", inputCol));
        innerDeliParam.setValue(
                Common.Record.newBuilder()
                        .setBytesList(
                                Common.BytesList.newBuilder()
                                        .addValue(
                                                ByteString.copyFromUtf8(
                                                        Base64.getEncoder()
                                                                .encodeToString(
                                                                        innerDeli.getBytes(
                                                                                Charsets
                                                                                        .UTF_8))))));
        String valType =
                params.getStringOrDefault(
                        "valType", Datasource.FeatureDataType.forNumber(3).toString());
        Common.RecordEntry.Builder valTypeParam = Common.RecordEntry.newBuilder();
        valTypeParam.setKey(String.format("%s_type", inputCol));
        valTypeParam.setValue(
                Common.Record.newBuilder()
                        .setBytesList(
                                Common.BytesList.newBuilder()
                                        .addValue(
                                                ByteString.copyFromUtf8(
                                                        Base64.getEncoder()
                                                                .encodeToString(
                                                                        valType.getBytes(
                                                                                Charsets
                                                                                        .UTF_8))))));
        OperationBuilder operation = new OperationBuilder(outputCol, vectorSize, 3);
        operation.addInputFeatures(inputCol);
        BaseOperatorBuilder operator = new ToVectorBuilder(inputCol);

        operator.addParam(vecSizeParam.build());
        operator.addParam(innerDeliParam.build());
        operator.addParam(valTypeParam.build());
        operation.addOperator(operator.getBuiltOperator());

        return new JsonFormat().printToString(operation.getBuiltOperation());
    }
}
