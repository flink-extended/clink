package com.feature.stageparser;

import com.alibaba.alink.pipeline.PipelineStageBase;
import com.alibaba.alink.pipeline.sql.Select;
import com.alibaba.fastjson.JSONObject;
import com.feature.protoparser.BaseOperatorBuilder;
import com.feature.protoparser.KeepBuilder;
import com.feature.protoparser.OperationBuilder;
import com.googlecode.protobuf.format.JsonFormat;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;

import java.util.List;

public class KeepParser extends BaseStageParser {

    @Override
    public PipelineStageBase parseJsonToPipelineStage(JSONObject obj) {
        Params opParams = this.genParams(obj);
        return new Select(opParams);
    }

    @Override
    public String serializeModelToJson(Tuple3<PipelineStageBase<?>, TableSchema, List<Row>> t3) {
        Params params = (t3.f0).getParams();

        String inputCol = params.getStringArray("selectedCols")[0];
        String outputCol = params.getStringArray("outputCols")[0];
        OperationBuilder operation = new OperationBuilder(outputCol, 1, 1);
        operation.addInputFeatures(inputCol);
        BaseOperatorBuilder operator = new KeepBuilder(inputCol);
        operation.addOperator(operator.getBuiltOperator());

        return new JsonFormat().printToString(operation.getBuiltOperation());
    }
}
