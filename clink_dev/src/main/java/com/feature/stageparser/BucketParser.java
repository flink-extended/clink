package com.feature.stageparser;

import com.alibaba.alink.operator.common.feature.QuantileDiscretizerModelDataConverter;
import com.alibaba.alink.pipeline.PipelineStageBase;
import com.alibaba.alink.pipeline.feature.QuantileDiscretizer;
import com.alibaba.fastjson.JSONObject;
import com.feature.protoparser.BaseOperatorBuilder;
import com.feature.protoparser.BucketBuilder;
import com.feature.protoparser.OperationBuilder;
import com.googlecode.protobuf.format.JsonFormat;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;
import perception_feature.proto.Common;

import java.util.Arrays;
import java.util.List;

public class BucketParser extends BaseStageParser {

    @Override
    public PipelineStageBase parseJsonToPipelineStage(JSONObject obj) {
        Params opParams = this.genParams(obj);
        return new QuantileDiscretizer(opParams);
    }

    @Override
    public String serializeModelToJson(Tuple3<PipelineStageBase<?>, TableSchema, List<Row>> t3) {
        QuantileDiscretizerModelDataConverter modelData =
                new QuantileDiscretizerModelDataConverter().load(t3.f2);
        Params params = (t3.f0).getParams();

        String inputCol = params.getStringArray("selectedCols")[0];
        String outputCol = params.getStringArray("outputCols")[0];
        int featureSize =
                modelData.data.get(inputCol).splitsArray.length + 1
                                > params.getInteger("numBuckets")
                        ? params.getInteger("numBuckets") + 1
                        : modelData.data.get(inputCol).splitsArray.length + 2;
        OperationBuilder operation = new OperationBuilder(outputCol, featureSize, 2);
        operation.addInputFeatures(inputCol);
        BaseOperatorBuilder operator = new BucketBuilder(inputCol);
        Common.RecordEntry.Builder bucketBoundries = Common.RecordEntry.newBuilder();
        Common.RecordEntry.Builder indexOnly = Common.RecordEntry.newBuilder();
        Common.DoubleList.Builder boundryValues = Common.DoubleList.newBuilder();
        Arrays.stream(modelData.data.get(inputCol).splitsArray)
                .mapToDouble(i -> i.doubleValue())
                .forEach(boundryValues::addValue);
        bucketBoundries
                .setKey(String.format("%s_bucket_boundaries", inputCol))
                .setValue(Common.Record.newBuilder().setDoubleList(boundryValues));
        Common.BoolList.Builder indexOnlyVal = Common.BoolList.newBuilder();
        indexOnlyVal.addValue(params.getBool("indexOnly"));
        indexOnly
                .setKey(String.format("%s_index_only", inputCol))
                .setValue(Common.Record.newBuilder().setBoolList(indexOnlyVal));
        operator.addParam(bucketBoundries.build());
        operator.addParam(indexOnly.build());
        operation.addOperator(operator.getBuiltOperator());

        return new JsonFormat().printToString(operation.getBuiltOperation());
    }
}
