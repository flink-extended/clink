package com.feature.stageparser;

import com.alibaba.alink.pipeline.PipelineStageBase;
import com.alibaba.alink.pipeline.dataproc.StandardScalerModel;
import com.alibaba.alink.pipeline.dataproc.vector.VectorAssembler;
import com.alibaba.alink.pipeline.feature.OneHotEncoderModel;
import com.alibaba.alink.pipeline.feature.QuantileDiscretizerModel;
import com.alibaba.alink.pipeline.sql.Select;
import com.alibaba.fastjson.JSONObject;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;
import perception_feature.proto.Operations;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class BaseStageParser implements Serializable {

    /** This structure maps a operationName to pipelineStageParser. */
    public static final Map<String, BaseStageParser> STAGE_PARSERS = new HashMap<>();

    static {
        STAGE_PARSERS.put("BUCKET", new BucketParser());
        STAGE_PARSERS.put(QuantileDiscretizerModel.class.toString(), new BucketParser());
        STAGE_PARSERS.put("STANDARDIZE", new StandardParser());
        STAGE_PARSERS.put(StandardScalerModel.class.toString(), new StandardParser());
        STAGE_PARSERS.put("ONEHOTENCODER", new OneHotEncoderParser());
        STAGE_PARSERS.put(OneHotEncoderModel.class.toString(), new OneHotEncoderParser());
        STAGE_PARSERS.put("VECTORASSEMBLER", new VectorAssemblerParser());
        STAGE_PARSERS.put(VectorAssembler.class.toString(), new VectorAssemblerParser());
        STAGE_PARSERS.put("UDF", new UdfParser());
        STAGE_PARSERS.put(Select.class.toString(), new UdfParser());
    }

    /**
     * Parse json object to pipeline stage.
     *
     * @param obj the json object of operation.
     * @return pipeline stage of operation.
     */
    public abstract PipelineStageBase parseJsonToPipelineStage(JSONObject obj);

    /**
     * Serialize model to a special format for c/c++ transformation.
     *
     * @param t3 pipeline info and model.
     * @return serialize result.
     */
    public abstract String serializeModelToJson(
            Tuple3<PipelineStageBase<?>, TableSchema, List<Row>> t3);
}
