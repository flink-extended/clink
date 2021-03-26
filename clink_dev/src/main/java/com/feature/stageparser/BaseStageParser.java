package com.feature.stageparser;

import com.alibaba.alink.pipeline.PipelineStageBase;
import com.alibaba.fastjson.JSONObject;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;

public abstract class BaseStageParser implements Serializable {

    public static PipelineStageBase parseJsonToPipelineWithName(JSONObject obj, String parserName)
            throws ClassNotFoundException, IllegalAccessException, InstantiationException {
        Class clazz = Class.forName(String.format("com.feature.stageparser.%s", parserName));
        BaseStageParser thisParser = (BaseStageParser) clazz.newInstance();
        return thisParser.parseJsonToPipelineStage(obj);
    }

    public static String serializeModelToJsonWithName(
            Tuple3<PipelineStageBase<?>, TableSchema, List<Row>> t3, String parserClassName)
            throws ClassNotFoundException, IllegalAccessException, InstantiationException {
        Class clazz = Class.forName(parserClassName);
        BaseStageParser thisParser = (BaseStageParser) clazz.newInstance();
        return thisParser.serializeModelToJson(t3);
    }

    public Params genParams(JSONObject obj) {
        String params = obj.get("params").toString();
        String extraParams = obj.getOrDefault("extraParams", new HashMap<>()).toString();
        Params opParams = new Params().fromJson(params);
        opParams.merge(new Params().fromJson(extraParams));
        opParams.set("parserName", this.getClass().getName());
        return opParams;
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
