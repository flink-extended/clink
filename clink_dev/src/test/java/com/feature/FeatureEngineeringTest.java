package com.feature;

import com.alibaba.alink.common.io.filesystem.FilePath;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.operator.stream.StreamOperator;
import com.alibaba.alink.operator.stream.source.CsvSourceStreamOp;
import com.alibaba.alink.pipeline.Pipeline;
import com.alibaba.alink.pipeline.PipelineModel;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.junit.jupiter.api.Test;

import static com.feature.FeatureEngineeringUtils.parseJsonToPipelineStage;

class FeatureEngineeringTest {
    private static final String[] COL_NAMES = new String[] {
            "id", "click", "dt", "C1", "banner_pos",
            "site_id", "site_domain", "site_category", "app_id", "app_domain",
            "app_category", "device_id", "device_ip", "device_model", "device_type",
            "device_conn_type", "C14", "C15", "C16", "C17",
            "C18", "C19", "C20", "C21"
    };

    private static final String[] COL_TYPES = new String[] {
            "string", "int", "string", "string", "int",
            "string", "string", "string", "string", "string",
            "string", "string", "string", "string", "string",
            "string", "int", "int", "int", "int",
            "int", "int", "int", "int"
    };



    @Test
    public void feature() throws Exception {
        String dataDir = getClass().getResource("/").getPath() + "/feature/data.csv";
        String confDir = getClass().getResource("/").getPath() + "/feature/feature.json";

        BatchOperator.setParallelism(1);
        StringBuilder sbd = new StringBuilder();
        for (int i = 0; i < COL_NAMES.length; i++) {
            if (i != 0) {
                sbd.append(",");
            }
            sbd.append(COL_NAMES[i]).append(" ").append(COL_TYPES[i]);
        }

        BatchOperator data = new CsvSourceBatchOp()
                .setIgnoreFirstLine(true)
                .setFilePath(dataDir)
                .setSchemaStr(sbd.toString());

        StreamOperator sdata = new CsvSourceStreamOp()
                .setIgnoreFirstLine(true)
                .setFilePath(dataDir)
                .setSchemaStr(sbd.toString());

        Object obj = JSON.parse(FeatureEngineeringUtils.getJsonFromFilePath(new FilePath(confDir)));

        Object ops = ((JSONObject) obj).get("operations");
        JSONArray opArr = (JSONArray) ops;

        Pipeline pipeline = new Pipeline();

        for (int i = 0; i < opArr.size(); i++) {
            pipeline.add(parseJsonToPipelineStage((JSONObject) opArr.get(i)));
        }

        PipelineModel model = pipeline.fit(data);

        model.transform(data).print();
        model.transform(sdata).print();
        StreamOperator.execute();
    }
}