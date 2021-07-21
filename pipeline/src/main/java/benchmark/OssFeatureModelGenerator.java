package benchmark;

import com.alibaba.alink.common.AlinkGlobalConfiguration;
import com.alibaba.alink.common.io.filesystem.BaseFileSystem;
import com.alibaba.alink.common.io.filesystem.FilePath;
import com.alibaba.alink.common.io.filesystem.OssFileSystem;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.sink.CsvSinkBatchOp;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.pipeline.Pipeline;
import com.alibaba.alink.pipeline.PipelineModel;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.feature.FeatureEngineeringTransferOp;
import com.feature.FeatureEngineeringUtils;

import static com.feature.FeatureEngineeringUtils.parseJsonToPipelineStage;

public class OssFeatureModelGenerator {
    private static final String[] COL_NAMES = new String[]{
            "id", "click", "dt", "C1", "banner_pos",
            "site_id", "site_domain", "site_category", "app_id", "app_domain",
            "app_category", "device_id", "device_ip", "device_model", "device_type",
            "device_conn_type", "C14", "C15", "C16", "C17",
            "C18", "C19", "C20", "C21"
    };

    private static final String[] COL_TYPES = new String[]{
            "string", "int", "string", "string", "int",
            "string", "string", "string", "string", "string",
            "string", "string", "string", "string", "string",
            "string", "int", "int", "int", "int",
            "int", "int", "int", "int"
    };

    private static String getSchema() {
        StringBuilder sbd = new StringBuilder();
        for (int i = 0; i < COL_NAMES.length; i++) {
            if (i != 0) {
                sbd.append(",");
            }
            sbd.append(COL_NAMES[i]).append(" ").append(COL_TYPES[i]);
        }
        return sbd.toString();
    }

    public static void main(String[] args) throws Exception {
//        String version = args[0];
//        String endpoint = args[1];
//        String bucketName = args[2];
//        String accessId = args[3];
//        String accessKey = args[4];
//        String filePath = args[5];
//        String confPath = args[6];
        String version = "3.4.1";
        String endpoint = "http://oss-cn-hangzhou-zmf.aliyuncs.com/";
        String bucketName = "weibozhao";
        String accessId = "*";
        String accessKey = "*";
        final String pluginDir = "/Users/weibo/.alink_plugins";
        String filePath = "/feature/small.csv";
        String confPath = "/feature/feature.json";

        AlinkGlobalConfiguration.setPluginDir(pluginDir);
        BaseFileSystem<?> fs = new OssFileSystem(version, endpoint, bucketName, accessId, accessKey);

        BatchOperator data = new CsvSourceBatchOp()
                .setIgnoreFirstLine(true)
                .setFilePath(new FilePath(filePath, fs))
                .setSchemaStr(getSchema());

        Object obj = JSON.parse(FeatureEngineeringUtils
                .getJsonFromFilePath(new FilePath(confPath, fs)));
        Object ops = ((JSONObject) obj).get("operations");
        JSONArray opArr = (JSONArray) ops;

        Pipeline pipeline = new Pipeline();

        for (int i = 0; i < opArr.size(); i++) {
            pipeline.add(parseJsonToPipelineStage((JSONObject) opArr.get(i)));
        }
        PipelineModel model = pipeline.fit(data);

        model.save()
                .link(new FeatureEngineeringTransferOp())
                .link(
                        new CsvSinkBatchOp()
                                .setOverwriteSink(true)
                                .setQuoteChar(null)
                                .setFilePath(new FilePath("/feature/model_20210315.json", fs)));
        BatchOperator.execute();

    }
}
