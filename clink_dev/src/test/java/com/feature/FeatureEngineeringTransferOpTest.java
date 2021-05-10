package com.feature;

import com.alibaba.alink.common.io.filesystem.FilePath;
import com.alibaba.alink.common.io.filesystem.FlinkFileSystem;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.sink.TextSinkBatchOp;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.pipeline.Pipeline;
import com.alibaba.alink.pipeline.PipelineModel;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.feature.common.FileHandler;
import com.feature.common.S3Handler;
import com.feature.common.TarHandler;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.types.Row;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileWriter;
import java.util.List;

import static com.feature.FeatureEngineeringUtils.parseJsonToPipelineStage;

class FeatureEngineeringTransferOpTest {
    private static final String[] COL_NAMES =
            new String[] {
                "id", "click", "dt", "C1", "banner_pos",
                "site_id", "site_domain", "site_category", "app_id", "app_domain",
                "app_category", "device_id", "device_ip", "device_model", "device_type",
                "device_conn_type", "C14", "C15", "C16", "C17",
                "C18", "C19", "C20", "C21"
            };
    private static final String[] COL_TYPES =
            new String[] {
                "string", "int", "string", "string", "int",
                "string", "string", "string", "string", "string",
                "string", "string", "string", "string", "string",
                "string", "int", "int", "int", "int",
                "int", "int", "int", "int"
            };
    private static String libfgSoPath;

    @Test
    public void transfer() throws Exception {
        libfgSoPath =
                new File("").getAbsolutePath()
                        + "/src/test/resources/feature/libperception_feature_plugin.dylib";
        String dataPath = getClass().getResource("/").getPath() + "/feature/data.csv";
        String confPath = getClass().getResource("/").getPath() + "/feature/feature.json";
        String schemaPath = getClass().getResource("/").getPath() + "/feature/schema.csv";

        BatchOperator.setParallelism(1);
        StringBuilder sbd = new StringBuilder();

        String tmpConfDir = FileHandler.getTempDir("/tmp/conf");
        String endPoint = "yourS3EndPoint";
        String bucketName = "yourS3BucketName";
        String s3Key = "yourS3Key";
        String s3AccessKey = "yourS3AccessKey";
        String s3AccessSecret = "yourS3AccessSecret";
        String outputPath = "yourOutputPath";

        for (int i = 0; i < COL_NAMES.length; i++) {
            if (i != 0) {
                sbd.append(",");
            }
            sbd.append(COL_NAMES[i]).append(" ").append(COL_TYPES[i]);
        }

        BatchOperator data =
                new CsvSourceBatchOp()
                        .setIgnoreFirstLine(true)
                        .setFilePath(dataPath)
                        .setSchemaStr(sbd.toString());

        String taskMode = "transform";

        switch (taskMode) {
            case "fit":
                Object obj =
                        JSON.parse(
                                FeatureEngineeringUtils.getJsonFromFilePath(
                                        new FilePath(confPath)));
                Object ops = ((JSONObject) obj).get("operations");
                JSONArray opArr = (JSONArray) ops;

                Pipeline pipeline = new Pipeline();

                for (int i = 0; i < opArr.size(); i++) {
                    pipeline.add(parseJsonToPipelineStage((JSONObject) opArr.get(i)));
                }

                PipelineModel model = pipeline.fit(data);

                /** Compact operations config file */
                List<Row> operConfList =
                        model.save().link(new FeatureEngineeringTransferOp()).collect();
                FileWriter operConfFilePath =
                        FileHandler.createTempFile(tmpConfDir, "operation.conf");
                FileHandler.writeFileOnce(
                        FeatureEngineeringUtils.genOperListJsonString(operConfList),
                        operConfFilePath);

                /** Compact data source config file */
                FileWriter dataSrcFilePath =
                        FileHandler.createTempFile(tmpConfDir, "datasource.conf");
                FileHandler.writeFileOnce(
                        FeatureEngineeringUtils.genDataSrcListJsonString(
                                "prophet_feature_engine", 1, "search", "text_data.conf"),
                        dataSrcFilePath);

                /** Compact data schema config file */
                FileWriter dataSchemaFilePath =
                        FileHandler.createTempFile(tmpConfDir, "text_data.conf");
                FileHandler.writeFileOnce(
                        FeatureEngineeringUtils.genSchemaJsonString(
                                schemaPath, "v1.0", "prophet", "data", ",", ".csv"),
                        dataSchemaFilePath);

                /** Compress configuration files */
                String confTarFile = TarHandler.archive(tmpConfDir);
                String confTgzFile = TarHandler.compressArchive(confTarFile);

                /** Upload to s3 cluster */
                S3Handler.initS3Client(s3AccessKey, s3AccessSecret, endPoint);
                S3Handler.uploadFile(bucketName, s3Key, confTgzFile);
                break;
            case "transform":
                /** Use libfg to transform data * */
                String libfgConfLocalPath = "/tmp/libfgConf";
                String libfgConfRemotePath = endPoint + "/" + bucketName + "/" + s3Key;
                Params params = new Params();
                params.set("libfgSoPath", libfgSoPath);
                params.set("libfgConfLocalPath", libfgConfLocalPath);
                params.set("libfgConfRemotePath", libfgConfRemotePath);
                TextSinkBatchOp outputSink =
                        new TextSinkBatchOp()
                                .setFilePath(
                                        new FilePath(outputPath, new FlinkFileSystem(outputPath)));
                data.link(new LibfgTransferOp(params)).link(outputSink);
                BatchOperator.execute();
                break;
            default:
                System.out.println("Unknown task mode: " + taskMode);
        }
    }
}
