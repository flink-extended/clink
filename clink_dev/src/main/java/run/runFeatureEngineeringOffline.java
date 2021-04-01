package run;

import com.alibaba.alink.common.io.filesystem.FilePath;
import com.alibaba.alink.common.io.filesystem.FlinkFileSystem;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.pipeline.Pipeline;
import com.alibaba.alink.pipeline.PipelineModel;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.feature.FeatureEngineeringTransferOp;
import com.feature.FeatureEngineeringUtils;
import com.feature.common.FileHandler;
import com.feature.common.S3Handler;
import com.feature.common.TarHandler;
import org.apache.flink.types.Row;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.List;
import java.util.StringJoiner;

import static com.feature.FeatureEngineeringUtils.parseJsonToPipelineStage;

public class runFeatureEngineeringOffline {
    @Option(name = "--inputDataPath", usage = "Input data's file path for feature engineering")
    private String inputDataPath;

    @Option(
            name = "--isFirstLineHeader",
            usage = "Flag for check if data's first line is header line")
    private boolean isFirstLineHeader = false;

    @Option(name = "--columnSep", usage = "Column delimiter")
    private String columnSep = ",";

    @Option(
            name = "--inputSchemaPath",
            usage = "Input data schema's file path for feature engineering")
    private String inputSchemaPath;

    @Option(name = "--taskConfPath", usage = "Task configuration file path")
    private String taskConfPath;

    @Option(name = "--s3AccessKey", usage = "S3 cluster's access user id")
    private String s3AccessKey;

    @Option(name = "--s3AccessSecret", usage = "S3 cluster's access password")
    private String s3AccessSecret;

    @Option(name = "--s3EndPoint", usage = "S3 cluster's endpoint")
    private String s3EndPoint;

    @Option(name = "--s3Bucket", usage = "S3 cluster's bucket name")
    private String s3Bucket;

    @Option(name = "--s3Key", usage = "S3 cluster's object's key name")
    private String s3Key;

    public static void main(String[] args) throws Exception {
        new runFeatureEngineeringOffline().Main(args);
    }

    public void Main(String[] args) throws Exception {
        CmdLineParser parser = new CmdLineParser(this);
        parser.parseArgument(args);

        StringJoiner schemaBuilder = new StringJoiner(",");
        BufferedReader reader = new BufferedReader(new FileReader(inputSchemaPath));
        try {
            reader.lines()
                    .forEach(
                            row -> {
                                schemaBuilder.add(row);
                            });
        } catch (Exception e) {
            throw e;
        } finally {
            reader.close();
        }

        BatchOperator data;

        if (inputDataPath.startsWith("hdfs://")) {
            data =
                    new CsvSourceBatchOp()
                            .setIgnoreFirstLine(isFirstLineHeader)
                            .setFieldDelimiter(columnSep)
                            .setFilePath(
                                    new FilePath(inputDataPath, new FlinkFileSystem(inputDataPath)))
                            .setSchemaStr(schemaBuilder.toString());
        } else {
            data =
                    new CsvSourceBatchOp()
                            .setIgnoreFirstLine(isFirstLineHeader)
                            .setFieldDelimiter(columnSep)
                            .setFilePath(inputDataPath)
                            .setSchemaStr(schemaBuilder.toString());
        }

        Object obj =
                JSON.parse(FeatureEngineeringUtils.getJsonFromFilePath(new FilePath(taskConfPath)));
        Object ops = ((JSONObject) obj).get("operations");
        JSONArray opArr = (JSONArray) ops;

        Pipeline pipeline = new Pipeline();

        for (int i = 0; i < opArr.size(); i++) {
            pipeline.add(parseJsonToPipelineStage((JSONObject) opArr.get(i)));
        }

        PipelineModel model = pipeline.fit(data);

        /** Compact operations config file */
        List<Row> operConfList = model.save().link(new FeatureEngineeringTransferOp()).collect();
        String tmpConfDir = FileHandler.getTempDir("/flink/");
        FileWriter operConfFilePath = FileHandler.createTempFile(tmpConfDir, "operation.conf");
        FileHandler.writeFileOnce(
                FeatureEngineeringUtils.genOperListJsonString(operConfList), operConfFilePath);

        /** Compact data source config file */
        FileWriter dataSrcFilePath = FileHandler.createTempFile(tmpConfDir, "datasource.conf");
        FileHandler.writeFileOnce(
                FeatureEngineeringUtils.genDataSrcListJsonString(
                        "prophet_feature_engine", 1, "search", "text_data.conf"),
                dataSrcFilePath);

        /** Compact data schema config file */
        FileWriter dataSchemaFilePath = FileHandler.createTempFile(tmpConfDir, "text_data.conf");
        FileHandler.writeFileOnce(
                FeatureEngineeringUtils.genSchemaJsonString(
                        inputSchemaPath, "v1.0", "prophet", "data", ",", ".csv"),
                dataSchemaFilePath);

        /** Compress configuration files */
        String confTarFile = TarHandler.archive(tmpConfDir);
        String confTgzFile = TarHandler.compressArchive(confTarFile);

        /** Upload to s3 cluster */
        S3Handler.initS3Client(s3AccessKey, s3AccessSecret, s3EndPoint);
        S3Handler.uploadFile(s3Bucket, s3Key, confTgzFile);
    }
}
