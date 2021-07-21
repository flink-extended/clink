package run;

import com.alibaba.alink.common.io.filesystem.FilePath;
import com.alibaba.alink.common.io.filesystem.FlinkFileSystem;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.sink.TextSinkBatchOp;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.operator.stream.StreamOperator;
import com.alibaba.alink.operator.stream.sink.KafkaSinkStreamOp;
import com.alibaba.alink.operator.stream.source.KafkaSourceStreamOp;
import com.alibaba.alink.pipeline.Pipeline;
import com.alibaba.alink.pipeline.PipelineModel;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.feature.FeatureEngineeringTransferOp;
import com.feature.FeatureEngineeringUtils;
import com.feature.LibfgTransferBatchOp;
import com.feature.LibfgTransferStreamOp;
import com.feature.common.FileHandler;
import com.feature.common.S3Handler;
import com.feature.common.TarHandler;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.types.Row;
import org.apache.log4j.Logger;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.List;
import java.util.StringJoiner;

import static com.feature.FeatureEngineeringUtils.parseJsonToPipelineStage;

public class runFeatureEngineeringOnline {
    public static Logger logger = Logger.getLogger(runFeatureEngineeringOnline.class);

    @Option(name = "--taskMode", usage = "Mode of task, can be chosen from 'transform'")
    private String taskMode = "transform";

    @Option(name = "--inputKafkaStartupMode", usage = "Startup mode for input stream data")
    private String inputKafkaStartupMode = "EARLIEST";
    @Option(name = "--outputKafkaFormat", usage = "Output data format for output stream data")
    private String outputKafkaFormat = "CSV";
    @Option(
            name = "--inputSchemaPath",
            usage = "Input data schema's file path for feature engineering")
    private String inputSchemaPath;
    @Option(name = "--taskConfPath", usage = "Task configuration file path")
    private String taskConfPath;
    @Option(name = "--libfgSoPath", usage = "Libfg so file path for feature transformation")
    private String libfgSoPath;
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
    @Option(name = "--inputKafkaBroker", usage = "Broker ips for input stream data")
    private String inputKafkaBroker;
    @Option(name = "--inputKafkaTopic", usage = "Kafka topic for input stream data")
    private String inputKafkaTopic;
    @Option(name = "--inputKafkaGroup", usage = "Group ID for input stream data")
    private String inputKafkaGroup;
    @Option(name = "--outputKafkaBroker", usage = "Broker ips for output stream data")
    private String outputKafkaBroker;
    @Option(name = "--outputKafkaTopic", usage = "Kafka topic for output stream data")
    private String outputKafkaTopic;
    @Option(name = "--kafkaPluinVersion", usage = "Alink kafka plugin version")
    private String kafkaPluinVersion = "0.11";

    public static void main(String[] args) throws Exception {
        new runFeatureEngineeringOnline().Main(args);
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

        StreamOperator streamInput =
                new KafkaSourceStreamOp()
                        .setPluginVersion(kafkaPluinVersion)
                        .setBootstrapServers(inputKafkaBroker)
                        .setTopic(inputKafkaTopic)
                        .setGroupId(inputKafkaGroup)
                        .setStartupMode(inputKafkaStartupMode);

        /** Currently we only support streaming transform */
        switch (taskMode) {
            case "transform":
                /** Use libfg to transform data * */
                String libfgConfLocalPath = "/tmp/libfgConf";
                String libfgConfRemotePath = s3EndPoint + "/" + s3Bucket + "/" + s3Key;
                Params params = new Params();
                params.set("libfgSoPath", libfgSoPath);
                params.set("libfgConfLocalPath", libfgConfLocalPath);
                params.set("libfgConfRemotePath", libfgConfRemotePath);
                KafkaSinkStreamOp streamOutput =
                        new KafkaSinkStreamOp()
                                .setPluginVersion(kafkaPluinVersion)
                                .setDataFormat(outputKafkaFormat)
                                .setBootstrapServers(outputKafkaBroker)
                                .setTopic(outputKafkaTopic);
                streamInput.link(new LibfgTransferStreamOp(params)).link(streamOutput);
                StreamOperator.execute();
            default:
                logger.warn("Unknown task mode: " + taskMode);
        }
    }
}
