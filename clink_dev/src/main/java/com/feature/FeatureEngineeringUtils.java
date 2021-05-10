package com.feature;

import com.alibaba.alink.common.io.filesystem.FilePath;
import com.alibaba.alink.pipeline.PipelineStageBase;
import com.alibaba.fastjson.JSONObject;
import com.feature.protoparser.*;
import com.feature.stageparser.BaseStageParser;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.types.Row;
import perception_feature.proto.Operations;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.apache.commons.io.IOUtils.readLines;

public class FeatureEngineeringUtils {

    public static PipelineStageBase<?> parseJsonToPipelineStage(JSONObject obj)
            throws IllegalAccessException, InstantiationException, ClassNotFoundException {
        String opName = ((String) obj.get("operName")).toLowerCase();
        opName = opName.substring(0, 1).toUpperCase() + opName.substring(1);
        return BaseStageParser.parseJsonToPipelineWithName(obj, opName + "Parser");
    }

    public static String getJsonFromFilePath(FilePath path) {
        FileSystem fs = path.getFileSystem();
        StringBuilder jsonStr = new StringBuilder();
        try {
            FSDataInputStream stream = fs.open(path.getPath());
            List<String> lines = readLines(stream);
            for (String line : lines) {
                jsonStr.append(line.trim());
            }
        } catch (Exception e) {
            throw new RuntimeException("Feature engineering config file io err.", e);
        }
        return jsonStr.toString();
    }

    public static String genOperListJsonString(List<Row> operConfList)
            throws InvalidProtocolBufferException {
        OperationListBuilder operListBuilder = new OperationListBuilder("v1.0", 1);
        operConfList.forEach(
                x -> {
                    try {
                        Operations.Operation.Builder structBuilder =
                                Operations.Operation.newBuilder();
                        JsonFormat.parser().merge(x.toString(), structBuilder);
                        operListBuilder.addOperation(structBuilder.build());
                    } catch (InvalidProtocolBufferException e) {
                        e.printStackTrace();
                    }
                });

        return JsonFormat.printer()
                .includingDefaultValueFields()
                .preservingProtoFieldNames()
                .print(operListBuilder.getBuiltOperationList());
    }

    public static String genOperListJsonStringOp(DataSet<Row> operConfList)
            throws InvalidProtocolBufferException {
        OperationListBuilder operListBuilder = new OperationListBuilder("v1.0", 1);
        operConfList.map(
                x -> {
                    try {
                        Operations.Operation.Builder structBuilder =
                                Operations.Operation.newBuilder();
                        JsonFormat.parser().merge(x.toString(), structBuilder);
                        operListBuilder.addOperation(structBuilder.build());
                    } catch (InvalidProtocolBufferException e) {
                        e.printStackTrace();
                    }
                    return x;
                });

        return JsonFormat.printer()
                .includingDefaultValueFields()
                .preservingProtoFieldNames()
                .print(operListBuilder.getBuiltOperationList());
    }

    public static String genDataSrcListJsonString(
            String description, Integer dataType, String bizName, String schemaPath)
            throws InvalidProtocolBufferException {
        DataSourceListBuilder dataSrcListBuilder = new DataSourceListBuilder("v1.0", description);
        DataSourceBuilder dataSrcBuilder = new DataSourceBuilder(dataType, bizName, schemaPath);
        dataSrcListBuilder.addDataSource(dataSrcBuilder.getBuiltDataSource());

        return JsonFormat.printer()
                .includingDefaultValueFields()
                .preservingProtoFieldNames()
                .print(dataSrcListBuilder.getBuiltDataSourceList());
    }

    public static String genSchemaJsonString(
            String schemaFile,
            String version,
            String description,
            String dataSrc,
            String sep,
            String fileFormat)
            throws FileNotFoundException, InvalidProtocolBufferException {
        BufferedReader reader = new BufferedReader(new FileReader(schemaFile));
        AtomicInteger colIndex = new AtomicInteger();
        SchemaBuilder schemaBuilder =
                new SchemaBuilder(version, description, dataSrc, sep, fileFormat);
        reader.lines()
                .forEach(
                        row -> {
                            String[] colSpecArray = row.trim().split(" ");
                            ColumnSpecBuilder colSpecBuilder =
                                    new ColumnSpecBuilder(
                                            colIndex.getAndIncrement(),
                                            colSpecArray[0],
                                            colSpecArray[1]);
                            schemaBuilder.addColumnSpec(colSpecBuilder.getBuiltColumnSpec());
                        });

        return JsonFormat.printer()
                .includingDefaultValueFields()
                .preservingProtoFieldNames()
                .print(schemaBuilder.getBuiltSchema());
    }
}
