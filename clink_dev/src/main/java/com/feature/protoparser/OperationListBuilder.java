package com.feature.protoparser;

import perception_feature.proto.Common.OutputFormat;
import perception_feature.proto.Operations.Operation;
import perception_feature.proto.Operations.OperationList;

public class OperationListBuilder {

    private final OperationList.Builder defaultBuilder = OperationList.newBuilder();

    public OperationListBuilder(String version, Integer outputFormat) {
        this.defaultBuilder.setOutputFormat(OutputFormat.forNumber(outputFormat));
        this.defaultBuilder.setVersion(version);
    }

    public void addOperation(Operation operation) {
        defaultBuilder.addOperation(operation);
    }

    public OperationList getBuiltOperationList() {
        return defaultBuilder.build();
    }
}
