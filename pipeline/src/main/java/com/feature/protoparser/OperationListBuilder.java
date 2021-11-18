package com.feature.protoparser;

import clink.proto.Common.OutputFormat;
import clink.proto.Operations.Operation;
import clink.proto.Operations.OperationList;

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
