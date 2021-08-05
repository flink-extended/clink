package com.feature.protoparser;

import libfg.proto.Common.OutputFormat;
import libfg.proto.Operations.Operation;
import libfg.proto.Operations.OperationList;

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
