package com.feature.protoparser;

import libfg.proto.Operations.Transform;

public class StdBuilder extends BaseOperatorBuilder {

    private final Transform.Builder defaultBuilder = Transform.newBuilder();

    public StdBuilder(String formula) {
        super(String.format("STD(%s)", formula));
    }
}
