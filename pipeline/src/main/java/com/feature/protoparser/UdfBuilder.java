package com.feature.protoparser;

import clink.proto.Operations.Transform;

public class UdfBuilder extends BaseOperatorBuilder {

    private final Transform.Builder defaultBuilder = Transform.newBuilder();

    public UdfBuilder(String formula) {
        super(formula);
    }
}
