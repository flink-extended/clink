package com.feature.protoparser;

import perception_feature.proto.Operations.Transform;

public class UdfBuilder extends BaseOperatorBuilder {

    private final Transform.Builder defaultBuilder = Transform.newBuilder();

    public UdfBuilder(String formula) {
        super(formula);
    }
}
