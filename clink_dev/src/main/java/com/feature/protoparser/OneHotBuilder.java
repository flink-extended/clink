package com.feature.protoparser;

import perception_feature.proto.Operations.Transform;

public class OneHotBuilder extends BaseOperatorBuilder {

    private final Transform.Builder defaultBuilder = Transform.newBuilder();

    public OneHotBuilder(String formula) {
        super(String.format("ONE_HOT(%s)", formula));
    }
}
