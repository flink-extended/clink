package com.feature.protoparser;

import perception_feature.proto.Operations.Transform;

public class MultiHotBuilder extends BaseOperatorBuilder {

    private final Transform.Builder defaultBuilder = Transform.newBuilder();

    public MultiHotBuilder(String formula) {
        super(String.format("MULTI_HOT(%s)", formula));
    }
}
