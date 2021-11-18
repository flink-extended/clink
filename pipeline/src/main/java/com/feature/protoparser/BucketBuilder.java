package com.feature.protoparser;

import clink.proto.Operations.Transform;

public class BucketBuilder extends BaseOperatorBuilder {

    private final Transform.Builder defaultBuilder = Transform.newBuilder();

    public BucketBuilder(String formula) {
        super(String.format("BUCKET(%s)", formula));
    }
}
