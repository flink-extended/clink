package com.feature.protoparser;

import perception_feature.proto.Operations.Transform;

public class LabelEncoderBuilder extends BaseOperatorBuilder {

    private final Transform.Builder defaultBuilder = Transform.newBuilder();

    public LabelEncoderBuilder(String formula) {
        super(String.format("LABEL_ENCODER(%s)", formula));
    }
}
