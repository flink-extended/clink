package com.feature.protoparser;

import perception_feature.proto.Common;
import perception_feature.proto.Operations.Operation;
import perception_feature.proto.Operations.Transform;

public class OperationBuilder {

    private final Operation.Builder defaultBuilder = Operation.newBuilder();

    public OperationBuilder(String outputFeature, Integer featureSize, Integer outputType) {
        this.defaultBuilder.setOutputFeature(outputFeature);
        this.defaultBuilder.setFeatureSize(featureSize);
        this.defaultBuilder.setOutputFeatureType(Common.FeatureType.forNumber(outputType));
    }

    public void addInputFeatures(String inputFeature) {
        defaultBuilder.addInputFeatures(inputFeature);
    }

    public void addOperator(Transform operator) {
        defaultBuilder.addTransform(operator);
    }

    public Operation getBuiltOperation() {
        return defaultBuilder.build();
    }
}
