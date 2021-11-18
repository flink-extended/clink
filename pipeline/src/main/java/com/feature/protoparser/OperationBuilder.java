package com.feature.protoparser;

import clink.proto.Common;
import clink.proto.Operations.Operation;
import clink.proto.Operations.Transform;

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
