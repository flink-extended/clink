package com.feature.protoparser;

import libfg.proto.Common;
import libfg.proto.Operations.Operation;
import libfg.proto.Operations.Transform;

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
