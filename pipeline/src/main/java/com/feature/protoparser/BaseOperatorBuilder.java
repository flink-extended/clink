package com.feature.protoparser;

import libfg.proto.Common.RecordEntry;
import libfg.proto.Operations.Transform;

public class BaseOperatorBuilder {

    private final Transform.Builder defaultBuilder = Transform.newBuilder();

    public BaseOperatorBuilder(String formula) {
        this.defaultBuilder.setFormula(formula);
    }

    public void addParam(RecordEntry record) {
        defaultBuilder.addParams(record);
    }

    public Transform getBuiltOperator() {
        return defaultBuilder.build();
    }
}
