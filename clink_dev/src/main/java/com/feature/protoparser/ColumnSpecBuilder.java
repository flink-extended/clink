package com.feature.protoparser;

import perception_feature.proto.Datasource;

public class ColumnSpecBuilder {

    private final Datasource.CsvDataConfig.Builder defaultBuilder =
            Datasource.CsvDataConfig.newBuilder();

    public ColumnSpecBuilder(Integer colIndex, String colName, String colType) {
        this.defaultBuilder.setColumn(colIndex);
        this.defaultBuilder.setFeatureName(colName);
        this.defaultBuilder.setFeatureDataType(
                Datasource.FeatureDataType.forNumber(ColTypePbMap.getIntType(colType)));
    }

    public Datasource.CsvDataConfig getBuiltColumnSpec() {
        return defaultBuilder.build();
    }
}


/**
 * Column type name mapping
 */
class ColTypePbMap {

    public static Integer getIntType(String strType) {

        Integer intType = 1;
        switch (strType) {
            case "int":
            case "float":
                intType = 3;
        }

        return intType;
    }
}
