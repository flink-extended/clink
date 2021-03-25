package com.feature.protoparser;

import perception_feature.proto.Datasource;

public class DataSourceBuilder {

    private final Datasource.DataSource.Builder defaultBuilder = Datasource.DataSource.newBuilder();

    public DataSourceBuilder(Integer dataType, String bizName, String dataConf) {
        this.defaultBuilder.setDataType(Datasource.DataSourceType.forNumber(dataType));
        this.defaultBuilder.setBizName(bizName);
        this.defaultBuilder.setDataConf(dataConf);
    }

    public Datasource.DataSource getBuiltDataSource() {
        return defaultBuilder.build();
    }
}
