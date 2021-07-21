package com.feature.protoparser;

import libfg.proto.Datasource;

public class DataSourceListBuilder {

    private final Datasource.DataSourceList.Builder defaultBuilder =
            Datasource.DataSourceList.newBuilder();

    public DataSourceListBuilder(String version, String description) {
        this.defaultBuilder.setVersion(version);
        this.defaultBuilder.setDescription(description);
    }

    public void addDataSource(Datasource.DataSource datasource) {
        defaultBuilder.addDataSource(datasource);
    }

    public Datasource.DataSourceList getBuiltDataSourceList() {
        return defaultBuilder.build();
    }
}
