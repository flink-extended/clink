package com.feature.protoparser;

import perception_feature.proto.Datasource;

public class SchemaBuilder {

    private final Datasource.CsvDataConfigList.Builder defaultBuilder =
            Datasource.CsvDataConfigList.newBuilder();

    public SchemaBuilder(
            String version, String description, String dataSrc, String sep, String fileFormat) {
        this.defaultBuilder.setVersion(version);
        this.defaultBuilder.setDescription(description);
        this.defaultBuilder.setDataPath(dataSrc);
        this.defaultBuilder.setSeparator(sep);
        this.defaultBuilder.setFileExtension(fileFormat);
    }

    public void addColumnSpec(Datasource.CsvDataConfig columnSpec) {
        defaultBuilder.addConfigList(columnSpec);
    }

    public Datasource.CsvDataConfigList getBuiltSchema() {
        return defaultBuilder.build();
    }
}
