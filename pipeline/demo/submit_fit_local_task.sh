#!/usr/bin/env bash

clinkSoPath="$(cd $(dirname "../src/test/resources/feature"); pwd -P)/feature/libperception_feature_plugin.dylib"

echo "Clink library file path: ${clinkSoPath}"

localOpConfDir="clink_conf"

if [[ -d "${localOpConfDir}" ]];then
 rm -rf ${localOpConfDir}
fi
mkdir ${localOpConfDir}

flink run -c run.runFeatureEngineeringOffline \
../target/Clink-0.1-jar-with-dependencies.jar \
--inputDataPath="../src/test/resources/feature/data.csv" \
--isFirstLineHeader \
--taskMode="fit" \
--inputSchemaPath="../src/test/resources/feature/schema.csv" \
--taskConfPath="../src/test/resources/feature/feature.json" \
--clinkSoPath="${clinkSoPath}" \
--localOpConfDir="${localOpConfDir}"

