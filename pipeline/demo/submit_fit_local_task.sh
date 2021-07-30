#!/usr/bin/env bash

libfgSoPath="$(cd $(dirname "../src/test/resources/feature"); pwd -P)/feature/libperception_feature_plugin.dylib"

echo "libfg library file path: ${libfgSoPath}"

localOpConfDir="libfg_conf"

if [[ -d "${localOpConfDir}" ]];then
 rm -rf ${localOpConfDir}
fi
mkdir ${localOpConfDir}

flink run -c run.runFeatureEngineeringOffline \
../target/clink-0.1-jar-with-dependencies.jar \
--inputDataPath="../src/test/resources/feature/data.csv" \
--isFirstLineHeader \
--taskMode="fit" \
--inputSchemaPath="../src/test/resources/feature/schema.csv" \
--taskConfPath="../src/test/resources/feature/feature.json" \
--libfgSoPath="${libfgSoPath}" \
--localOpConfDir="${localOpConfDir}"

