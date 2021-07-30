#!/usr/bin/env bash

libfgSoPath="$(cd $(dirname "../src/test/resources/feature"); pwd -P)/feature/libperception_feature_plugin.dylib"

echo "libfg library file path: ${libfgSoPath}"

if [[ -f "test_transform_result" ]];then
 rm test_transform_result
fi

localOpConfDir="libfg_conf"

#libfg will read configuration files from localOpConfDir/conf, 
#so we need to move files to conf sub-directory if not existed.
if [[ ! -d "${localOpConfDir}/conf" ]];then
 mkdir -p ${localOpConfDir}/conf
 mv ${localOpConfDir}/*.conf ${localOpConfDir}/conf
fi

flink run -c run.runFeatureEngineeringOffline \
../target/clink-0.1-jar-with-dependencies.jar \
--inputDataPath="../src/test/resources/feature/data.csv" \
--inputSchemaPath="../src/test/resources/feature/schema.csv" \
--isFirstLineHeader \
--taskMode="transform" \
--libfgSoPath="${libfgSoPath}" \
--localOpConfDir="${localOpConfDir}" \
--outputPath="test_transform_result"

