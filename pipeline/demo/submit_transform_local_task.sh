#!/usr/bin/env bash

clinkSoPath="$(cd $(dirname "../src/test/resources/feature"); pwd -P)/feature/libperception_feature_plugin.dylib"

echo "Clink library file path: ${clinkSoPath}"

if [[ -f "test_transform_result" ]];then
 rm test_transform_result
fi

localOpConfDir="clink_conf"

#Clink will read configuration files from localOpConfDir/conf,
#so we need to move files to conf sub-directory if not existed.
if [[ ! -d "${localOpConfDir}/conf" ]];then
 mkdir -p ${localOpConfDir}/conf
 mv ${localOpConfDir}/*.conf ${localOpConfDir}/conf
fi

flink run -c run.runFeatureEngineeringOffline \
../target/Clink-0.1-jar-with-dependencies.jar \
--inputDataPath="../src/test/resources/feature/data.csv" \
--inputSchemaPath="../src/test/resources/feature/schema.csv" \
--isFirstLineHeader \
--taskMode="transform" \
--clinkSoPath="${clinkSoPath}" \
--localOpConfDir="${localOpConfDir}" \
--outputPath="test_transform_result"

