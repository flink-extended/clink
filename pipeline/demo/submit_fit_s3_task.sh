#!/usr/bin/env bash

s3AccessKey=$1
s3AccessSecret=$2
s3EndPoint=$3
s3Bucket=$4
s3Key=$5

if [[ -z "${s3AccessKey}" ]];then
 echo "Empty s3AccessKey"
 exit 1
fi

if [[ -z "${s3AccessSecret}" ]];then
 echo "Empty s3AccessSecret"
 exit 1
fi

if [[ -z "${s3EndPoint}" ]];then
 echo "Empty s3EndPoint"
 exit 1
fi

if [[ -z "${s3Bucket}" ]];then
 echo "Empty s3Bucket"
 exit 1
fi

if [[ -z "${s3Key}" ]];then
 echo "Empty s3Key"
 exit 1
fi

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
--localOpConfDir="${localOpConfDir}" \
--storeConfInS3 \
--s3AccessKey="${s3AccessKey}" \
--s3AccessSecret="${s3AccessSecret}" \
--s3EndPoint="${s3EndPoint}" \
--s3Bucket="${s3Bucket}" \
--s3Key="${s3Key}"

