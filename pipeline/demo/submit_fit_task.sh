#!/usr/bin/env bash

s3_access_key=$1
s3_access_secret=$2
s3_endpoint=$3
s3_bucket=$4
s3_key=$5

if [[ -z "${s3_access_key}" ]];then
 echo "Empty s3_access_key"
 exit 1
fi

if [[ -z "${s3_access_secret}" ]];then
 echo "Empty s3_access_secret"
 exit 1
fi

if [[ -z "${s3_endpoint}" ]];then
 echo "Empty s3_endpoint"
 exit 1
fi

if [[ -z "${s3_bucket}" ]];then
 echo "Empty s3_bucket"
 exit 1
fi

if [[ -z "${s3_key}" ]];then
 echo "Empty s3_key"
 exit 1
fi

libfg_so_path="$(cd $(dirname "../src/test/resources/feature"); pwd -P)/feature/libperception_feature_plugin.dylib"

echo "libfg.so file path: ${libfg_so_path}"

if [[ -d "libfg_conf" ]];then
 rm -rf libfg_conf
fi
mkdir libfg_conf

flink run -c run.runFeatureEngineeringOffline \
../target/clink-0.1-jar-with-dependencies.jar \
--inputDataPath="../src/test/resources/feature/data.csv" \
--isFirstLineHeader \
--taskMode="fit" \
--inputSchemaPath="../src/test/resources/feature/schema.csv" \
--taskConfPath="../src/test/resources/feature/feature.json" \
--libfgSoPath="${libfg_so_path}" \
--localOpConfDir="libfg_conf" \
--s3AccessKey="${s3_access_key}" \
--s3AccessSecret="${s3_access_secret}" \
--s3EndPoint="${s3_endpoint}" \
--s3Bucket="${s3_bucket}" \
--s3Key="${s3_key}"
