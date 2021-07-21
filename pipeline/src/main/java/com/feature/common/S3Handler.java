package com.feature.common;

import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.S3ClientOptions;

import java.io.File;

public class S3Handler {

    public static AmazonS3Client INSTANCE;

    public static AmazonS3Client initS3Client(String accessKey, String secretKey, String endPoint) {
        if (null == INSTANCE) {
            BasicAWSCredentials yourAWSCredentials = new BasicAWSCredentials(accessKey, secretKey);
            AmazonS3Client amazonS3Client = new AmazonS3Client(yourAWSCredentials);
            amazonS3Client.setEndpoint(endPoint);
            S3ClientOptions clientOptions = new S3ClientOptions();
            clientOptions.setPathStyleAccess(true);
            clientOptions.disableChunkedEncoding();

            amazonS3Client.setS3ClientOptions(clientOptions);
            INSTANCE = amazonS3Client;
        }
        return INSTANCE;
    }

    public static void uploadFile(String bucket, String key, String filePath) throws Exception {
        if (null == INSTANCE) {
            throw new Exception("please init AmazonS3Client");
        }
        INSTANCE.putObject(bucket, key, new File(filePath));
    }
}
