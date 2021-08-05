package com.feature.common;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.UUID;

public class FileHandler {

    public static String getTempDir(String baseDir) {
        String uuidDir = UUID.randomUUID().toString();
        return String.format("%s/%s", baseDir, uuidDir);
    }

    public static FileWriter createTempFile(String fileDir, String fileName) throws IOException {
        File file = new File(String.format("%s/%s", fileDir, fileName));
        file.getParentFile().mkdirs();
        return new FileWriter(file);
    }

    public static void writeFileOnce(String content, FileWriter fileWriter) throws IOException {
        fileWriter.write(content);
        fileWriter.flush();
        fileWriter.close();
    }

    public static void deleteFile(String filePath) {
        FileUtils.deleteQuietly(new File(filePath));
    }

    public static void deleteDir(String dirPath) throws IOException {
        FileUtils.deleteDirectory(new File(dirPath));
    }
}
