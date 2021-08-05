package com.feature.common;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;

import java.io.*;

public class TarHandler {

    public static String archive(String entry) throws IOException {
        File file = new File(entry);
        TarArchiveOutputStream tos =
                new TarArchiveOutputStream(new FileOutputStream(file.getAbsolutePath() + ".tar"));
        if (file.isDirectory()) archiveDir(file, tos);
        else archiveHandle(tos, file);
        tos.close();
        return file.getAbsolutePath() + ".tar";
    }

    public static void archiveDir(File file, TarArchiveOutputStream tos) throws IOException {
        File[] listFiles = file.listFiles();
        for (File fi : listFiles) {
            if (fi.isDirectory()) archiveDir(fi, tos);
            else archiveHandle(tos, fi);
        }
    }

    public static void archiveHandle(TarArchiveOutputStream tos, File file) throws IOException {
        TarArchiveEntry tEntry = new TarArchiveEntry(File.separator + file.getName());
        tEntry.setSize(file.length());
        tos.putArchiveEntry(tEntry);
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(file));
        byte[] buffer = new byte[1024];
        int read;

        while ((read = bis.read(buffer)) != -1) {
            tos.write(buffer, 0, read);
        }

        bis.close();
        tos.closeArchiveEntry();
    }

    public static String compressArchive(String path) throws IOException {
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(path));
        GzipCompressorOutputStream gcos =
                new GzipCompressorOutputStream(
                        new BufferedOutputStream(new FileOutputStream(path + ".gz")));
        byte[] buffer = new byte[1024];
        int read;

        while ((read = bis.read(buffer)) != -1) {
            gcos.write(buffer, 0, read);
        }

        gcos.close();
        bis.close();
        return path + ".gz";
    }
}
