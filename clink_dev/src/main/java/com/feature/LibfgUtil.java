package com.feature;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public class LibfgUtil {

    private final LibfgWrapper LIBFG_INSTANCE;

    public LibfgUtil(String libfgSoPath) {
        this.LIBFG_INSTANCE = Native.load(libfgSoPath, LibfgWrapper.class);
    }

    public String FeatureExtract(String input, String localPath, String remotePath) {
        Pointer pRemotePath = new Memory((remotePath.length() + 1) * Native.WCHAR_SIZE);
        pRemotePath.setString(0, remotePath);
        Pointer pPath = new Memory((localPath.length() + 1) * Native.WCHAR_SIZE);
        pPath.setString(0, localPath);
        Pointer pInput = new Memory((input.length() + 1) * Native.WCHAR_SIZE);
        pInput.setString(0, input);
        PointerByReference ptrRef = new PointerByReference(Pointer.NULL);
        int res = LIBFG_INSTANCE.FeatureExtractOffline(pRemotePath, pPath, pInput, ptrRef);
        if (res != 0) {
            return null;
        }
        final Pointer p = ptrRef.getValue();
        // extract the null-terminated string from the Pointer
        final String val = p.getString(0);
        LIBFG_INSTANCE.FeatureOfflineCleanUp(p);
        return val;
    }
}
