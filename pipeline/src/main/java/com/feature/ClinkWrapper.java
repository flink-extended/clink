package com.feature;

import com.sun.jna.Library;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public interface ClinkWrapper extends Library {
    int FeatureExtractOffline(
            Pointer remotePath, Pointer localPath, Pointer input, PointerByReference output);

    void FeatureOfflineCleanUp(Pointer output);
}
