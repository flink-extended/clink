package com.feature;

import com.alibaba.alink.operator.batch.BatchOperator;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;

import java.io.Serializable;

public class LibfgTransferBatchOp extends BatchOperator<LibfgTransferBatchOp>
        implements LibfgTransferParams<LibfgTransferBatchOp>, Serializable {

    private static String libfgSoPath;
    private static LibfgWrapper LIBFG_INSTANCE;

    public LibfgTransferBatchOp(Params params) {
        super(params);
        libfgSoPath =
                params.getStringOrDefault(
                        "libfgSoPath", "/flink/usrlib/libperception_feature_plugin.so");
        LIBFG_INSTANCE = Native.load(libfgSoPath, LibfgWrapper.class);
    }

    private static String FeatureExtract(
            String input, String localPath, String remotePath, String libfgSoPath) {
        /** Flink workers doesn't initialize LIBFG_INSTANCE */
        if (null == LIBFG_INSTANCE) {
            System.out.printf("Init LIBFG_INSTANCE in worker");
            LIBFG_INSTANCE = Native.load(libfgSoPath, LibfgWrapper.class);
        }
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

    @Override
    public LibfgTransferBatchOp linkFrom(BatchOperator<?>... inputs) {
        String libfgConfLocalPath = getParams().getString("libfgConfLocalPath");
        String libfgConfRemotePath = getParams().getString("libfgConfRemotePath");
        String libfgSoPath =
                getParams()
                        .getStringOrDefault(
                                "libfgSoPath", "/flink/usrlib/libperception_feature_plugin.so");
        DataSet<Row> ret =
                inputs[0]
                        .getDataSet()
                        .map(
                                x ->
                                        Row.of(
                                                FeatureExtract(
                                                        x.toString(),
                                                        libfgConfLocalPath,
                                                        libfgConfRemotePath,
                                                        libfgSoPath)));
        setOutput(ret, new TableSchema(new String[] {"out"}, new TypeInformation[] {Types.STRING}));
        return this;
    }
}
