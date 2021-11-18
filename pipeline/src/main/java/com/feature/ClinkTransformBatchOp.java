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

public class ClinkTransformBatchOp extends BatchOperator<ClinkTransformBatchOp>
        implements ClinkTransformParams<ClinkTransformBatchOp>, Serializable {

    private static String clinkSoPath;
    private static ClinkWrapper CLINK_INSTANCE;

    public ClinkTransformBatchOp(Params params) {
        super(params);
        clinkSoPath =
                params.getStringOrDefault(
                        "clinkSoPath", "/flink/usrlib/libperception_feature_plugin.so");
        CLINK_INSTANCE = Native.load(clinkSoPath, ClinkWrapper.class);
    }

    private static String FeatureExtract(
            String input, String localPath, String remotePath, String clinkSoPath) {
        /** Flink workers doesn't initialize CLINK_INSTANCE */
        if (null == CLINK_INSTANCE) {
            System.out.printf("Init CLINK_INSTANCE in worker");
            CLINK_INSTANCE = Native.load(clinkSoPath, ClinkWrapper.class);
        }
        Pointer pRemotePath = new Memory((remotePath.length() + 1) * Native.WCHAR_SIZE);
        pRemotePath.setString(0, remotePath);
        Pointer pPath = new Memory((localPath.length() + 1) * Native.WCHAR_SIZE);
        pPath.setString(0, localPath);
        Pointer pInput = new Memory((input.length() + 1) * Native.WCHAR_SIZE);
        pInput.setString(0, input);
        PointerByReference ptrRef = new PointerByReference(Pointer.NULL);
        int res = CLINK_INSTANCE.FeatureExtractOffline(pRemotePath, pPath, pInput, ptrRef);
        if (res != 0) {
            return null;
        }
        final Pointer p = ptrRef.getValue();
        // extract the null-terminated string from the Pointer
        final String val = p.getString(0);
        CLINK_INSTANCE.FeatureOfflineCleanUp(p);
        return val;
    }

    @Override
    public ClinkTransformBatchOp linkFrom(BatchOperator<?>... inputs) {
        String clinkConfLocalPath = getParams().getString("clinkConfLocalPath");
        String clinkConfRemotePath = getParams().getString("clinkConfRemotePath");
        String clinkSoPath =
                getParams()
                        .getStringOrDefault(
                                "clinkSoPath", "/flink/usrlib/libperception_feature_plugin.so");
        DataSet<Row> ret =
                inputs[0]
                        .getDataSet()
                        .map(
                                x ->
                                        Row.of(
                                                FeatureExtract(
                                                        x.toString(),
                                                        clinkConfLocalPath,
                                                        clinkConfRemotePath,
                                                        clinkSoPath)));
        setOutput(ret, new TableSchema(new String[] {"out"}, new TypeInformation[] {Types.STRING}));
        return this;
    }
}
