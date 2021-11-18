package com.feature;

import com.alibaba.alink.operator.stream.StreamOperator;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.apache.commons.lang3.StringUtils;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;

import java.io.Serializable;
import java.util.Arrays;

public class ClinkTransformStreamOp extends StreamOperator<ClinkTransformStreamOp>
        implements ClinkTransformParams<ClinkTransformStreamOp>, Serializable {

    private static String clinkSoPath;
    private static ClinkWrapper CLINK_INSTANCE;

    public ClinkTransformStreamOp(Params params) {
        super(params);
        clinkSoPath =
                params.getStringOrDefault(
                        "clinkSoPath", "/flink/usrlib/libperception_feature_plugin.so");
        CLINK_INSTANCE = Native.load(clinkSoPath, ClinkWrapper.class);
    }

    private static String FeatureExtract(String input, String localPath, String remotePath, String clinkSoPath) {
        /** Flink workers doesn't initialize CLINK_INSTANCE */
        if (null == CLINK_INSTANCE) {
            System.out.printf("Init CLINK_INSTANCE in worker");
            CLINK_INSTANCE = Native.load(clinkSoPath, ClinkWrapper.class);
        }
        /** Alink kafka consumer will add extra fields */
        String[] inputArr = input.split(",");
        String truncated_input =
                StringUtils.join(Arrays.asList(inputArr).subList(1, inputArr.length - 3), ",");
        Pointer pRemotePath = new Memory((remotePath.length() + 1) * Native.WCHAR_SIZE);
        pRemotePath.setString(0, remotePath);
        Pointer pPath = new Memory((localPath.length() + 1) * Native.WCHAR_SIZE);
        pPath.setString(0, localPath);
        Pointer pInput = new Memory((truncated_input.length() + 1) * Native.WCHAR_SIZE);
        pInput.setString(0, truncated_input);
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
    public ClinkTransformStreamOp linkFrom(StreamOperator<?>... inputs) {
        String clinkConfLocalPath = getParams().getString("clinkConfLocalPath");
        String clinkConfRemotePath = getParams().getString("clinkConfRemotePath");
        String clinkSoPath = getParams().getStringOrDefault(
                "clinkSoPath", "/flink/usrlib/libperception_feature_plugin.so");
        SingleOutputStreamOperator<Row> ret =
                inputs[0]
                        .getDataStream()
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
