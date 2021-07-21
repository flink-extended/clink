package benchmark;

import com.alibaba.alink.common.AlinkGlobalConfiguration;
import com.alibaba.alink.common.io.filesystem.BaseFileSystem;
import com.alibaba.alink.common.io.filesystem.FilePath;
import com.alibaba.alink.common.io.filesystem.S3HadoopFileSystem;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.sink.CsvSinkBatchOp;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.pipeline.Pipeline;
import com.alibaba.alink.pipeline.PipelineModel;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.feature.FeatureEngineeringTransferOp;
import com.feature.FeatureEngineeringUtils;

import static com.feature.FeatureEngineeringUtils.parseJsonToPipelineStage;

public class S3FeatureModelGenerator {
	private static final String[] COL_NAMES = new String[] {
		"id", "click", "dt", "C1", "banner_pos",
		"site_id", "site_domain", "site_category", "app_id", "app_domain",
		"app_category", "device_id", "device_ip", "device_model", "device_type",
		"device_conn_type", "C14", "C15", "C16", "C17",
		"C18", "C19", "C20", "C21"
	};

	private static final String[] COL_TYPES = new String[] {
		"string", "int", "string", "string", "int",
		"string", "string", "string", "string", "string",
		"string", "string", "string", "string", "string",
		"string", "int", "int", "int", "int",
		"int", "int", "int", "int"
	};

	private static String getSchema() {
		StringBuilder sbd = new StringBuilder();
		for (int i = 0; i < COL_NAMES.length; i++) {
			if (i != 0) {
				sbd.append(",");
			}
			sbd.append(COL_NAMES[i]).append(" ").append(COL_TYPES[i]);
		}
		return sbd.toString();
	}

	public static void main(String[] args) throws Exception {

		String version = "1.11.788";
		String endpoint = "http://10.101.201.66:9000";
		String bucketName = "test";
		String accessId = "*";
		String accessKey = "*";
		final String pluginDir = "/Users/x/Downloads/alink_plugin";

		AlinkGlobalConfiguration.setPluginDir(pluginDir);

		BaseFileSystem <?> fs = new S3HadoopFileSystem(version, endpoint, bucketName, accessId, accessKey, false);

		BatchOperator.setParallelism(1);

		BatchOperator <?> data = new CsvSourceBatchOp()
			.setIgnoreFirstLine(true)
			.setFilePath(new FilePath("/feature/small.csv", fs))
			.setSchemaStr(getSchema());

		Object obj = JSON.parse(FeatureEngineeringUtils.getJsonFromFilePath(new FilePath("/feature/feature.json", fs)));
		Object ops = ((JSONObject) obj).get("operations");
		JSONArray opArr = (JSONArray) ops;

		Pipeline pipeline = new Pipeline();

		for (Object o : opArr) {
			pipeline.add(parseJsonToPipelineStage((JSONObject) o));
		}

		PipelineModel model = pipeline.fit(data);

		model
			.save()
			.link(new FeatureEngineeringTransferOp())
			.link(
				new CsvSinkBatchOp()
					.setOverwriteSink(true)
					.setQuoteChar(null)
					.setFilePath(new FilePath("/feature/model_20210315.json", fs))
			);
		BatchOperator.execute();

	}
}