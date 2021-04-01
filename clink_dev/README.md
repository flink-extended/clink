## Clink Pipeline

Clink Pipeline是基于Alink的特征工程pipeline构建工具，用于生成libfg转换特征使用的配置文件。

Clink Pipeline的工作流程如下：

1. 编写任务算子配置文件，配置一次任务中使用到的算子和算子对应的参数。
2. 编写待处理数据的schema文件。
3. 使用Clink Pipeline读取上述配置文件和任务数据，进行fit操作，得到libfg transform所需的配置文件。
4. 打包压缩产出的配置文件，上传至指定的S3集群。
5. libfg加载S3集群中的配置文件，提供特征转换服务。

##### 上述操作支持本地和VVP平台两种运行方式，数据读取支持本地数据和HDFS数据。

项目提供了用于进行测试的数据，schema和任务配置，可以在`src/test/resources/feature` 下找到。



##### 提交任务所需指定的参数包括：

1. inputDataPath: 任务数据路径。
2. isFirstLineHeader: 任务数据首行是否为字段名。
3. columnSep: 列分隔符。
4. inputSchemaPath: 任务数据schema文件路径。
5. taskConfPath: 任务算子配置文件。
6. s3AccessKey: S3集群访问账号。
7. s3AccessSecret: S3集群访问密码。
8. s3EndPoint: S3集群访问地址。
9. s3Bucket: S3集群bucket名称。
10. s3Key: 当前任务上传产出配置文件压缩包所使用的S3 object key名称。



##### 目前为了兼容旧的实现，Clink Pipeline有以下限制:

- 每个算子仅支持单个输入和单个输出，即一个算子仅针对单个特征进行分析处理。
- 目前仅支持CSV格式数据，输出仅支持LibSVM格式数据。
- 仅支持部分算子实现，目前支持的算子包括：
  - OneHot
  - StandardScaler
  - LabelEncoder
  - Bucket
  - Keep
  - Drop
  - UDF（libfg形式的UDF）

