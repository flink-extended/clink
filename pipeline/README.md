## Clink Pipeline

Clink Pipeline is a Flink-based feature engineering flow builder that enpowers stream- and batch-feature transformation.

Traditional architectures typically use a java-based offline processing framework like spark to process bulk data, and then use a C/C++-based implementation for online services to provide more efficient processing. This implementation makes it difficult to verify the consistency of offline online results. Clink Pipeline leverages the streaming processing power of flink and the efficient processing power of Clink engine to improve data throughput while ensuring efficiency.



### Features

+ Utilization of Clink engine for efficient data processing
+ Flexible APIs in python for developers to build processing pipeline
+ Using Apache Flink that supports high throughput and low latency at the same time
+ Rich operator library provides developers more options
+ Built-in DAG which optimizes the pipeline generation
+ Integration with HDFS, Kafka, and other file systems



### Building Clink Pipeline from Source

Prerequisites for building Clink Pipeline:

+ Unix-like environment(we use Linux, Mac OS X)
+ Git
+ Maven
+ Java8
+ Protobuf3

```
git clone https://github.com/Qihoo360/Clink.git
cd pipeline
mvn clean package -DskipTests
```

NOTE: Users may need to download plugins like kafka, we recommand use the plugin downloader located in ` pipeline/src/main/java/com/feature/common/PluginHandler`  to download the specific plugins. 

After build process, in target directory, you can find Clink-0.1.jar that only contains Clink-pipeline classes, and Clink-0.1-jar-with-dependencies.jar that contains dependencies. User can use Clink-0.1-jar-with-dependencies.jar for convenience.



### Demo

We add simple examples in demo directory, user can try Clink-pipeline in local flink cluster by running given scripts. Tested on MacOS. Before trying demo scripts, user need to install flink and create a local flink cluster, detailed local installation instruction can refer to [Local Installation | Apache Flink](https://ci.apache.org/projects/flink/flink-docs-release-1.13/docs/try-flink/local_installation/#:~:text=Local Installation. 1 Step 1%3A Download. To be,Job. 4 Step 4%3A Stop the Cluster. ).
