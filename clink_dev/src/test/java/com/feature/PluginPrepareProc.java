package com.feature;

import com.alibaba.alink.common.AlinkGlobalConfiguration;
import com.alibaba.alink.common.io.plugin.PluginDownloader;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

public class PluginPrepareProc {
    @Test
    public void pluginPrepare() throws Exception {
        // 设置插件下载的位置，当路径不存在时会自行创建路径
        AlinkGlobalConfiguration.setPluginDir("/Users/weibo/.alink_plugins/");

        // 获得Alink插件下载器
        PluginDownloader pluginDownloader = AlinkGlobalConfiguration.getPluginDownloader();

        // 从远程加载插件的配置项
        pluginDownloader.loadConfig();

        // 展示所有可用的插件名称
        List<String> plugins = pluginDownloader.listAvailablePlugins();
        System.out.println(Arrays.toString(plugins.toArray()));
        // 输出结果：[oss, hive, derby, mysql, hadoop, sqlite]

        // 显示第0个插件的所有版本
        String pluginName = plugins.get(7); // oss
        //List<String> availableVersions = pluginDownloader.listAvailablePluginVersions(pluginName);
        // 输出结果：[3.4.1]

        // 下载某个插件的特定版本
        //String pluginVersion = availableVersions.get(0);
        //pluginDownloader.downloadPlugin(pluginName, pluginVersion);
        // 运行结束后，插件会被下载到"/Users/xxx/alink_plugins/"目录中

        // 下载某个插件的默认版本
        //pluginDownloader.downloadPlugin(pluginName);
        // 运行结束后，插件会被下载到"/Users/xxx/alink_plugins/"目录中

        // 下载配置文件中的所有插件的默认版本
        //pluginDownloader.downloadAll();

        // 插件升级
        // 在升级的过程中，会先对旧的插件进行备份，备份文件名称后缀为.old；等到插件更新完毕后，会统一删除旧的插件包
        // 若插件更新中断，用户可以从.old文件恢复旧版插件
//        pluginDownloader.upgrade();
    }
}