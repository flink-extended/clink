package com.feature.common;

import com.alibaba.alink.common.AlinkGlobalConfiguration;
import com.alibaba.alink.common.io.plugin.PluginDownloader;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class PluginHandler {
    public void pluginPrepare() throws Exception {
        // Setting plugins' downloading directory, will create it if not existed
        String currentDirectory;
        File file = new File(".");
        currentDirectory = file.getAbsolutePath();
        AlinkGlobalConfiguration.setPluginDir(currentDirectory + "alink_plugins");

        // Acquiring plugin-downloader
        PluginDownloader pluginDownloader = AlinkGlobalConfiguration.getPluginDownloader();

        // Loading plugins' configurations from remote
        pluginDownloader.loadConfig();

        // Displaying plugins' names
        List<String> plugins = pluginDownloader.listAvailablePlugins();
        System.out.println(Arrays.toString(plugins.toArray()));

        // Displaying the selected plugin's version
        String pluginName = plugins.get(0);
        List<String> availableVersions = pluginDownloader.listAvailablePluginVersions(pluginName);

        // Downloading the selected plugin with specific version
        String pluginVersion = availableVersions.get(0);
        pluginDownloader.downloadPlugin(pluginName, pluginVersion);

        // Downloading the selected plugin with default version
        pluginDownloader.downloadPlugin(pluginName);

        // Downloading all plugins with default versions
        pluginDownloader.downloadAll();

        // Upgrade downloaded plugins
        // Legacy plugins will be archived in a .old file, the legacy local plugins' files will be
        // deleted
        // after the upgrade finished. Users can use .old file to recover the legacy plugins
        // pluginDownloader.upgrade();
    }
}
