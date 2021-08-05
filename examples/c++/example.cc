// Copyright (c) 2020 Qihoo Inc. All rights reserved.
// File  demo.cpp
// Author 360
// Date   2020/10/14 上午10:42
// Brief

#include <sys/time.h>
#include <unistd.h>

#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>

#include "clink.h"
using namespace std;

void FeatureExtractExample() {
  clink::Clink processor;
  int status = processor.LoadConfig("../../feature_pipeline");
  if (status != 0) {
    std::cout << "load config error" << std::endl;
  }

  std::string data_file = "../../dataset/data.csv";
  std::ifstream ifs;

  std::string line;
  ifs.open(data_file, std::ios::in | std::ios::binary);
  if (ifs) {
  } else {
    std::cout << "fail to open " << data_file << std::endl;
  }
  while (std::getline(ifs, line)) {
    std::vector<uint32_t> indexs;
    std::vector<float> values;
    processor.FeatureExtract(line, &indexs, &values);
    for (int i = 0; i < indexs.size(); ++i) {
      std::cout << indexs.at(i) << ":" << values.at(i) << " ";
    }
    std::cout << endl;
  }
}

void FeatureExtractOfflileExample() {
  int ret = FeatureOfflineInit("", "../../feature_pipeline");

  // clink::Clink processor;
  // int status = processor.LoadConfig("../../feature_pipeline");
  // if (status != 0) {
  //   std::cout << "load config error" << std::endl;
  // }

  std::string data_file = "../../dataset/data.csv";
  std::ifstream ifs;

  std::string line;
  ifs.open(data_file, std::ios::in | std::ios::binary);
  if (ifs) {
  } else {
    std::cout << "fail to open " << data_file << std::endl;
  }
  while (std::getline(ifs, line)) {
    char* output;
    FeatureExtractOffline(line.c_str(), &output);
    std::cout << output << std::endl;
  }
}

int main() { FeatureExtractOfflileExample(); }
