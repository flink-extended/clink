/* Copyright (c) 2021, Qihoo, Inc.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/

#ifndef CORE_UTILS_TOP_SORT_H_
#define CORE_UTILS_TOP_SORT_H_
#include <iostream>
#include <list>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
namespace clink {
namespace top_sort {
template <typename T>
using Relation = std::unordered_map<T, std::unordered_set<T>>;

template <typename T>
bool BfsTraverse(const Relation<T>& feature_relation, std::vector<T>& res) {
  res.clear();
  if (feature_relation.empty()) {
    return false;
  }
  //检查所有节点的入度
  std::unordered_map<T, int> degree;
  for (auto& iter : feature_relation) {
    if (degree.find(iter.first) == degree.end()) {
      degree.insert(std::make_pair(iter.first, 0));
    }
    for (auto& it : iter.second) {
      if (degree.find(it) == degree.end()) {
        degree.insert(std::make_pair(it, 1));
      } else {
        degree[it]++;
      }
    }
  }
  std::queue<T> queue;
  for (auto it = degree.begin(); it != degree.end();) {
    if (it->second == 0) {
      queue.emplace(it->first);
      it = degree.erase(it);
    } else {
      ++it;
    }
  }
  while (!queue.empty()) {
    T& top_node = queue.front();
    res.emplace_back(top_node);
    queue.pop();
    auto child_set = feature_relation.find(top_node);
    if (child_set == feature_relation.end()) {
      continue;
    }
    for (auto& iter : child_set->second) {
      --degree[iter];
      if (degree[iter] == 0) {
        queue.emplace(iter);
        degree.erase(iter);
      }
    }
  }
  if (!degree.empty()) {
    //有特征未遍历到,无法生成拓扑图
    res.clear();
    return false;
  }
  return true;
}
}  // namespace top_sort

}  // namespace clink
#endif  // CORE_UTILS_TOP_SORT_H_
