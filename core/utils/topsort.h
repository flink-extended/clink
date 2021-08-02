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
 
#ifndef CORE_UTILS_TOPSORT_H_
#define CORE_UTILS_TOPSORT_H_
#include <butil/logging.h>

#include <memory>
#include <queue>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace clink {
namespace utils {

template <typename T, typename = typename std::enable_if<
                          std::is_integral<T>::value ||
                          std::is_same<std::string, T>::value>::type>
class TopSort {
 public:
  TopSort() { Reset(); }

  virtual ~TopSort() { Reset(); }

  void Reset() {
    completed_ = false;
    degree_.clear();
    edges_.clear();
    sequence_.clear();
  }

  void AddRelation(const T& from, const T& to) {
    auto iter = edges_.find(from);
    if (iter != edges_.end()) {
      if (from != to) {
        iter->second->emplace(to);
      }
    } else {
      auto new_set = std::make_shared<std::set<T>>();
      if (from != to) {
        new_set->emplace(to);
      }
      edges_[from] = std::move(new_set);
    }
  }

  bool BfsTraversal() {
    //计算所有节点的入度
    for (auto& iter : edges_) {
      if (degree_.find(iter.first) == degree_.end()) {
        degree_[iter.first] = 0;
      }
      for (auto& it : *(iter.second)) {
        ++degree_[it];
      }
    }

    //入度为0的节点入队
    std::queue<T> queue;
    for (auto iter = degree_.begin(); iter != degree_.end();) {
      if (iter->second == 0) {
        queue.emplace(iter->first);
        iter = degree_.erase(iter);
      } else {
        ++iter;
      }
    }
    while (!queue.empty()) {
      int cur_level_size = queue.size();
      std::vector<T> vec;
      for (int i = 0; i < cur_level_size; ++i) {
        T& top_node = queue.front();
        queue.pop();
        vec.emplace_back(top_node);
        auto iter = edges_.find(top_node);
        if (iter == edges_.end()) {
          continue;
        }

        for (auto& item : *(iter->second)) {
          --degree_[item];
          if (degree_[item] == 0) {
            queue.emplace(item);
            degree_.erase(item);
          }
        }
      }
      sequence_.emplace_back(vec);
    }

    if (!degree_.empty()) {
      LOG(ERROR) << "topsort error,cyclic topology";
      //有特征未遍历到,无法生成拓扑图
      sequence_.clear();
      completed_ = false;
    } else {
      completed_ = true;
    }
    return completed_;
  }

  void GetSerialSequence(std::vector<T>* output) {
    if (!completed_ && !BfsTraversal()) {
      return;
    }

    for (auto& item : sequence_) {
      std::copy(item.begin(), item.end(), std::back_inserter(*output));
    }
  }

  const std::vector<std::vector<T>>& GetParallelSequence() { return sequence_; }

 private:
  //有向图的边
  std::unordered_map<T, std::shared_ptr<std::set<T>>> edges_;
  std::unordered_map<T, int> degree_;
  std::vector<std::vector<T>> sequence_;  //可并行的顺序
  bool completed_;
};
}  // namespace utils
}  // namespace clink
#endif