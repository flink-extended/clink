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
 

#ifndef CORE_UTILS_BTHREAD_GROUP_H
#define CORE_UTILS_BTHREAD_GROUP_H

#include <bthread/bthread.h>
#include <bthread/countdown_event.h>

#include <memory>

namespace clink {

// This class is NOT thread safe.
class BthreadGroup {
 public:
  class BthreadTask;

  BthreadGroup();
  ~BthreadGroup();

  // Always return true if call it before "Start()", return false otherwise.
  bool AddOneTask(std::unique_ptr<BthreadTask>&& task);

  void Start(int timeout_us = 0);

  // The signature of TCallable is " void (BthreadTask*)".
  // Note: if u start the group with a timeout, then use this function to
  // retrieve your tasks, or there is a race condition.
  template <typename TCallable>
  void ForeachFinishedTask(const TCallable& func);

 private:
  struct SharedPart;

  void StartBthread();

  // This just set a "stop flag" simply, and the user logic should stop
  // according to BthreadTask::ShouldStop().
  void NotifyStop();

  bool is_started_ = false;
  std::shared_ptr<SharedPart> sp_;
  std::vector<std::unique_ptr<BthreadTask>> tasks_;
};

class BthreadGroup::BthreadTask {
 public:
  BthreadTask();
  virtual ~BthreadTask();

  virtual void Run() = 0;

 protected:
  // User should stop the task if this function return true. User can also
  // ignore this function, and then this task will stop and delete itself.
  bool ShouldStop() const;

 private:
  friend class BthreadGroup;

  static void* InternalRun(void* data);

  std::atomic<bool> is_finished_{false};
  std::atomic<bool> should_stop_{false};
  bthread_t bthread_ = INVALID_BTHREAD;
  std::weak_ptr<BthreadGroup::SharedPart> sp_;
};

}  // namespace gutil
#endif  // GAIA_GUTIL_BTHREAD_GROUP_H
