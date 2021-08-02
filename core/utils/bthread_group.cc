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
 

#include "core/utils/bthread_group.h"

namespace clink {

template <typename TCallable>
void BthreadGroup::ForeachFinishedTask(const TCallable& func) {
  for (std::unique_ptr<BthreadTask>& task : tasks_) {
    if (task && task->is_finished_.exchange(true, std::memory_order_seq_cst)) {
      func(task.get());
    } else {
      task.release();
    }
  }
}

namespace {

bvar::Adder<int> g_bthread_groups("bthread_group_count");
bvar::Adder<int> g_bthread_group_timeout_count("bthread_group_timeout_count");
bvar::Adder<int> g_bthread_group_tasks("bthread_group_task_count");

}  // namespace

struct BthreadGroup::SharedPart {
  SharedPart() : ce(0) {}

  bthread::CountdownEvent ce;
};

BthreadGroup::BthreadGroup() : sp_(std::make_shared<SharedPart>()) {
  g_bthread_groups << 1;
}

bool BthreadGroup::AddOneTask(std::unique_ptr<BthreadTask>&& task) {
  if (is_started_) {
    return false;
  }
  task->sp_ = sp_;
  tasks_.emplace_back(std::move(task));
  sp_->ce.add_count();
  return true;
}

void BthreadGroup::Start(int timeout_us) {
  StartBthread();
  is_started_ = true;
  if (timeout_us > 0) {
    const timespec end_time = butil::microseconds_from_now(timeout_us);
    const int status = sp_->ce.timed_wait(end_time);
    if (status == ETIMEDOUT) {
      g_bthread_group_timeout_count << 1;
      LOG(ERROR) << "bthread group is stopped du to: " << berror(status);
    } else if (status != 0) {
      LOG(ERROR) << "bthread group is stopped du to: " << berror(status);
    }
  } else {
    const int status = sp_->ce.wait();
    if (status != 0) {
      LOG(ERROR) << "bthread group is stopped du to: " << berror(status);
    }
  }
  NotifyStop();
}

void BthreadGroup::StartBthread() {
  if (tasks_.empty()) {
    return;
  }

  for (size_t i = 0; i < tasks_.size() - 1; ++i) {
    CHECK_EQ(
        0, bthread_start_background(&tasks_[i]->bthread_, nullptr,
                                    BthreadTask::InternalRun, tasks_[i].get()))
        << "Failed to start a bthread, error: " << berror();
  }
  CHECK_EQ(0,
           bthread_start_urgent(&tasks_.back()->bthread_, nullptr,
                                BthreadTask::InternalRun, tasks_.back().get()))
      << "Failed to start a bthread, error: " << berror();
}

void BthreadGroup::NotifyStop() {
  for (const std::unique_ptr<BthreadTask>& task : tasks_) {
    task->should_stop_.store(true, std::memory_order_release);
  }
}

BthreadGroup::~BthreadGroup() {
  // Ensure the task thread aware of the finish state.
  ForeachFinishedTask([](BthreadTask* dummy) {});
  g_bthread_groups << -1;
}

BthreadGroup::BthreadTask::BthreadTask() { g_bthread_group_tasks << 1; }

BthreadGroup::BthreadTask::~BthreadTask() { g_bthread_group_tasks << -1; }

bool BthreadGroup::BthreadTask::ShouldStop() const {
  return should_stop_.load(std::memory_order_acquire) ||
         1 == bthread_stopped(bthread_self());
}

void* BthreadGroup::BthreadTask::InternalRun(void* data) {
  BthreadTask* task = static_cast<BthreadTask*>(data);
  task->Run();
  if (task->is_finished_.exchange(true, std::memory_order_seq_cst)) {
    delete task;
  } else if (std::shared_ptr<SharedPart> sp = task->sp_.lock()) {
    sp->ce.signal();
  }

  return nullptr;
}

}  // namespace clink
