/*
 Copyright (c) 2012 Jakob Progsch, Václav Zeman

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
   distribution.
 */
#ifndef CORE_UTILS_SIMPLP_THREAD_POOL_H_
#define CORE_UTILS_SIMPLP_THREAD_POOL_H_
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
namespace perception_feature {

enum AffinityType { AFFINITY_DISABLE, AFFINITY_AVERAGE, AFFINITY_SINGLE };
enum TaskStatus { TASK_INITED = 0, TASK_FINISHED, TASK_EXPIRED };

// AffinityType affinity_type = AFFINITY_DISABLE, int core_index = 0)
class SimpleThreadPool {
 public:
  explicit SimpleThreadPool(size_t threads_count)
      : stop_(false), core_index_(0) {
    // unsigned num_cpus = std::thread::hardware_concurrency();

    for (size_t i = 0; i < threads_count; ++i) {
      std::thread t = std::thread([this] {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex_);
            this->condition_.wait(lock, [this] {
              return this->stop_ || !this->task_queue_.empty();
            });
            if (this->stop_ && this->task_queue_.empty()) return;

            task = std::move(this->task_queue_.front());
            this->task_queue_.pop();
          }

          task();
        }
      });

      // if (affinity_type != AFFINITY_DISABLE) {
      //     int index = core_index_;
      //     cpu_set_t cpuset;
      //     CPU_ZERO(&cpuset);
      //     if (affinity_type == AFFINITY_AVERAGE) {
      //         index = core_index_++ % num_cpus;
      //         if (core_index >= 0 && index == core_index) {
      //             index = core_index_++ % num_cpus;
      //         }
      //     } else if (affinity_type == AFFINITY_SINGLE) {
      //         if (index >= num_cpus || index < 0) {
      //             index = 0;
      //         }
      //     }

      //     std::vector<int> cores = { index };
      //     SetThreadAffinity(t, cores);
      // }
      workers_.emplace_back(std::move(t));
    }
  }

  ~SimpleThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
      worker.join();
    }
  }

  SimpleThreadPool(const SimpleThreadPool&) = delete;
  SimpleThreadPool& operator=(const SimpleThreadPool&) = delete;
  SimpleThreadPool(SimpleThreadPool&&) = delete;
  SimpleThreadPool& operator=(SimpleThreadPool&&) = delete;

  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> enqueue(
      F&& f, Args&&... args) {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();

    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      task_queue_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    return res;
  }

  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> enqueue_nolock(
      F&& f, Args&&... args) {
    using return_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> res = task->get_future();
    task_queue_.emplace([task]() { (*task)(); });
    condition_.notify_all();
    return res;
  }

  void Notify() { condition_.notify_one(); }

  std::mutex& queue_mutex() { return queue_mutex_; }

  size_t task_queue_size() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
  }

  size_t task_queue_size_nolock() const { return task_queue_.size(); }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> task_queue_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  std::atomic_bool stop_;
  std::atomic<int> core_index_;
  bool enable_affinity_;
};  // SimpleSimpleThreadPool
}  // namespace perception_feature

#endif  // CORE_UTILS_SIMPLP_THREAD_POOL_H_
