#pragma once

#include <vector>
#include <thread>
#include <functional>
#include <future>
#include <memory>
#include "ThreadSafeQueue.h"

class ThreadPool {
public:
    // Constructor with dynamic thread and queue size determination
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency(),
               size_t max_queue_size = 2 * std::thread::hardware_concurrency());

    // Delete copy and move constructors and assignment operators
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Enqueue a task. Blocks if the queue is full.
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        // Package the task
        auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();

        // Enqueue the task. This will block if the queue is full.
        tasks_.push([task](){ (*task)(); });

        return res;
    }

    // Destructor
    ~ThreadPool();

private:
    // Vector of worker threads
    std::vector<std::thread> workers_;

    // Bounded task queue
    ThreadSafeQueue<std::function<void()>> tasks_;

    // Synchronization
    bool stop_;

    // Worker thread function
    void worker_thread();
};