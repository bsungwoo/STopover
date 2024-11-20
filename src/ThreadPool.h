#pragma once

#include <vector>
#include <thread>
#include <functional>
#include <future>
#include <memory>
#include "ThreadSafeQueue.h"

class ThreadPool {
public:
    // Get the singleton instance of ThreadPool
    static ThreadPool& getInstance(size_t num_threads = std::thread::hardware_concurrency(), size_t max_queue_size = 1000) {
        static ThreadPool instance(num_threads, max_queue_size);
        return instance;
    }
    
    // Delete copy and move constructors and assignment operators
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
    
    // Enqueue a task. Blocks if the queue is full.
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    // Destructor
    ~ThreadPool();
    
private:
    ThreadPool(size_t num_threads, size_t max_queue_size);
    
    // Vector of worker threads
    std::vector<std::thread> workers_;
    
    // Bounded task queue
    ThreadSafeQueue<std::function<void()>> tasks_;
    
    // Synchronization
    bool stop_;
    
    // Worker thread function
    void worker_thread();
};