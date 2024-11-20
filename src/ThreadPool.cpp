#include "ThreadPool.h"
#include <stdexcept>
#include <algorithm>

// Constructor with dynamic thread and queue size determination
ThreadPool::ThreadPool(size_t num_threads, size_t max_queue_size)
    : tasks_(max_queue_size), stop_(false)
{
    // Limit the number of threads to a reasonable maximum (e.g., 32) to prevent excessive resource usage
    size_t hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads == 0) {
        hardware_threads = 4; // Fallback to 4 threads if hardware_concurrency cannot determine
    }
    num_threads = std::min(num_threads, static_cast<size_t>(hardware_threads));

    for(size_t i = 0; i < num_threads; ++i)
        workers_.emplace_back(&ThreadPool::worker_thread, this);
}

ThreadPool::~ThreadPool()
{
    // Signal all threads to stop
    tasks_.set_finished();
    stop_ = true;

    // Join all threads
    for(auto &worker: workers_)
        if(worker.joinable())
            worker.join();
}

void ThreadPool::worker_thread()
{
    while(true)
    {
        std::function<void()> task;
        if(!tasks_.pop(task)) {
            // Queue is finished and empty
            break;
        }
        try {
            task();
        } catch (const std::exception& e) {
            // Handle exceptions from tasks if necessary
            // For example, log them or propagate
            // Since we cannot log here, consider alternative error handling
        }
    }
}