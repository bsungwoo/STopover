#include "ThreadPool.h"
#include <stdexcept>

ThreadPool::ThreadPool(size_t num_threads, size_t max_queue_size)
    : tasks_(max_queue_size), stop_(false)
{
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
        task();
    }
}