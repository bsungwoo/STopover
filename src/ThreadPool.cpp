#include "ThreadPool.h"
#include <stdexcept>

ThreadPool::ThreadPool(size_t num_threads, size_t max_queue_size)
    : tasks_(max_queue_size), stop_(false)
{
    for(size_t i = 0;i < num_threads;++i)
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

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
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

// Explicit template instantiation
template auto ThreadPool::enqueue(std::function<void()>&&, ...)->std::future<void>;