#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <thread>
#include <future>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>
#include <iostream>
#include "thread_safe_queue.h" // Include the enhanced ThreadSafeQueue

class ThreadPool {
public:
    // Deleted methods to enforce Singleton property
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Static method to get the Singleton instance
    static ThreadPool& getInstance(size_t threads = std::thread::hardware_concurrency());

    // Enqueue a task and return a future
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    // Destructor
    ~ThreadPool();

private:
    // Private constructor for Singleton
    ThreadPool(size_t threads);

    // Worker threads
    std::vector<std::thread> workers;

    // Task queue using ThreadSafeQueue
    ThreadSafeQueue<std::function<void()>> tasks;

    // Atomic flag to stop the pool
    std::atomic<bool> stop;
};

// Implementation

// Singleton accessor
inline ThreadPool& ThreadPool::getInstance(size_t threads) {
    static ThreadPool instance(threads);
    return instance;
}

// Constructor
inline ThreadPool::ThreadPool(size_t threads)
    : stop(false)
{
    if (threads == 0) {
        threads = 4; // Fallback to 4 threads if hardware_concurrency is not available
    }

    for(size_t i = 0;i < threads;++i)
        workers.emplace_back(
            [this]
            {
                while(true)
                {
                    std::function<void()> task;
                    if (tasks.pop(task)) { // Retrieve task from ThreadSafeQueue
                        try {
                            task();
                        } catch(const std::exception& e) {
                            std::cerr << "Task exception: " << e.what() << std::endl;
                        } catch(...) {
                            std::cerr << "Task unknown exception." << std::endl;
                        }
                    } else { // If pop returns false, queue is finished
                        return;
                    }
                }
            }
        );
}

// Destructor
inline ThreadPool::~ThreadPool()
{
    tasks.set_finished(); // Signal all worker threads to finish
    for(std::thread &worker: workers)
        if(worker.joinable())
            worker.join();
}

// Enqueue method
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::invoke_result<F, Args...>::type>
{
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    if (!tasks.push([task]() { (*task)(); })) { // Push task into ThreadSafeQueue
        throw std::runtime_error("enqueue on stopped ThreadPool");
    }
    return res;
}

#endif // THREAD_POOL_H