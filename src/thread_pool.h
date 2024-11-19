// thread_pool.h
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <thread>
#include <queue>
#include <future>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>
#include <iostream>

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

    // Task queue
    std::queue<std::function<void()>> tasks;

    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;

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

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop.load() || !this->tasks.empty(); });
                        if(this->stop.load() && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    try {
                        task();
                    } catch(const std::exception& e) {
                        std::cerr << "Task exception: " << e.what() << std::endl;
                    } catch(...) {
                        std::cerr << "Task unknown exception." << std::endl;
                    }
                }
            }
        );
}

// Destructor
inline ThreadPool::~ThreadPool()
{
    stop.store(true);
    condition.notify_all();
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
    {
        std::lock_guard<std::mutex> lock(queue_mutex);

        if(stop.load())
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

#endif // THREAD_POOL_H