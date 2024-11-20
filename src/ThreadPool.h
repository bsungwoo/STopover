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
#include <exception>
#include <iostream>

class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool();

    // Enqueue a task and return a future
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    // Workers
    std::vector<std::thread> workers;
    
    // Task queue
    std::queue<std::function<void()>> tasks;
    
    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
};

// Constructor
inline ThreadPool::ThreadPool(size_t threads)
    : stop(false)
{
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this]
            {
                while(true)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    try {
                        task();
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Exception in ThreadPool task: " << e.what() << std::endl;
                        // Optionally, rethrow or handle the exception as needed
                    }
                    catch (...) {
                        std::cerr << "Unknown exception in ThreadPool task." << std::endl;
                        // Optionally, rethrow or handle the exception as needed
                    }
                }
            }
        );
}

// Destructor
inline ThreadPool::~ThreadPool()
{
    stop = true;
    condition.notify_all();
    for(std::thread &worker: workers)
        if(worker.joinable())
            worker.join();
}

// Enqueue method
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
        
    std::future<return_type> res = task->get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex);

        // Don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ 
            try {
                (*task)(); 
            }
            catch (const std::exception& e) {
                std::cerr << "Exception in enqueued task: " << e.what() << std::endl;
                throw; // Re-throw to allow future to capture it
            }
            catch (...) {
                std::cerr << "Unknown exception in enqueued task." << std::endl;
                throw;
            }
        });
    }
    condition.notify_one();
    return res;
}

#endif // THREAD_POOL_H