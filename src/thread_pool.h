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
#include <iostream> // For exception logging (optional)

// ThreadPool Class Definition
class ThreadPool {
public:
    // Constructor with threads and max_queue_size
    ThreadPool(size_t threads, size_t max_queue_size);
    ~ThreadPool();

    // Enqueue a task and return a future
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type>;

private:
    // Workers
    std::vector<std::thread> workers;

    // Task queue
    std::queue<std::function<void()>> tasks;

    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable queue_not_full; // For bounded queue
    size_t max_queue_size;
    std::atomic<bool> stop;
};

// Constructor Implementation
inline ThreadPool::ThreadPool(size_t threads, size_t max_queue)
    : stop(false), max_queue_size(max_queue)
{
    for(size_t i = 0; i < threads; ++i)
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
                        // Notify that there's space in the queue
                        this->queue_not_full.notify_one();
                    }

                    try {
                        task();
                    } catch(const std::exception& e) {
                        // Handle known exceptions
                        std::cerr << "Task exception: " << e.what() << std::endl;
                    } catch(...) {
                        // Handle any other exceptions
                        std::cerr << "Task unknown exception." << std::endl;
                    }
                }
            }
        );
}

// Destructor Implementation
inline ThreadPool::~ThreadPool()
{
    stop = true;
    condition.notify_all();
    queue_not_full.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

// Enqueue Method Implementation
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
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Wait until the queue has space or the pool is stopping
        queue_not_full.wait(lock, [this]{ return this->tasks.size() < this->max_queue_size || this->stop; });

        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

#endif // THREAD_POOL_H