// ThreadSafeQueue.h
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

// A thread-safe queue with bounded capacity
template<typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(size_t max_size) : max_size_(max_size) {}
    
    // Pushes an item into the queue. Blocks if the queue is full.
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cond_not_full_.wait(lock, [this]() { return queue_.size() < max_size_ || stop_; });
        if (stop_) {
            throw std::runtime_error("ThreadSafeQueue is stopping. Cannot push new items.");
        }
        queue_.push(item);
        cond_not_empty_.notify_one();
    }
    
    // Pops an item from the queue. Blocks if the queue is empty.
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cond_not_empty_.wait(lock, [this]() { return !queue_.empty() || stop_; });
        if (queue_.empty()) {
            return false; // Stop has been called and queue is empty
        }
        item = std::move(queue_.front());
        queue_.pop();
        cond_not_full_.notify_one();
        return true;
    }
    
    // Signals the queue to stop processing. Unblocks any waiting threads.
    void set_finished() {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
        cond_not_empty_.notify_all();
        cond_not_full_.notify_all();
    }
    
    // Returns the current size of the queue
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.size();
    }

private:
    mutable std::mutex mtx_;
    std::condition_variable cond_not_empty_;
    std::condition_variable cond_not_full_;
    std::queue<T> queue_;
    size_t max_size_;
    bool stop_ = false;
};