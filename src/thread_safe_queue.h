#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <memory>

template <typename T>
class ThreadSafeQueue {
public:
    // Constructor with optional maximum size (0 means unlimited)
    explicit ThreadSafeQueue(size_t max_size = 0)
        : finished(false), max_size(max_size) {}

    // Disable copy and assignment
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    // Push a new item into the queue
    // Blocks if the queue is full (only if max_size > 0)
    bool push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        if (finished) {
            std::cerr << "ThreadSafeQueue: Push attempted after set_finished.\n";
            return false;
        }

        // Wait until there's space in the queue or the queue is finished
        cv_not_full.wait(lock, [this]() { return q.size() < max_size || finished; });

        if (finished) {
            std::cerr << "ThreadSafeQueue: Push attempted after set_finished.\n";
            return false;
        }

        q.push(item);
        std::cerr << "ThreadSafeQueue: Pushed item.\n";
        cv_not_empty.notify_one();
        return true;
    }

    // Push an item using move semantics
    bool push(T&& item) {
        std::unique_lock<std::mutex> lock(mtx);
        if (finished) {
            std::cerr << "ThreadSafeQueue: Push attempted after set_finished.\n";
            return false;
        }

        // Wait until there's space in the queue or the queue is finished
        cv_not_full.wait(lock, [this]() { return q.size() < max_size || finished; });

        if (finished) {
            std::cerr << "ThreadSafeQueue: Push attempted after set_finished.\n";
            return false;
        }

        q.push(std::move(item));
        std::cerr << "ThreadSafeQueue: Pushed item via move.\n";
        cv_not_empty.notify_one();
        return true;
    }

    // Pop an item from the queue
    // Returns false if the queue is finished and empty
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_empty.wait(lock, [this]() { return !q.empty() || finished; });

        if (q.empty()) {
            std::cerr << "ThreadSafeQueue: No more items to pop. Exiting pop.\n";
            return false;
        }

        item = std::move(q.front());
        q.pop();
        std::cerr << "ThreadSafeQueue: Popped item.\n";
        cv_not_full.notify_one();
        return true;
    }

    // Indicate that no more items will be added to the queue
    void set_finished() {
        std::lock_guard<std::mutex> lock(mtx);
        finished = true;
        std::cerr << "ThreadSafeQueue: set_finished called. Notifying all.\n";
        cv_not_empty.notify_all();
        cv_not_full.notify_all();
    }

    // Check if the queue is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return q.empty();
    }

    // Get the current size of the queue
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return q.size();
    }

private:
    mutable std::mutex mtx;
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
    std::queue<T> q;
    bool finished;
    size_t max_size; // 0 for unlimited
};

#endif // THREAD_SAFE_QUEUE_H