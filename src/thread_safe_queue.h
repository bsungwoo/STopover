#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>

class ThreadSafeQueue {
public:
    // Push a new message into the queue
    void push(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(msg);
        cond_var_.notify_one();
    }

    // Pop a message from the queue. Blocks if the queue is empty.
    bool pop(std::string& msg) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty() && !finished_) {
            cond_var_.wait(lock);
        }
        if (!queue_.empty()) {
            msg = std::move(queue_.front());
            queue_.pop();
            return true;
        }
        return false;
    }

    // Signal that no more messages will be pushed
    void set_finished() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cond_var_.notify_all();
    }

private:
    std::queue<std::string> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    bool finished_ = false;
};

#endif // THREAD_SAFE_QUEUE_H
