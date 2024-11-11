#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>

class ThreadSafeQueue {
public:
    ThreadSafeQueue() : finished(false) {}

    // Push a new message into the queue
    void push(const std::string& message) {
        std::lock_guard<std::mutex> lock(mtx);
        q.push(message);
        std::cerr << "ThreadSafeQueue: Pushed message: " << message;
        cv.notify_one();
    }

    // Pop a message from the queue; returns false if finished and queue is empty
    bool pop(std::string& message) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return !q.empty() || finished; });
        if (!q.empty()) {
            message = q.front();
            q.pop();
            std::cerr << "ThreadSafeQueue: Popped message: " << message;
            return true;
        }
        std::cerr << "ThreadSafeQueue: No more messages. Exiting pop." << std::endl;
        return false;
    }

    // Indicate that no more messages will be added
    void set_finished() {
        std::lock_guard<std::mutex> lock(mtx);
        finished = true;
        std::cerr << "ThreadSafeQueue: set_finished called. Signaling all waiting threads." << std::endl;
        cv.notify_all();
    }

private:
    std::queue<std::string> q;
    std::mutex mtx;
    std::condition_variable cv;
    bool finished;
};

#endif // THREAD_SAFE_QUEUE_H