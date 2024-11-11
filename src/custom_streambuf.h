#ifndef CUSTOM_STREAMBUF_H
#define CUSTOM_STREAMBUF_H

#include <streambuf>
#include <string>
#include <mutex>
#include "thread_safe_queue.h"

class CustomStreamBuf : public std::streambuf {
public:
    CustomStreamBuf(ThreadSafeQueue& queue) : queue_(queue) {}

protected:
    // Override the overflow method to capture each character
    virtual int overflow(int c) override {
        if (c != EOF) {
            std::lock_guard<std::mutex> lock(mutex_);
            buffer_ += static_cast<char>(c);
            if (c == '\n') {
                queue_.push(buffer_);
                buffer_.clear();
            }
        }
        return c;
    }

    // Optionally, override sync if needed
    virtual int sync() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!buffer_.empty()) {
            queue_.push(buffer_);
            buffer_.clear();
        }
        return 0;
    }

private:
    ThreadSafeQueue& queue_;
    std::string buffer_;
    std::mutex mutex_;
};

#endif // CUSTOM_STREAMBUF_H