#ifndef COUT_REDIRECTOR_H
#define COUT_REDIRECTOR_H

#include <streambuf>
#include <iostream>
#include <string>
#include "thread_safe_queue.h"

class CoutRedirector : public std::streambuf {
public:
    CoutRedirector(ThreadSafeQueue& queue) : queue_(queue), original_buf_(std::cout.rdbuf(this)) {}
    ~CoutRedirector() {
        std::cout.rdbuf(original_buf_);
    }

protected:
    virtual int overflow(int c) override {
        if (c == EOF) {
            return EOF;
        }
        else {
            buffer_ += static_cast<char>(c);
            if (c == '\n') {
                queue_.push(buffer_);
                buffer_.clear();
            }
            return c;
        }
    }

private:
    ThreadSafeQueue& queue_;
    std::streambuf* original_buf_;
    std::string buffer_;
};

#endif // COUT_REDIRECTOR_H