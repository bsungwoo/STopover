#ifndef COUT_REDIRECTOR_H
#define COUT_REDIRECTOR_H

#include <streambuf>
#include <iostream>
#include <string>
#include "thread_safe_queue.h"

class CoutRedirector : public std::streambuf {
public:
    CoutRedirector(ThreadSafeQueue& queue) : queue_(queue), original_buf_(std::cout.rdbuf(this)) {
        std::cerr << "CoutRedirector initialized." << std::endl;
    }
    ~CoutRedirector() {
        std::cout.rdbuf(original_buf_);
        std::cerr << "CoutRedirector destructed. Restored original std::cout buffer." << std::endl;
    }

protected:
    virtual int overflow(int c) override {
        if (c == EOF) {
            return EOF;
        }
        else {
            buffer_ += static_cast<char>(c);
            if (c == '\n') {
                std::cerr << "CoutRedirector pushing message: " << buffer_;
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