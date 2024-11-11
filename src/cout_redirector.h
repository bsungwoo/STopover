#ifndef COUT_REDIRECTOR_H
#define COUT_REDIRECTOR_H

#include <iostream>
#include "custom_streambuf.h"

class CoutRedirector {
public:
    // Constructor accepting a ThreadSafeQueue reference
    CoutRedirector(ThreadSafeQueue& queue) : custom_buf_(queue) {
        original_buf_ = std::cout.rdbuf(&custom_buf_);
    }

    // Destructor restores the original buffer
    ~CoutRedirector() {
        std::cout.rdbuf(original_buf_);
    }

private:
    CustomStreamBuf custom_buf_;
    std::streambuf* original_buf_;
};

#endif // COUT_REDIRECTOR_H