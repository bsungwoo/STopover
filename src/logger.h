#ifndef LOGGER_H
#define LOGGER_H

#include <thread>
#include <atomic>
#include "thread_safe_queue.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

class Logger {
public:
    Logger(ThreadSafeQueue& queue, py::function callback)
        : queue_(queue), callback_(callback), stop_flag_(false) {
        logger_thread_ = std::thread(&Logger::process, this);
    }

    ~Logger() {
        stop_flag_ = true;
        queue_.set_finished();
        if (logger_thread_.joinable()) {
            logger_thread_.join();
        }
    }

private:
    void process() {
        std::string msg;
        while (!stop_flag_) {
            if (queue_.pop(msg)) {
                // Acquire GIL before calling Python
                py::gil_scoped_acquire acquire;
                callback_(msg);
            }
        }
    }

    ThreadSafeQueue& queue_;
    py::function callback_;
    std::thread logger_thread_;
    std::atomic<bool> stop_flag_;
};

#endif // LOGGER_H