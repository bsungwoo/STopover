#ifndef LOGGER_H
#define LOGGER_H

#include <thread>
#include <string>
#include "thread_safe_queue.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

class Logger {
public:
    Logger(ThreadSafeQueue& queue, py::function callback)
        : queue_(queue), callback_(callback), logger_thread_(&Logger::process, this) {}

    ~Logger() {
        // Signal that no more messages will be added
        queue_.set_finished();
        if (logger_thread_.joinable()) {
            logger_thread_.join();
        }
    }

private:
    void process() {
        std::string msg;
        while (queue_.pop(msg)) {  // Continue until pop returns false
            std::cerr << "Logger processing message: " << msg << std::endl;  // Debug Statement
            try {
                // Acquire GIL before calling Python
                py::gil_scoped_acquire acquire;
                callback_(msg);
            }
            catch (const py::error_already_set& e) {
                // If Python callback fails, log the error
                std::cerr << "Python error in log_callback: " << e.what() << std::endl;
            }
        }
        std::cerr << "Logger thread exiting." << std::endl;  // Debug Statement
    }

    ThreadSafeQueue& queue_;
    py::function callback_;
    std::thread logger_thread_;
};

#endif // LOGGER_H