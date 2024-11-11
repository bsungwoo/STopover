#ifndef LOGGER_SIMPLE_H
#define LOGGER_SIMPLE_H

#include <iostream>
#include <thread>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class LoggerSimple {
public:
    LoggerSimple(py::function callback)
        : callback_(callback), finished_(false), logger_thread_(&LoggerSimple::process, this) {}

    ~LoggerSimple() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            finished_ = true;
            cv_.notify_all();
        }
        if (logger_thread_.joinable()) {
            logger_thread_.join();
        }
    }

    void log(const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            q_.push(message);
        }
        cv_.notify_one();
    }

private:
    void process() {
        std::string message;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this]{ return !q_.empty() || finished_; });
                if (finished_ && q_.empty()) break;
                message = q_.front();
                q_.pop();
            }
            try {
                py::gil_scoped_acquire acquire;
                callback_(message);
            }
            catch (const py::error_already_set& e) {
                std::cerr << "Python error in log_callback: " << e.what() << std::endl;
            }
        }
    }

    py::function callback_;
    std::queue<std::string> q_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool finished_;
    std::thread logger_thread_;
};

#endif // LOGGER_SIMPLE_H