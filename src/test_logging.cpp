// test_logging.cpp

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <vector>
#include <future>
#include <iostream>
#include <thread>
#include <chrono>

#include "thread_pool.h"
#include "thread_safe_queue.h"
#include "cout_redirector.h"
#include "logger.h"

namespace py = pybind11;

// Function to perform a simple computation and log messages
std::vector<int> test_logging_function(int num_tasks, py::function progress_callback, py::function log_callback) {
    // Initialize Logger and CoutRedirector
    ThreadSafeQueue queue;
    Logger logger(queue, log_callback);
    CoutRedirector redirector(queue);

    std::cerr << "Starting test_logging_function with " << num_tasks << " tasks." << std::endl;

    // Initialize ThreadPool
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;

    for(int i = 0; i < num_tasks; ++i) {
        std::cerr << "Enqueuing task " << i << std::endl;
        futures.emplace_back(pool.enqueue([i]() -> int {
            std::cout << "Task " << i << " is running" << std::endl;
            // Simulate computation
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "Task " << i << " completed" << std::endl;
            return i * i;
        }));
        if(progress_callback) {
            try {
                py::gil_scoped_acquire acquire;  // Acquire GIL
                progress_callback();
            }
            catch (const py::error_already_set& e) {
                std::cerr << "Python error in progress_callback: " << e.what() << std::endl;
            }
        }
    }

    std::cerr << "All tasks enqueued." << std::endl;

    // Collect results
    std::vector<int> results;
    for(auto &fut : futures) {
        results.emplace_back(fut.get());
    }

    std::cerr << "All tasks completed. Results collected." << std::endl;

    // No need to call queue.set_finished(); Logger destructor handles it

    return results;
}

PYBIND11_MODULE(test_logging, m) {
    m.def("test_logging_function", &test_logging_function, "Test logging with ThreadPool",
          py::arg("num_tasks"),
          py::arg("progress_callback"),
          py::arg("log_callback"));
}
