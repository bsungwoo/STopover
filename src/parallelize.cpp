#include "parallelize.h"
#include "topological_comp.h"
#include "jaccard.h"
#include "thread_pool.h"

#include <iostream>
#include <fstream>
#include "thread_safe_queue.h"
#include "custom_streambuf.h"
#include "cout_redirector.h"
#include "logger.h"

#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>        // For automatic conversion of STL containers
#include <pybind11/numpy.h>      // For handling NumPy arrays
#include <future>                // For std::future
#include <vector>
#include <utility>               // For std::pair

namespace py = pybind11;

Eigen::VectorXd array_to_vector(const py::array_t<double>& array) {
    // Request a contiguous buffer
    py::buffer_info buf = array.request();
    
    // Ensure the array is one-dimensional
    if (buf.ndim != 1) {
        throw std::invalid_argument("All input arrays must be one-dimensional.");
    }
    
    size_t size = buf.shape[0];
    const double* data_ptr = static_cast<const double*>(buf.ptr);

    // Copy data into Eigen::VectorXd
    Eigen::VectorXd vec(size);
    std::memcpy(vec.data(), data_ptr, size * sizeof(double));

    return vec;
}

Eigen::MatrixXd array_to_matrix(const py::array_t<double>& array) {
    // Request a buffer (ensure it's contiguous)
    py::buffer_info buf = array.request();
    
    // Ensure the array is two-dimensional
    if (buf.ndim != 2) {
        throw std::invalid_argument("All input arrays must be two-dimensional.");
    }
    
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    const double* data_ptr = static_cast<const double*>(buf.ptr);

    // Copy data into Eigen::MatrixXd
    // Need to handle the memory layout (NumPy is row-major, Eigen is column-major by default)
    // Map the data with RowMajor and then copy to a column-major matrix
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_map(data_ptr, rows, cols);
    Eigen::MatrixXd mat = mat_map;

    return mat;
}

std::vector<Eigen::VectorXd> parallel_topological_comp(
    const std::vector<py::array_t<double>>& locs,
    const std::string& spatial_type, double fwhm,
    const std::vector<py::array_t<double>>& feats,
    int min_size, int thres_per, const std::string& return_mode, int num_workers,
    py::function progress_callback,
    py::function log_callback) {

    // Pre-convert locs and feats to Eigen types by copying data
    std::vector<Eigen::MatrixXd> locs_eigen;
    std::vector<Eigen::VectorXd> feats_eigen;

    locs_eigen.reserve(locs.size());
    feats_eigen.reserve(feats.size());

    for (size_t i = 0; i < feats.size(); ++i) {
        // Convert locs[i] to Eigen::MatrixXd
        Eigen::MatrixXd loc = array_to_matrix(locs[i]);
        locs_eigen.push_back(loc);

        // Convert feats[i] to Eigen::VectorXd
        Eigen::VectorXd feat = array_to_vector(feats[i]);
        feats_eigen.push_back(feat);
    }

    // Initialize ThreadPool
    ThreadPool pool(num_workers);
    std::vector<std::future<std::pair<size_t, Eigen::VectorXd>>> results;

    // Dispatch parallel tasks
    for (size_t i = 0; i < feats.size(); ++i) {
        size_t index = i;
        Eigen::MatrixXd loc = locs_eigen[i];
        Eigen::VectorXd feat = feats_eigen[i];

        // Enqueue the task
        results.emplace_back(pool.enqueue([=]() -> std::pair<size_t, Eigen::VectorXd> {
            Eigen::VectorXd res = topological_comp_res(loc, spatial_type, fwhm, feat, min_size, thres_per, return_mode);
            return {index, res};
        }));

        // Call the progress callback
        if (progress_callback) {
            progress_callback();
        }
    }

    // Collect the results in the correct order
    std::vector<Eigen::VectorXd> output(feats.size());
    for (auto& result_future : results) {
        auto result_pair = result_future.get();
        size_t index = result_pair.first;
        Eigen::VectorXd res = result_pair.second;
        output[index] = res;
    }

    return output;
}

// Updated parallel_jaccard_composite function to handle lists of NumPy arrays
std::vector<double> parallel_jaccard_composite_py(
    py::list CCx_loc_sums_list,
    py::list CCy_loc_sums_list,
    py::list feat_xs_list,
    py::list feat_ys_list,
    const std::string& jaccard_type,
    int num_workers,
    py::function progress_callback,
    py::function log_callback) {

    // Create a thread-safe queue and initialize the logger with log_callback
    ThreadSafeQueue queue;
    Logger logger(queue, log_callback);

    // Create a CoutRedirector instance to redirect std::cout to the queue
    CoutRedirector redirector(queue);

    // Check that all lists have the same size
    size_t list_size = CCx_loc_sums_list.size();
    if (CCy_loc_sums_list.size() != list_size ||
        feat_xs_list.size() != list_size ||
        feat_ys_list.size() != list_size) {
        throw std::invalid_argument("All input lists must have the same length.");
    }

    // Convert each NumPy array in the lists to Eigen::VectorXd
    std::vector<Eigen::VectorXd> CCx_loc_sums_vec;
    std::vector<Eigen::VectorXd> CCy_loc_sums_vec;
    std::vector<Eigen::VectorXd> feat_xs_vec;
    std::vector<Eigen::VectorXd> feat_ys_vec;

    CCx_loc_sums_vec.reserve(list_size);
    CCy_loc_sums_vec.reserve(list_size);
    feat_xs_vec.reserve(list_size);
    feat_ys_vec.reserve(list_size);

    for (size_t i = 0; i < list_size; ++i) {
        try {
            // Convert each array to Eigen::VectorXd
            Eigen::VectorXd CCx_sum = array_to_vector(CCx_loc_sums_list[i].cast<py::array_t<double>>());
            Eigen::VectorXd CCy_sum = array_to_vector(CCy_loc_sums_list[i].cast<py::array_t<double>>());
            Eigen::VectorXd feat_x = array_to_vector(feat_xs_list[i].cast<py::array_t<double>>());
            Eigen::VectorXd feat_y = array_to_vector(feat_ys_list[i].cast<py::array_t<double>>());

            CCx_loc_sums_vec.push_back(CCx_sum);
            CCy_loc_sums_vec.push_back(CCy_sum);
            feat_xs_vec.push_back(feat_x);
            feat_ys_vec.push_back(feat_y);
        }
        catch (const py::cast_error& e) {
            throw std::invalid_argument("All elements in input lists must be NumPy arrays of type float64.");
        }
    }

    ThreadPool pool(num_workers);
    std::vector<std::future<std::pair<size_t, double>>> results;

    // Dispatch parallel tasks
    for (size_t i = 0; i < list_size; ++i) {
        // Capture index and data by value
        size_t index = i;
        Eigen::VectorXd CCx_sum = CCx_loc_sums_vec[i];
        Eigen::VectorXd CCy_sum = CCy_loc_sums_vec[i];
        Eigen::VectorXd feat_x = feat_xs_vec[i];
        Eigen::VectorXd feat_y = feat_ys_vec[i];

        results.emplace_back(pool.enqueue([=]() -> std::pair<size_t, double> {
            // Perform computations using C++ types
            double jaccard_index; // Declare outside of if-else to ensure scope
            if (jaccard_type == "default") {
                jaccard_index = jaccard_composite(CCx_sum, CCy_sum, nullptr, nullptr);
            } else if (jaccard_type == "weighted") {
                jaccard_index = jaccard_composite(CCx_sum, CCy_sum, &feat_x, &feat_y);
            } else {
                throw std::invalid_argument("Invalid jaccard_type: " + jaccard_type);
            }
            return {index, jaccard_index};
        }));

        // Update progress
        if (progress_callback) {
            progress_callback();
        }
    }

    // Collect the results
    std::vector<double> output(list_size, 0.0);
    for (auto& result_future : results) {
        auto result_pair = result_future.get();
        size_t index = result_pair.first;
        double value = result_pair.second;
        output[index] = value;
    }

    return output;
}

// Function to perform a simple computation with LoggerSimple only
std::vector<int> test_logging_function(int num_tasks, py::function progress_callback, py::function log_callback) {
    // Initialize LoggerSimple only
    LoggerSimple logger(log_callback);

    logger.log("Starting test_logging_function with " + std::to_string(num_tasks) + " tasks.\n");

    // Initialize ThreadPool
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;

    for(int i = 0; i < num_tasks; ++i) {
        logger.log("Enqueuing task " + std::to_string(i) + "\n");
        futures.emplace_back(pool.enqueue([i, &logger]() -> int {
            logger.log("Task " + std::to_string(i) + " is running\n");
            // Simulate computation
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            logger.log("Task " + std::to_string(i) + " completed\n");
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

    logger.log("All tasks enqueued.\n");

    // Collect results
    std::vector<int> results;
    for(auto &fut : futures) {
        try {
            results.emplace_back(fut.get());
        }
        catch (const std::exception& e) {
            logger.log("Exception while getting result: " + std::string(e.what()) + "\n");
            throw;
        }
    }

    logger.log("All tasks completed. Results collected.\n");

    return results;
}


// Expose to Python via Pybind11
PYBIND11_MODULE(parallelize, m) {  // Module name within the STopover package
    m.def("parallel_topological_comp", &parallel_topological_comp, "Parallelized topological_comp_res function",
          py::arg("locs"), py::arg("spatial_type") = "visium", py::arg("fwhm") = 2.5, py::arg("feats"), 
          py::arg("min_size") = 5, py::arg("thres_per") = 30, py::arg("return_mode") = "all", 
          py::arg("num_workers") = 4, py::arg("progress_callback"), py::arg("log_callback"));

    m.def("parallel_jaccard_composite", &parallel_jaccard_composite_py, "Parallelized jaccard_composite function accepting lists of NumPy arrays",
          py::arg("CCx_loc_sums"), py::arg("CCy_loc_sums"), py::arg("feat_xs"), py::arg("feat_ys"), 
          py::arg("jaccard_type") = "default", py::arg("num_workers") = 4, py::arg("progress_callback"), py::arg("log_callback"));

    m.def("test_logging_function", &test_logging_function, "Test logging with ThreadPool",
          py::arg("num_tasks"),
          py::arg("progress_callback"),
          py::arg("log_callback"));
}