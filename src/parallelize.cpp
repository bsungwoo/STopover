#include "parallelize.h"
#include "topological_comp.h"
#include "jaccard.h"
#include "ThreadPool.h"

#include <iostream>
#include <fstream>

#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>        // For automatic conversion of STL containers
#include <pybind11/numpy.h>      // For handling NumPy arrays
#include <future>                // For std::future
#include <vector>
#include <utility>               // For std::pair
#include <functional>
#include <mutex>

namespace py = pybind11;

Eigen::VectorXd array_to_vector(const py::array_t<double>& array) {
    // Ensure the array is one-dimensional
    if (array.ndim() != 1) {
        throw std::invalid_argument("Input array must be one-dimensional.");
    }

    // Request a buffer descriptor from the NumPy array
    py::buffer_info buf = array.request();

    // Check if the array data type is double
    if (buf.format != py::format_descriptor<double>::format()) {
        throw std::invalid_argument("Input array must be of type float64.");
    }

    size_t size = buf.shape[0];
    double* data_ptr = static_cast<double*>(buf.ptr);
    ssize_t stride = buf.strides[0] / sizeof(double);

    // Create an Eigen::Map with custom stride
    typedef Eigen::Stride<Eigen::Dynamic, 1> StrideType;
    Eigen::Map<Eigen::VectorXd, 0, StrideType> vec(
        data_ptr, size, StrideType(stride, 1));

    // Make a copy of the vector to return
    return Eigen::VectorXd(vec);
}


Eigen::MatrixXd array_to_matrix(const py::array_t<double>& array) {
    if (array.ndim() != 2) {
        throw std::invalid_argument("Input array must be two-dimensional.");
    }

    py::buffer_info buf = array.request();
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];

    double* data_ptr = static_cast<double*>(buf.ptr);
    ssize_t stride_row = buf.strides[0] / sizeof(double);
    ssize_t stride_col = buf.strides[1] / sizeof(double);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        mat(data_ptr, rows, cols, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(stride_row, stride_col));

    return Eigen::MatrixXd(mat); // Make a deep copy if necessary
}

std::vector<Eigen::VectorXd> parallel_topological_comp(
    const std::vector<py::array_t<double>>& locs,
    const std::string& spatial_type, double fwhm,
    const std::vector<py::array_t<double>>& feats,
    int min_size, double thres_per, const std::string& return_mode,
    int num_workers,
    py::function progress_callback,
    py::function log_callback) 
{
    if (locs.size() != feats.size()) {
        throw std::invalid_argument("The size of 'locs' and 'feats' must be equal.");
    }

    size_t total_tasks = feats.size();
    std::vector<Eigen::MatrixXd> locs_eigen;
    std::vector<Eigen::VectorXd> feats_eigen;

    locs_eigen.reserve(locs.size());
    feats_eigen.reserve(feats.size());

    for (size_t i = 0; i < feats.size(); ++i) {
        locs_eigen.emplace_back(array_to_matrix(locs[i]));
        feats_eigen.emplace_back(array_to_vector(feats[i]));
    }

    // Initialize ThreadPool
    ThreadPool& pool = ThreadPool::getInstance(num_workers, 1000); // max_queue_size=1000

    std::vector<std::future<std::pair<size_t, Eigen::VectorXd>>> results;
    results.reserve(total_tasks);

    std::vector<Eigen::VectorXd> output(total_tasks);

    for (size_t i = 0; i < total_tasks; ++i) {
        size_t index = i;
        const Eigen::MatrixXd& loc = locs_eigen[i];
        const Eigen::VectorXd& feat = feats_eigen[i];

        // Enqueue the task
        results.emplace_back(
            pool.enqueue([=, &spatial_type, &return_mode, &progress_callback, &log_callback]() -> std::pair<size_t, Eigen::VectorXd> {
                try {
                    // Release GIL during computation
                    py::gil_scoped_release release;

                    Eigen::VectorXd res = topological_comp_res(loc, spatial_type, fwhm, feat, min_size, thres_per, return_mode);

                    // Acquire GIL before invoking Python callbacks
                    {
                        std::lock_guard<std::mutex> lock(callback_mutex);
                        if (progress_callback) {
                            py::gil_scoped_acquire acquire;
                            progress_callback();
                        }
                    }

                    return {index, res};
                }
                catch (const std::exception& e) {
                    // Log exception
                    {
                        std::lock_guard<std::mutex> lock(callback_mutex);
                        if (log_callback) {
                            py::gil_scoped_acquire acquire;
                            log_callback(std::string("topological_comp_res exception: ") + e.what());
                        }
                    }
                    throw; // Re-throw to be handled later
                }
                catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(callback_mutex);
                        if (log_callback) {
                            py::gil_scoped_acquire acquire;
                            log_callback("topological_comp_res encountered an unknown exception.");
                        }
                    }
                    throw;
                }
            })
        );
    }

    // Collect results
    for (auto& result_future : results) {
        try {
            auto result_pair = result_future.get();
            size_t index = result_pair.first;
            Eigen::VectorXd res = result_pair.second;
            output[index] = res;
        }
        catch (const std::exception& e) {
            // Handle task exceptions
            {
                std::lock_guard<std::mutex> lock(callback_mutex);
                if (log_callback) {
                    py::gil_scoped_acquire acquire;
                    log_callback(std::string("Task exception: ") + e.what());
                }
            }
            // Depending on requirements, decide how to handle failed tasks
            // For example, skip, fill with default values, or re-throw
        }
    }

    return output;
}

std::vector<double> parallel_jaccard_composite(
    const std::vector<py::array_t<double>>& CCx_loc_sums,
    const std::vector<py::array_t<double>>& CCy_loc_sums,
    const std::vector<py::array_t<double>>& feat_xs,
    const std::vector<py::array_t<double>>& feat_ys,
    const std::string& jaccard_type,
    int num_workers,
    py::function progress_callback,
    py::function log_callback) 
{
    size_t total_tasks = CCx_loc_sums.size();
    std::vector<Eigen::VectorXd> CCx_eigen;
    std::vector<Eigen::VectorXd> CCy_eigen;
    std::vector<Eigen::VectorXd> feat_x_eigen;
    std::vector<Eigen::VectorXd> feat_y_eigen;

    CCx_eigen.reserve(total_tasks);
    CCy_eigen.reserve(total_tasks);
    feat_x_eigen.reserve(total_tasks);
    feat_y_eigen.reserve(total_tasks);

    for (size_t i = 0; i < total_tasks; ++i) {
        CCx_eigen.emplace_back(array_to_vector(CCx_loc_sums[i]));
        CCy_eigen.emplace_back(array_to_vector(CCy_loc_sums[i]));
        feat_x_eigen.emplace_back(array_to_vector(feat_xs[i]));
        feat_y_eigen.emplace_back(array_to_vector(feat_ys[i]));
    }

    // Initialize ThreadPool
    ThreadPool& pool = ThreadPool::getInstance(num_workers, 1000); // max_queue_size=1000

    std::vector<std::future<std::pair<size_t, double>>> results;
    results.reserve(total_tasks);

    std::vector<double> output(total_tasks);

    for (size_t i = 0; i < total_tasks; ++i) {
        size_t index = i;
        const Eigen::VectorXd& CCx_sum = CCx_eigen[i];
        const Eigen::VectorXd& CCy_sum = CCy_eigen[i];
        const Eigen::VectorXd& feat_x = feat_x_eigen[i];
        const Eigen::VectorXd& feat_y = feat_y_eigen[i];

        // Enqueue the task
        results.emplace_back(
            pool.enqueue([=, &jaccard_type, &progress_callback, &log_callback]() -> std::pair<size_t, double> {
                try {
                    // Release GIL during computation
                    py::gil_scoped_release release;

                    double res = jaccard_composite(CCx_sum, CCy_sum, &feat_x, &feat_y, jaccard_type);

                    // Acquire GIL before invoking Python callbacks
                    {
                        std::lock_guard<std::mutex> lock(callback_mutex);
                        if (progress_callback) {
                            py::gil_scoped_acquire acquire;
                            progress_callback();
                        }
                    }

                    return {index, res};
                }
                catch (const std::exception& e) {
                    // Log exception
                    {
                        std::lock_guard<std::mutex> lock(callback_mutex);
                        if (log_callback) {
                            py::gil_scoped_acquire acquire;
                            log_callback(std::string("jaccard_composite exception: ") + e.what());
                        }
                    }
                    throw; // Re-throw to be handled later
                }
                catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(callback_mutex);
                        if (log_callback) {
                            py::gil_scoped_acquire acquire;
                            log_callback("jaccard_composite encountered an unknown exception.");
                        }
                    }
                    throw;
                }
            })
        );
    }

    // Collect results
    for (auto& result_future : results) {
        try {
            auto result_pair = result_future.get();
            size_t index = result_pair.first;
            double res = result_pair.second;
            output[index] = res;
        }
        catch (const std::exception& e) {
            // Handle task exceptions
            {
                std::lock_guard<std::mutex> lock(callback_mutex);
                if (log_callback) {
                    py::gil_scoped_acquire acquire;
                    log_callback(std::string("Task exception: ") + e.what());
                }
            }
            // Depending on requirements, decide how to handle failed tasks
            // For example, skip, fill with default values, or re-throw
        }
    }

    return output;
}

PYBIND11_MODULE(parallelize, m) {
    m.doc() = "Parallelize module using C++ ThreadPool and Pybind11";

    m.def("parallel_topological_comp", &parallel_topological_comp, 
          "Parallel topological component extraction",
          py::arg("locs"),
          py::arg("spatial_type"),
          py::arg("fwhm"),
          py::arg("feats"),
          py::arg("min_size"),
          py::arg("thres_per"),
          py::arg("return_mode"),
          py::arg("num_workers"),
          py::arg("progress_callback"),
          py::arg("log_callback"));

    m.def("parallel_jaccard_composite", &parallel_jaccard_composite, 
          "Parallel Jaccard composite index computation",
          py::arg("CCx_loc_sums"),
          py::arg("CCy_loc_sums"),
          py::arg("feat_xs"),
          py::arg("feat_ys"),
          py::arg("jaccard_type"),
          py::arg("num_workers"),
          py::arg("progress_callback"),
          py::arg("log_callback"));
}