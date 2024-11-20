// parallelize.cpp
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
#include <mutex>

namespace py = pybind11;

// Global mutex for synchronizing callback invocations
std::mutex callback_mutex;

// Convert a NumPy array to Eigen::VectorXd
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

// Convert a NumPy array to Eigen::MatrixXd
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

// Parallel function for topological_comp_res
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

    // Automatically determine the number of worker threads if not specified
    if (num_workers <= 0) {
        num_workers = std::min(static_cast<size_t>(4), std::thread::hardware_concurrency());
        if (num_workers == 0) num_workers = 4; // Fallback to 4 threads
    }

    // Initialize ThreadPool with limited number of threads
    ThreadPool pool(num_workers);
    std::vector<std::future<std::pair<size_t, Eigen::VectorXd>>> results;
    results.reserve(total_tasks);

    // Prepare the output vector
    std::vector<Eigen::VectorXd> output(total_tasks);

    for (size_t i = 0; i < total_tasks; ++i) {
        size_t index = i;
        Eigen::MatrixXd loc = locs_eigen[i];
        Eigen::VectorXd feat = feats_eigen[i];

        // Enqueue the task with necessary captures by value
        results.emplace_back(
            pool.enqueue([index, loc, spatial_type, fwhm, feat, min_size, thres_per, return_mode, progress_callback, log_callback]() -> std::pair<size_t, Eigen::VectorXd> {
                try {
                    // Release GIL during computation
                    py::gil_scoped_release release;

                    Eigen::VectorXd res = topological_comp_res(loc, spatial_type, fwhm, feat, min_size, thres_per, return_mode);

                    // Acquire GIL before invoking Python callbacks
                    if (progress_callback) {
                        std::lock_guard<std::mutex> lock(callback_mutex);
                        py::gil_scoped_acquire acquire;
                        progress_callback();
                    }

                    return {index, res};
                }
                catch (const std::exception& e) {
                    // Log exception
                    if (log_callback) {
                        std::lock_guard<std::mutex> lock(callback_mutex);
                        py::gil_scoped_acquire acquire;
                        log_callback(std::string("topological_comp_res exception: ") + e.what());
                    }
                    throw; // Re-throw to be handled later
                }
                catch (...) {
                    if (log_callback) {
                        std::lock_guard<std::mutex> lock(callback_mutex);
                        py::gil_scoped_acquire acquire;
                        log_callback("topological_comp_res encountered an unknown exception.");
                    }
                    throw;
                }
            })
        );

        // Update progress
        if (progress_callback) {
            std::lock_guard<std::mutex> lock(callback_mutex);
            py::gil_scoped_acquire acquire;
            progress_callback();
        }
    }

    // Collect the results in the correct order
    for (auto& result_future : results) {
        try {
            auto result_pair = result_future.get();
            size_t index = result_pair.first;
            Eigen::VectorXd res = result_pair.second;
            output[index] = res;
        }
        catch (const std::exception& e) {
            // Handle task exceptions
            if (log_callback) {
                std::lock_guard<std::mutex> lock(callback_mutex);
                py::gil_scoped_acquire acquire;
                log_callback(std::string("Task exception: ") + e.what());
            }
            // Depending on requirements, decide how to handle failed tasks
            // For example, skip, fill with default values, or re-throw
        }
    }

    return output;
}

// Parallel function for jaccard_composite
std::vector<double> parallel_jaccard_composite_py(
    py::list CCx_loc_sums_list,
    py::list CCy_loc_sums_list,
    py::list feat_xs_list,
    py::list feat_ys_list,
    const std::string& jaccard_type,
    int num_workers,
    py::function progress_callback,
    py::function log_callback) 
{
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

    // Automatically determine the number of worker threads if not specified
    if (num_workers <= 0) {
        num_workers = std::min(static_cast<size_t>(4), std::thread::hardware_concurrency());
        if (num_workers == 0) num_workers = 4; // Fallback to 4 threads
    }

    // Initialize ThreadPool with limited number of threads
    ThreadPool pool(num_workers);
    std::vector<std::future<std::pair<size_t, double>>> results;
    results.reserve(list_size);

    // Prepare the output vector
    std::vector<doubl