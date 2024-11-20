#include "parallelize.h"
#include "topological_comp.h"  // Ensure this header defines topological_comp_res
#include "jaccard.h"           // Ensure this header defines jaccard_composite
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
#include <utility>               // For std::pair>
#include <memory>                // For smart pointers

namespace py = pybind11;

// Utility functions (assuming these are correctly defined in your project)
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

// Parallel function for topological_comp_res
std::vector<Eigen::VectorXd> parallel_topological_comp(
    const std::vector<py::array_t<double>>& locs,
    const std::string& spatial_type, double fwhm,
    const std::vector<py::array_t<double>>& feats,
    int min_size, double thres_per, const std::string& return_mode, 
    int num_workers,
    py::function progress_callback,
    py::function log_callback,
    int batch_size)  // Added batch_size
{
    // Validate input sizes
    size_t list_size = locs.size();
    if (feats.size() != list_size) {
        std::string error_msg = "locs and feats must have the same length.";
        log_callback(error_msg.c_str());
        throw std::invalid_argument(error_msg);
    }

    // Convert to Eigen types
    std::vector<Eigen::MatrixXd> locs_eigen;
    std::vector<Eigen::VectorXd> feats_eigen;

    locs_eigen.reserve(list_size);
    feats_eigen.reserve(list_size);

    for (size_t i = 0; i < list_size; ++i) {
        try {
            Eigen::MatrixXd loc = array_to_matrix(locs[i]);
            Eigen::VectorXd feat = array_to_vector(feats[i]);
            locs_eigen.push_back(loc);
            feats_eigen.push_back(feat);
        }
        catch (const std::exception& e) {
            std::string error_msg = "Error converting arrays to Eigen types: ";
            error_msg += e.what();
            log_callback(error_msg.c_str());
            throw;
        }
    }

    // Determine number of workers
    if (num_workers <= 0) {
        unsigned int hw_threads = std::thread::hardware_concurrency();
        size_t hw_threads_size = static_cast<size_t>(hw_threads);
        num_workers = std::min(static_cast<size_t>(4), hw_threads_size);
        if (hw_threads_size == 0) num_workers = 4; // Fallback to 4 threads
    }

    // Initialize ThreadPool
    ThreadPool pool(num_workers);
    std::vector<std::future<std::pair<size_t, Eigen::VectorXd>>> results;

    // Initialize output vector
    std::vector<Eigen::VectorXd> output(list_size);

    // Determine number of batches
    size_t num_batches = (list_size + batch_size - 1) / batch_size;

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        size_t start = batch_idx * batch_size;
        size_t end = std::min(start + batch_size, list_size);
        size_t current_batch_size = end - start;

        // Enqueue tasks for the current batch
        for (size_t i = start; i < end; ++i) {
            size_t index = i;
            Eigen::MatrixXd loc = locs_eigen[i];
            Eigen::VectorXd feat = feats_eigen[i];

            results.emplace_back(pool.enqueue([=]() -> std::pair<size_t, Eigen::VectorXd> {
                try {
                    Eigen::VectorXd res = topological_comp_res(loc, spatial_type, fwhm, feat, min_size, thres_per, return_mode);
                    return {index, res};
                }
                catch (const std::exception& e) {
                    std::string error_msg = "Error in topological_comp_res: ";
                    error_msg += e.what();
                    // Optionally, use log_callback here if thread-safe
                    log_callback(error_msg.c_str());
                    // Return a default or empty Eigen::VectorXd
                    return {index, Eigen::VectorXd()};
                }
            }));
        }

        // Update progress with the number of items in the current batch
        if (progress_callback) {
            try {
                progress_callback(static_cast<int>(current_batch_size));
            }
            catch (const py::error_already_set& e) {
                std::string error_msg = "Error in progress_callback: ";
                error_msg += e.what();
                log_callback(error_msg.c_str());
                throw std::runtime_error(error_msg);
            }
        }

        // Collect results for the current batch
        for (size_t i = 0; i < current_batch_size; ++i) {
            try {
                auto result_pair = results[i].get();
                size_t index = result_pair.first;
                Eigen::VectorXd res = result_pair.second;
                output[index] = res;
            }
            catch (const std::exception& e) {
                std::string error_msg = "Error retrieving topological_comp_res result: ";
                error_msg += e.what();
                log_callback(error_msg.c_str());
                throw;
            }
        }

        // Remove processed futures
        results.erase(results.begin(), results.begin() + current_batch_size);

        // Clear memory
        locs_eigen.erase(locs_eigen.begin(), locs_eigen.begin() + current_batch_size);
        feats_eigen.erase(feats_eigen.begin(), feats_eigen.begin() + current_batch_size);
        // Note: 'gc()' is not a standard C++ function. Remove it or replace with appropriate memory management.
    }

    return output;
}

// Parallel function for jaccard_composite
std::vector<double> parallel_jaccard_composite(
    py::list CCx_loc_sums_list,
    py::list CCy_loc_sums_list,
    py::list feat_xs_list,
    py::list feat_ys_list,
    const std::string& jaccard_type,
    int num_workers,
    py::function progress_callback,
    py::function log_callback,
    int batch_size)  // Added batch_size
{
    // Check that all lists have the same size
    size_t list_size = CCx_loc_sums_list.size();
    if (CCy_loc_sums_list.size() != list_size ||
        feat_xs_list.size() != list_size ||
        feat_ys_list.size() != list_size) {
        std::string error_msg = "All input lists must have the same length.";
        log_callback(error_msg.c_str());
        throw std::invalid_argument(error_msg);
    }

    // Convert lists to C++ vectors
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
            Eigen::VectorXd CCx_sum = array_to_vector(CCx_loc_sums_list[i].cast<py::array_t<double>>());
            Eigen::VectorXd CCy_sum = array_to_vector(CCy_loc_sums_list[i].cast<py::array_t<double>>());
            Eigen::VectorXd feat_x = array_to_vector(feat_xs_list[i].cast<py::array_t<double>>());
            Eigen::VectorXd feat_y = array_to_vector(feat_ys_list[i].cast<py::array_t<double>>());

            CCx_loc_sums_vec.push_back(CCx_sum);
            CCy_loc_sums_vec.push_back(CCy_sum);
            feat_xs_vec.push_back(feat_x);
            feat_ys_vec.push_back(feat_y);
        }
        catch (const std::exception& e) {
            std::string error_msg = "Error converting lists to Eigen vectors: ";
            error_msg += e.what();
            log_callback(error_msg.c_str());
            throw;
        }
    }

    // Determine number of workers
    if (num_workers <= 0) {
        unsigned int hw_threads = std::thread::hardware_concurrency();
        size_t hw_threads_size = static_cast<size_t>(hw_threads);
        num_workers = std::min(static_cast<size_t>(4), hw_threads_size);
        if (hw_threads_size == 0) num_workers = 4; // Fallback to 4 threads
    }

    // Initialize ThreadPool
    ThreadPool pool(num_workers);
    std::vector<std::future<std::pair<size_t, double>>> results;

    // Initialize output vector
    std::vector<double> output(list_size, 0.0);

    // Determine number of batches
    size_t num_batches = (list_size + batch_size - 1) / batch_size;

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        size_t start = batch_idx * batch_size;
        size_t end = std::min(start + batch_size, list_size);
        size_t current_batch_size = end - start;

        // Enqueue tasks for the current batch
        for (size_t i = start; i < end; ++i) {
            size_t index = i;
            Eigen::VectorXd CCx_sum = CCx_loc_sums_vec[i];
            Eigen::VectorXd CCy_sum = CCy_loc_sums_vec[i];
            Eigen::VectorXd feat_x = feat_xs_vec[i];
            Eigen::VectorXd feat_y = feat_ys_vec[i];

            results.emplace_back(pool.enqueue([=]() -> std::pair<size_t, double> {
                try {
                    double jaccard_index;
                    if (jaccard_type == "default") {
                        jaccard_index = jaccard_composite(CCx_sum, CCy_sum, nullptr, nullptr);
                    } else if (jaccard_type == "weighted") {
                        jaccard_index = jaccard_composite(CCx_sum, CCy_sum, &feat_x, &feat_y);
                    } else {
                        throw std::invalid_argument("Invalid jaccard_type: " + jaccard_type);
                    }
                    return {index, jaccard_index};
                }
                catch (const std::exception& e) {
                    std::string error_msg = "Error in jaccard_composite: ";
                    error_msg += e.what();
                    log_callback(error_msg.c_str());
                    return {index, 0.0}; // Assign a default value or handle as needed
                }
            }));
        }

        // Update progress with the number of items in the current batch
        if (progress_callback) {
            try {
                progress_callback(static_cast<int>(current_batch_size));
            }
            catch (const py::error_already_set& e) {
                std::string error_msg = "Error in progress_callback: ";
                error_msg += e.what();
                log_callback(error_msg.c_str());
                throw std::runtime_error(error_msg);
            }
        }

        // Collect results for the current batch
        for (size_t i = 0; i < current_batch_size; ++i) {
            try {
                auto result_pair = results[i].get();
                size_t index = result_pair.first;
                double value = result_pair.second;
                output[index] = value;
            }
            catch (const std::exception& e) {
                std::string error_msg = "Error retrieving jaccard_composite result: ";
                error_msg += e.what();
                log_callback(error_msg.c_str());
                throw;
            }
        }

        // Remove processed futures
        results.erase(results.begin(), results.begin() + current_batch_size);

        // Clear memory
        CCx_loc_sums_vec.erase(CCx_loc_sums_vec.begin(), CCx_loc_sums_vec.begin() + current_batch_size);
        CCy_loc_sums_vec.erase(CCy_loc_sums_vec.begin(), CCy_loc_sums_vec.begin() + current_batch_size);
        feat_xs_vec.erase(feat_xs_vec.begin(), feat_xs_vec.begin() + current_batch_size);
        feat_ys_vec.erase(feat_ys_vec.begin(), feat_ys_vec.begin() + current_batch_size);
        // Note: 'gc()' is not a standard C++ function. Remove it or replace with appropriate memory management.
    }

    return output;
}

// Expose to Python via Pybind11
PYBIND11_MODULE(parallelize, m) {  // Module name within the STopover package
    m.def("parallel_topological_comp", &parallel_topological_comp, "Parallelized topological_comp_res function",
          py::arg("locs"), py::arg("spatial_type") = "visium", py::arg("fwhm") = 2.5, py::arg("feats"), 
          py::arg("min_size") = 5, py::arg("thres_per") = 30, py::arg("return_mode") = "all", 
          py::arg("num_workers") = 4, py::arg("progress_callback"), py::arg("log_callback"),
          py::arg("batch_size") = 500);  // Exposed batch_size

    m.def("parallel_jaccard_composite", &parallel_jaccard_composite, "Parallelized jaccard_composite function accepting lists of NumPy arrays",
          py::arg("CCx_loc_sums"), py::arg("CCy_loc_sums"), py::arg("feat_xs"), py::arg("feat_ys"), 
          py::arg("jaccard_type") = "default", py::arg("num_workers") = 4, py::arg("progress_callback"), py::arg("log_callback"),
          py::arg("batch_size") = 500);  // Exposed batch_size
}