#include "parallelize.h"
#include "topological_comp.h"  // Ensure this header defines topological_comp_res
#include "jaccard.h"           // Ensure this header defines jaccard_composite
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <utility>
#include <stdexcept>
#include <omp.h>

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

// Parallel function for topological_comp_res using OpenMP
std::vector<Eigen::VectorXd> parallel_topological_comp(
    const std::vector<py::array_t<double>>& locs,
    const std::string& spatial_type,
    double fwhm,
    const std::vector<py::array_t<double>>& feats,
    int min_size,
    double thres_per,
    const std::string& return_mode,
    int num_workers,
    py::object progress_callback_obj,
    py::object log_callback_obj
) {
    // Use py::object to allow for None
    py::object progress_callback = progress_callback_obj.is_none() ? py::none() : progress_callback_obj;
    py::object log_callback = log_callback_obj.is_none() ? py::none() : log_callback_obj;
    
    // Log function start with parameters
    std::string log_msg = "Starting parallel_topological_comp with parameters: ";
    log_msg += "spatial_type=" + spatial_type + ", fwhm=" + std::to_string(fwhm);
    log_msg += ", min_size=" + std::to_string(min_size) + ", thres_per=" + std::to_string(thres_per);
    log_msg += ", return_mode=" + return_mode + ", num_workers=" + std::to_string(num_workers);
    log_message(log_msg);

    // Validate input sizes
    size_t list_size = locs.size();
    if (feats.size() != list_size) {
        std::string error_msg = "locs and feats must have the same length.";
        log_message(error_msg);
        if (!log_callback.is_none()) {
            log_callback(error_msg.c_str());
        }
        throw std::invalid_argument(error_msg);
    }
    
    log_message("Input validation passed. Processing " + std::to_string(list_size) + " items");

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
            if (!log_callback.is_none()) {
                log_callback(error_msg.c_str());
            }
            throw;
        }
    }

    // Determine number of threads
    if (num_workers > 0) {
        omp_set_num_threads(num_workers);
    }

    // Initialize output vector
    std::vector<Eigen::VectorXd> output(list_size);

    // Parallelize the loop with OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(list_size); ++i) {
        try {
            output[i] = topological_comp_res(locs_eigen[i], spatial_type, fwhm, feats_eigen[i], min_size, thres_per, return_mode);
        }
        catch (const std::exception& e) {
            // Acquire GIL before calling Python functions
            #pragma omp critical
            {
                std::string error_msg = "Error in topological_comp_res: ";
                error_msg += e.what();
                if (!log_callback.is_none()) {
                    py::gil_scoped_acquire acquire;
                    log_callback(error_msg.c_str());
                }
            }
            // Assign a default Eigen::VectorXd or handle as needed
            output[i] = Eigen::VectorXd();
        }

        // Update progress
        #pragma omp critical
        {
            if (!progress_callback.is_none()) {
                try {
                    // Acquire GIL before calling Python functions
                    py::gil_scoped_acquire acquire;
                    progress_callback(1);
                }
                catch (const py::error_already_set& e) {
                    std::string error_msg = "Error in progress_callback: ";
                    error_msg += e.what();
                    if (!log_callback.is_none()) {
                        py::gil_scoped_acquire acquire;
                        log_callback(error_msg.c_str());
                    }
                    throw std::runtime_error(error_msg);
                }
            }
        }
    }

    return output;
}

// Parallel function for jaccard_composite using OpenMP
std::vector<double> parallel_jaccard_composite(
    const std::vector<py::array_t<double>>& CCx_loc_sums,
    const std::vector<py::array_t<double>>& CCy_loc_sums,
    const std::vector<py::array_t<double>>& feat_xs,
    const std::vector<py::array_t<double>>& feat_ys,
    const std::string& jaccard_type,
    int num_workers,
    py::object progress_callback_obj,
    py::object log_callback_obj
) {
    // Use py::object to allow for None
    py::object progress_callback = progress_callback_obj.is_none() ? py::none() : progress_callback_obj;
    py::object log_callback = log_callback_obj.is_none() ? py::none() : log_callback_obj;
    
    // Log function start with parameters
    std::string log_msg = "Starting parallel_jaccard_composite with parameters: ";
    log_msg += "jaccard_type=" + jaccard_type + ", num_workers=" + std::to_string(num_workers);
    log_message(log_msg);

    // Validate input sizes
    size_t list_size = CCx_loc_sums.size();
    if (CCy_loc_sums.size() != list_size ||
        feat_xs.size() != list_size ||
        feat_ys.size() != list_size) {
        std::string error_msg = "All input lists must have the same length.";
        log_message(error_msg);
        if (!log_callback.is_none()) {
            log_callback(error_msg.c_str());
        }
        throw std::invalid_argument(error_msg);
    }
    
    log_message("Input validation passed. Processing " + std::to_string(list_size) + " items");

    // Convert to Eigen types
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
            Eigen::VectorXd CCx_sum = array_to_vector(CCx_loc_sums[i]);
            Eigen::VectorXd CCy_sum = array_to_vector(CCy_loc_sums[i]);
            Eigen::VectorXd feat_x = array_to_vector(feat_xs[i]);
            Eigen::VectorXd feat_y = array_to_vector(feat_ys[i]);

            CCx_loc_sums_vec.push_back(CCx_sum);
            CCy_loc_sums_vec.push_back(CCy_sum);
            feat_xs_vec.push_back(feat_x);
            feat_ys_vec.push_back(feat_y);
        }
        catch (const std::exception& e) {
            std::string error_msg = "Error converting lists to Eigen vectors: ";
            error_msg += e.what();
            if (!log_callback.is_none()) {
                log_callback(error_msg.c_str());
            }
            throw;
        }
    }

    // Determine number of threads
    if (num_workers > 0) {
        omp_set_num_threads(num_workers);
    }

    // Initialize output vector
    std::vector<double> output(list_size, 0.0);

    // Parallelize the loop with OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(list_size); ++i) {
        try {
            // Log the start of processing for this item
            std::string thread_log = "Thread " + std::to_string(omp_get_thread_num()) + 
                                    " processing item " + std::to_string(i);
            #pragma omp critical
            {
                log_message(thread_log);
            }
            
            // Get pointers to feature vectors if jaccard_type is not "default"
            const Eigen::VectorXd* feat_x_ptr = nullptr;
            const Eigen::VectorXd* feat_y_ptr = nullptr;
            
            if (jaccard_type != "default") {
                feat_x_ptr = &feat_xs_vec[i];
                feat_y_ptr = &feat_ys_vec[i];
            }
            
            // Call jaccard_composite with appropriate arguments
            output[i] = jaccard_composite(CCx_loc_sums_vec[i], CCy_loc_sums_vec[i], feat_x_ptr, feat_y_ptr);
            
            // Log successful completion
            #pragma omp critical
            {
                log_message("Thread " + std::to_string(omp_get_thread_num()) + 
                           " completed item " + std::to_string(i) + 
                           " with result " + std::to_string(output[i]));
            }
            
            // Update progress if callback is provided
            if (!progress_callback.is_none()) {
                py::gil_scoped_acquire acquire;
                progress_callback(1);  // Increment progress by 1
            }
        }
        catch (const std::exception& e) {
            #pragma omp critical
            {
                std::string error_msg = "Error processing item " + std::to_string(i) + ": ";
                error_msg += e.what();
                log_message(error_msg);
                if (!log_callback.is_none()) {
                    py::gil_scoped_acquire acquire;
                    log_callback(error_msg.c_str());
                }
            }
            // Don't throw here as it would terminate the parallel region
            // Instead, set a flag to indicate an error occurred
            #pragma omp critical
            {
                output[i] = 0.0;  // Set a default value
            }
        }
    }

    return output;
}

// Expose to Python via Pybind11
PYBIND11_MODULE(parallelize, m) {  // Module name within the STopover package
    m.def("parallel_topological_comp", 
          static_cast<std::vector<Eigen::VectorXd> (*)(const std::vector<py::array_t<double>>&, const std::string&, double, const std::vector<py::array_t<double>>&, int, double, const std::string&, int, py::object, py::object)>(&parallel_topological_comp), 
          "Parallelized topological_comp_res function using OpenMP",
          py::arg("locs"),
          py::arg("spatial_type") = "visium",
          py::arg("fwhm") = 2.5,
          py::arg("feats"),
          py::arg("min_size") = 5,
          py::arg("thres_per") = 30,
          py::arg("return_mode") = "all",
          py::arg("num_workers") = 4,
          py::arg("progress_callback") = py::none(),
          py::arg("log_callback") = py::none());

    m.def("parallel_jaccard_composite", 
          static_cast<std::vector<double> (*)(const std::vector<py::array_t<double>>&, const std::vector<py::array_t<double>>&, const std::vector<py::array_t<double>>&, const std::vector<py::array_t<double>>&, const std::string&, int, py::object, py::object)>(&parallel_jaccard_composite), 
          "Parallelized jaccard_composite function using OpenMP",
          py::arg("CCx_loc_sums"),
          py::arg("CCy_loc_sums"),
          py::arg("feat_xs"),
          py::arg("feat_ys"),
          py::arg("jaccard_type") = "default",
          py::arg("num_workers") = 4,
          py::arg("progress_callback") = py::none(),
          py::arg("log_callback") = py::none());
}