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
    // ... (same as previously defined)
}

Eigen::MatrixXd array_to_matrix(const py::array_t<double>& array) {
    // ... (same as previously defined)
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
    py::function progress_callback,
    py::function log_callback
) {
    // Validate input sizes
    size_t list_size = locs.size();
    if (feats.size() != list_size) {
        std::string error_msg = "locs and feats must have the same length.";
        py::gil_scoped_acquire acquire;
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
            py::gil_scoped_acquire acquire;
            log_callback(error_msg.c_str());
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
                py::gil_scoped_acquire acquire;
                log_callback(error_msg.c_str());
            }
            // Assign a default Eigen::VectorXd or handle as needed
            output[i] = Eigen::VectorXd();
        }

        // Update progress
        #pragma omp critical
        {
            if (progress_callback) {
                try {
                    // Acquire GIL before calling Python functions
                    py::gil_scoped_acquire acquire;
                    progress_callback(1);
                }
                catch (const py::error_already_set& e) {
                    std::string error_msg = "Error in progress_callback: ";
                    error_msg += e.what();
                    py::gil_scoped_acquire acquire;
                    log_callback(error_msg.c_str());
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
    py::function progress_callback,
    py::function log_callback
) {
    // Validate input sizes
    size_t list_size = CCx_loc_sums.size();
    if (CCy_loc_sums.size() != list_size ||
        feat_xs.size() != list_size ||
        feat_ys.size() != list_size) {
        std::string error_msg = "All input lists must have the same length.";
        py::gil_scoped_acquire acquire;
        log_callback(error_msg.c_str());
        throw std::invalid_argument(error_msg);
    }

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
            py::gil_scoped_acquire acquire;
            log_callback(error_msg.c_str());
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
            double jaccard_index;
            if (jaccard_type == "default") {
                jaccard_index = jaccard_composite(CCx_loc_sums_vec[i], CCy_loc_sums_vec[i], nullptr, nullptr);
            } else if (jaccard_type == "weighted") {
                jaccard_index = jaccard_composite(CCx_loc_sums_vec[i], CCy_loc_sums_vec[i], &feat_xs_vec[i], &feat_ys_vec[i]);
            } else {
                throw std::invalid_argument("Invalid jaccard_type: " + jaccard_type);
            }
            output[i] = jaccard_index;
        }
        catch (const std::exception& e) {
            // Acquire GIL before calling Python functions
            #pragma omp critical
            {
                std::string error_msg = "Error in jaccard_composite: ";
                error_msg += e.what();
                py::gil_scoped_acquire acquire;
                log_callback(error_msg.c_str());
            }
            // Assign a default value or handle as needed
            output[i] = 0.0;
        }

        // Update progress
        #pragma omp critical
        {
            if (progress_callback) {
                try {
                    // Acquire GIL before calling Python functions
                    py::gil_scoped_acquire acquire;
                    progress_callback(1);
                }
                catch (const py::error_already_set& e) {
                    std::string error_msg = "Error in progress_callback: ";
                    error_msg += e.what();
                    py::gil_scoped_acquire acquire;
                    log_callback(error_msg.c_str());
                    throw std::runtime_error(error_msg);
                }
            }
        }
    }

    return output;
}

// Expose to Python via Pybind11
PYBIND11_MODULE(parallelize, m) {  // Module name within the STopover package
    m.def("parallel_topological_comp_omp", &parallel_topological_comp_omp, "Parallelized topological_comp_res function using OpenMP",
          py::arg("locs"), py::arg("spatial_type") = "visium", py::arg("fwhm") = 2.5, py::arg("feats"), 
          py::arg("min_size") = 5, py::arg("thres_per") = 30, py::arg("return_mode") = "all", 
          py::arg("num_workers") = 4, py::arg("progress_callback"), py::arg("log_callback"));

    m.def("parallel_jaccard_composite_omp", &parallel_jaccard_composite_omp, "Parallelized jaccard_composite function using OpenMP",
          py::arg("CCx_loc_sums"), py::arg("CCy_loc_sums"), py::arg("feat_xs"), py::arg("feat_ys"), 
          py::arg("jaccard_type") = "default", py::arg("num_workers") = 4, py::arg("progress_callback"), py::arg("log_callback"));
}