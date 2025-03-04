// parallelize.h
#ifndef PARALLELIZE_H
#define PARALLELIZE_H

#include <future>
#include <vector>
#include <tuple>
#include <string>
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "logging.h"  // Include logging header

namespace py = pybind11;

// Utility functions for array conversion
Eigen::VectorXd array_to_vector(const py::array_t<double>& array);
Eigen::MatrixXd array_to_matrix(const py::array_t<double>& array);

// Parallel function for topological_comp_res
std::vector<Eigen::VectorXd> parallel_topological_comp(
    const std::vector<py::array_t<double>>& locs,
    const std::string& spatial_type, 
    double fwhm,
    const std::vector<py::array_t<double>>& feats,
    int min_size, 
    double thres_per, 
    const std::string& return_mode, 
    int num_workers,
    py::object progress_callback_obj,  // Changed from py::function to py::object
    py::object log_callback_obj);      // Changed from py::function to py::object

// Parallel function for jaccard_composite
std::vector<double> parallel_jaccard_composite(
    const std::vector<py::array_t<double>>& CCx_loc_sums,  // Changed from py::list
    const std::vector<py::array_t<double>>& CCy_loc_sums,  // Changed from py::list
    const std::vector<py::array_t<double>>& feat_xs,       // Changed from py::list
    const std::vector<py::array_t<double>>& feat_ys,       // Changed from py::list
    const std::string& jaccard_type,
    int num_workers,
    py::object progress_callback_obj,  // Changed from py::function to py::object
    py::object log_callback_obj);      // Changed from py::function to py::object

#endif // PARALLELIZE_H