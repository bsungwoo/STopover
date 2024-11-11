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

namespace py = pybind11;

// Parallel function for topological_comp_res
std::vector<Eigen::VectorXd> parallel_topological_comp(
    const std::vector<py::array_t<double>>& locs,
    const std::string& spatial_type, double fwhm,
    const std::vector<py::array_t<double>>& feats,
    int min_size, int thres_per, const std::string& return_mode, int num_workers,
    py::function progress_callback,
    py::function log_callback);

// Parallel function for jaccard_composite
std::vector<double> parallel_jaccard_composite(
    const std::vector<Eigen::VectorXd>& CCx_loc_sums, 
    const std::vector<Eigen::VectorXd>& CCy_loc_sums,
    const std::vector<Eigen::VectorXd>& feat_xs, 
    const std::vector<Eigen::VectorXd>& feat_ys, 
    const std::string& jaccard_type,
    int num_workers,
    py::function progress_callback,
    py::function log_callback);

#endif // PARALLELIZE_H