#ifndef TOPOLOGICAL_COMP_H
#define TOPOLOGICAL_COMP_H

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>


// Function to compute adjacency matrix and Gaussian smoothing mask based on spatial locations
std::tuple<Eigen::SparseMatrix<int>, Eigen::MatrixXd> extract_adjacency_spatial(const Eigen::MatrixXd& loc, const std::string& spatial_type = "visium", double fwhm = 2.5);

// Function to extract connected components
std::vector<std::vector<int>> extract_connected_comp(const Eigen::VectorXd& tx, const Eigen::SparseMatrix<int>& A_sparse, const std::vector<double>& threshold_x, int num_spots, int min_size = 5);

// Function to extract the connected location matrix
Eigen::SparseMatrix<int> extract_connected_loc_mat(const std::vector<std::vector<int>>& CC, int num_spots, const std::string& format = "sparse");

// Function for topological connected component analysis
std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>> topological_comp_res(const Eigen::VectorXd& feat, const Eigen::SparseMatrix<int>& A, const Eigen::MatrixXd& mask,
                                                                                         const std::string& spatial_type = "visium", int min_size = 5, int thres_per = 30, const std::string& return_mode = "all");

#endif // TOPOLOGICAL_COMP_H