#ifndef TOPOLOGICAL_COMP_H
#define TOPOLOGICAL_COMP_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <string>

// Function declarations with consistent types

// Function to compute adjacency matrix and Gaussian smoothing mask based on spatial locations
std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd> extract_adjacency_spatial(
    const Eigen::MatrixXd& loc, const std::string& spatial_type = "visium", double fwhm = 2.5);

// Placeholder function declarations - Implement these or link them properly
std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>, std::vector<int>> make_original_dendrogram_cc(
    const Eigen::VectorXd&, const Eigen::SparseMatrix<double>&, const std::vector<double>&);

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> make_smoothed_dendrogram(
    const std::vector<std::vector<int>>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const Eigen::ArrayXd&);

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> make_dendrogram_bar(
    const std::vector<int>&, const std::vector<int>&);

// Corrected function to extract connected components
std::vector<std::vector<int>> extract_connected_comp(
    const Eigen::VectorXd& tx, const Eigen::SparseMatrix<double>& A_sparse, 
    const std::vector<double>& threshold_x, int num_spots, int min_size);

// Function to extract the connected location matrix
Eigen::SparseMatrix<int> extract_connected_loc_mat(
    const std::vector<std::vector<int>>& CC, int num_spots, const std::string& format);

// Adjusted function to filter connected component locations based on expression values
Eigen::SparseMatrix<int> filter_connected_loc_exp(
    const Eigen::SparseMatrix<int>& CC_loc_mat, const Eigen::VectorXd& feat_data, int thres_per);

// Function for topological connected component analysis
std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>> topological_comp_res(
    const Eigen::VectorXd& feat, const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& mask,
    const std::string& spatial_type, int min_size, int thres_per, const std::string& return_mode);

// Function to log messages
void log_message(const std::string& message);

// Function to extract connected components
std::vector<std::vector<int>> extract_connected_comp_python_style(
    const Eigen::VectorXd& tx,
    const Eigen::SparseMatrix<double>& A_sparse,
    const std::vector<double>& threshold_x,
    int num_spots,
    int min_size);

// Function to smooth feature vector
Eigen::VectorXd smooth_feature_vector_python_style(
    const Eigen::VectorXd& feat,
    const Eigen::MatrixXd& mask);

#endif // TOPOLOGICAL_COMP_H