#ifndef ORIGINAL_DENDROGRAM_H
#define ORIGINAL_DENDROGRAM_H

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <set>
#include <queue>
#include <tuple>
#include "logging.h"  // Include logging header

// Function to extract connected nodes using breadth-first search (BFS)
std::set<int> extract_connected_nodes(const std::vector<std::vector<int>>& edge_list, int sel_node_idx);

// Function to generate connected components from a sparse adjacency matrix
std::vector<std::set<int>> connected_components_generator(const Eigen::SparseMatrix<int>& A);

// Function to create the original dendrogram with connected components
std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<double>, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_original_dendrogram_cc(const Eigen::VectorXd& U, const Eigen::SparseMatrix<int>& A, const std::vector<double>& threshold);

#endif // ORIGINAL_DENDROGRAM_H