#ifndef MAKE_ORIGINAL_DENDROGRAM_CC_H
#define MAKE_ORIGINAL_DENDROGRAM_CC_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <stdexcept>
#include <queue>

// Function to create the original dendrogram
std::tuple<
    std::vector<std::vector<int>>,        // CC: Connected Components
    Eigen::SparseMatrix<double, Eigen::RowMajor>,          // E: Connectivity Matrix between CCs
    Eigen::MatrixXd,                      // duration: Birth and Death of CCs
    std::vector<std::vector<int>>         // history: History of CCs
>
make_original_dendrogram_cc(
    const Eigen::VectorXd& U,
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
    const std::vector<double>& threshold
);

#endif // MAKE_ORIGINAL_DENDROGRAM_CC_H