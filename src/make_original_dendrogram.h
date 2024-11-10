#ifndef MAKE_ORIGINAL_DENDROGRAM_CC_H
#define MAKE_ORIGINAL_DENDROGRAM_CC_H

#include <vector>
#include <tuple>
#include <set>
#include <unordered_map>
#include <Eigen/Sparse>
#include <Eigen/Dense>

// Function to create the original dendrogram
std::tuple<
    std::vector<std::vector<int>>,        // CC: Connected Components
    Eigen::SparseMatrix<double>,          // E: Connectivity Matrix between CCs
    Eigen::MatrixXd,                      // duration: Birth and Death of CCs
    std::vector<std::vector<int>>         // history: History of CCs
>
make_original_dendrogram_cc(
    const Eigen::VectorXd& U,
    const Eigen::SparseMatrix<double>& A,
    const std::vector<double>& threshold
);

#endif // MAKE_ORIGINAL_DENDROGRAM_CC_H