#ifndef MAKE_SMOOTHED_DENDROGRAM_H
#define MAKE_SMOOTHED_DENDROGRAM_H

#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>

// Function to create the smoothed dendrogram
std::tuple<
    std::vector<std::vector<int>>,        // nCC: New Connected Components
    Eigen::SparseMatrix<double>,          // nE: Connectivity Matrix between nCCs
    Eigen::MatrixXd,                      // nduration: Birth and Death of nCCs
    std::vector<std::vector<int>>         // nchildren: History of nCCs
>
make_smoothed_dendrogram(
    const std::vector<std::vector<int>>& cCC,
    const Eigen::SparseMatrix<double>& cE,
    const Eigen::MatrixXd& cduration,
    const std::vector<std::vector<int>>& chistory,
    const Eigen::Vector2d lim_size = Eigen::Vector2d(0, std::numeric_limits<double>::infinity())
);

#endif // MAKE_SMOOTHED_DENDROGRAM_H