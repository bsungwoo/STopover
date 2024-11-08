#ifndef MAKE_SMOOTHED_DENDROGRAM_H
#define MAKE_SMOOTHED_DENDROGRAM_H

#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Function to create a smoothed dendrogram
std::tuple<
    std::vector<std::vector<int>>,   // nCC
    Eigen::MatrixXd,                 // nE
    Eigen::MatrixXd,                 // nduration
    std::vector<std::vector<int>>    // nchildren
>
make_smoothed_dendrogram(
    const std::vector<std::vector<int>>& cCC,
    const Eigen::MatrixXd& cE,
    const Eigen::MatrixXd& cduration,
    const std::vector<std::vector<int>>& chistory,
    const Eigen::Vector2d& lim_size = Eigen::Vector2d(0, std::numeric_limits<double>::infinity())
);

#endif // MAKE_SMOOTHED_DENDROGRAM_H