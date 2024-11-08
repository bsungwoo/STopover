#ifndef MAKE_SMOOTHED_DENDROGRAM_H
#define MAKE_SMOOTHED_DENDROGRAM_H

#include <vector>
#include <tuple>
#include <Eigen/Dense>

// Function to create the smoothed dendrogram
std::tuple<
    std::vector<std::vector<int>>,    // nCC
    Eigen::MatrixXd,                  // nE
    Eigen::MatrixXd,                  // nduration
    std::vector<std::vector<int>>     // nchildren
>
make_smoothed_dendrogram(
    const std::vector<std::vector<int>>& cCC,
    const Eigen::MatrixXd& cE,
    const Eigen::MatrixXd& cduration,
    const std::vector<std::vector<int>>& chistory,
    const Eigen::Vector2d& lim_size
);

#endif // MAKE_SMOOTHED_DENDROGRAM_H