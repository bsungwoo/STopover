#ifndef MAKE_DENDROGRAM_BAR_H
#define MAKE_DENDROGRAM_BAR_H

#include <vector>
#include <tuple>
#include <Eigen/Dense>

// Function to create dendrogram bars
std::tuple<
    Eigen::MatrixXd,                       // nvertical_x
    Eigen::MatrixXd,                       // nvertical_y
    Eigen::MatrixXd,                       // nhorizontal_x
    Eigen::MatrixXd,                       // nhorizontal_y
    Eigen::MatrixXd,                       // ndots
    std::vector<std::vector<int>>          // nlayer
>
make_dendrogram_bar(
    const std::vector<std::vector<int>>& history,
    const Eigen::MatrixXd& duration,
    const Eigen::MatrixXd& cvertical_x = Eigen::MatrixXd(),
    const Eigen::MatrixXd& cvertical_y = Eigen::MatrixXd(),
    const Eigen::MatrixXd& chorizontal_x = Eigen::MatrixXd(),
    const Eigen::MatrixXd& chorizontal_y = Eigen::MatrixXd(),
    const Eigen::MatrixXd& cdots = Eigen::MatrixXd()
);

#endif // MAKE_DENDROGRAM_BAR_H