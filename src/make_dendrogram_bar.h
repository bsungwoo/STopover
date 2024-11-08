#ifndef MAKE_DENDROGRAM_BAR_H
#define MAKE_DENDROGRAM_BAR_H

#include <vector>
#include <tuple>
#include <Eigen/Dense>

/**
 * @brief Creates a dendrogram bar representation based on history and duration of connected components.
 *
 * @param history A vector of vectors where history[i] contains the indices of the children of connected component i.
 * @param duration An Eigen::MatrixXd where duration(i, 0) is the birth time and duration(i, 1) is the death time of connected component i.
 * @param cvertical_x Optional Eigen::MatrixXd for existing vertical x-coordinates in the dendrogram.
 * @param cvertical_y Optional Eigen::MatrixXd for existing vertical y-coordinates in the dendrogram.
 * @param chorizontal_x Optional Eigen::MatrixXd for existing horizontal x-coordinates in the dendrogram.
 * @param chorizontal_y Optional Eigen::MatrixXd for existing horizontal y-coordinates in the dendrogram.
 * @param cdots Optional Eigen::MatrixXd for existing dots in the dendrogram.
 *
 * @return A tuple containing:
 *         - nvertical_x: Eigen::MatrixXd for vertical x-coordinates.
 *         - nvertical_y: Eigen::MatrixXd for vertical y-coordinates.
 *         - nhorizontal_x: Eigen::MatrixXd for horizontal x-coordinates.
 *         - nhorizontal_y: Eigen::MatrixXd for horizontal y-coordinates.
 *         - ndots: Eigen::MatrixXd for dots.
 *         - nlayer: A vector of vectors indicating the layers of the dendrogram.
 */
std::tuple<
    Eigen::MatrixXd,    // nvertical_x
    Eigen::MatrixXd,    // nvertical_y
    Eigen::MatrixXd,    // nhorizontal_x
    Eigen::MatrixXd,    // nhorizontal_y
    Eigen::MatrixXd,    // ndots
    std::vector<std::vector<int>> // nlayer
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