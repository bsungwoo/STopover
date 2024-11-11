// make_smoothed_dendrogram.h
#ifndef MAKE_SMOOTHED_DENDROGRAM_H
#define MAKE_SMOOTHED_DENDROGRAM_H

#include <vector>
#include <set>
#include <tuple>
#include <Eigen/Sparse>
#include <Eigen/Dense>

/**
 * @brief Smooths the dendrogram by adjusting connected components based on size and connectivity.
 *
 * @param cCC Current connected components.
 * @param cE Current connectivity matrix.
 * @param cduration Current duration matrix.
 * @param chistory Current history of connections.
 * @param lim_size Vector containing minimum and maximum size limits.
 * @return A tuple containing:
 *         - Smoothed Connected Components (vector of vectors of node indices)
 *         - Smoothed Connectivity Matrix (Eigen::SparseMatrix<double>)
 *         - Smoothed Duration Matrix (Eigen::MatrixXd)
 *         - Smoothed History of Connections (vector of vectors of parent indices)
 */
std::tuple<
    std::vector<std::vector<int>>,       // Smoothed Connected Components
    Eigen::SparseMatrix<double>,         // Smoothed E matrix
    Eigen::MatrixXd,                     // Smoothed duration
    std::vector<std::vector<int>>        // Smoothed history
>
make_smoothed_dendrogram(
    const std::vector<std::vector<int>>& cCC,
    const Eigen::SparseMatrix<double>& cE,
    const Eigen::MatrixXd& cduration,
    const std::vector<std::vector<int>>& chistory,
    const Eigen::Vector2d& lim_size
);

#endif // MAKE_SMOOTHED_DENDROGRAM_H