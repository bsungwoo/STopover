#ifndef MAKE_SMOOTHED_DENDROGRAM_H
#define MAKE_SMOOTHED_DENDROGRAM_H

#include <vector>
#include <tuple>
#include <Eigen/Sparse>    // For Eigen::SparseMatrix
#include <Eigen/Dense>     // For Eigen::MatrixXd and Eigen::Vector2d

/**
 * @brief Constructs a smoothed dendrogram based on provided history and duration matrices.
 *
 * @param cCC Vector of connected components (each component is a vector of integers).
 * @param cE Adjacency matrix or similar representation (Eigen::MatrixXd).
 * @param cduration Matrix containing duration information (Eigen::MatrixXd).
 * @param chistory Vector of connected components history (each component is a vector of integers).
 * @param lim_size Vector of two elements specifying [min_size, max_size].
 * @return A tuple containing:
 *         - nCC: Updated vector of connected components.
 *         - nE: Updated adjacency matrix.
 *         - nduration: Updated duration matrix.
 *         - nchildren: Updated vector of connected components history.
 */
std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<double>, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_smoothed_dendrogram(const std::vector<std::vector<int>>& cCC,
                         const Eigen::SparseMatrix<double>& cE,
                         const Eigen::MatrixXd& cduration,
                         const std::vector<std::vector<int>>& chistory,
                         const Eigen::Vector2d& lim_size = Eigen::Vector2d(0, std::numeric_limits<double>::infinity()));

#endif // MAKE_SMOOTHED_DENDROGRAM_H