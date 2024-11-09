#ifndef MAKE_DENDROGRAM_BAR_H
#define MAKE_DENDROGRAM_BAR_H

#include <vector>
#include <tuple>
#include <Eigen/Dense>


/**
 * @brief Constructs a dendrogram bar based on provided history and duration matrices.
 *
 * @param history Vector of connected components history.
 * @param duration Matrix containing duration information.
 * @param cvertical_x Vertical X coordinates (optional).
 * @param cvertical_y Vertical Y coordinates (optional).
 * @param chorizontal_x Horizontal X coordinates (optional).
 * @param chorizontal_y Horizontal Y coordinates (optional).
 * @param cdots Dots matrix (optional).
 * @return A tuple containing nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, and nlayer.
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_dendrogram_bar(const std::vector<std::vector<int>>& history,
                    const Eigen::MatrixXd& duration,
                    const Eigen::MatrixXd& cvertical_x = Eigen::MatrixXd(),
                    const Eigen::MatrixXd& cvertical_y = Eigen::MatrixXd(),
                    const Eigen::MatrixXd& chorizontal_x = Eigen::MatrixXd(),
                    const Eigen::MatrixXd& chorizontal_y = Eigen::MatrixXd(),
                    const Eigen::MatrixXd& cdots = Eigen::MatrixXd());

#endif // MAKE_DENDROGRAM_BAR_H