#ifndef JACCARD_H
#define JACCARD_H

#include <Eigen/Dense> // Include Eigen's Dense module
#include <stdexcept>   // For std::invalid_argument
#include <iostream>    // For debugging
#include "logging.h"   // Include logging header

/**
 * @brief Calculates the Jaccard composite index for a set of input vectors.
 *
 * @param CCx_loc_sum Eigen::VectorXd representing connected component location sums for feature x.
 * @param CCy_loc_sum Eigen::VectorXd representing connected component location sums for feature y.
 * @param feat_x Eigen::VectorXd representing feature x values (optional, for weighted Jaccard).
 * @param feat_y Eigen::VectorXd representing feature y values (optional, for weighted Jaccard).
 * @return double containing the Jaccard composite index.
 */
double jaccard_composite(const Eigen::VectorXd& CCx_loc_sum, 
                         const Eigen::VectorXd& CCy_loc_sum,
                         const Eigen::VectorXd* feat_x = nullptr,
                         const Eigen::VectorXd* feat_y = nullptr);

#endif // JACCARD_H