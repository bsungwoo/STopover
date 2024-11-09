#include "jaccard.h"          // Include the header file
#include <algorithm>          // For std::min, std::max
#include <iostream>           // Optional: For debugging or logging
#include <stdexcept>          // For std::invalid_argument

// Function to calculate the Jaccard composite index from arrays of connected component locations
Eigen::VectorXd jaccard_composite(const Eigen::VectorXd& CCx_loc_sum, 
                                  const Eigen::VectorXd& CCy_loc_sum,
                                  const Eigen::VectorXd& feat_x, 
                                  const Eigen::VectorXd& feat_y) {
    // Ensure all inputs are the same size
    if (CCx_loc_sum.size() != CCy_loc_sum.size() ||
        CCx_loc_sum.size() != feat_x.size() ||
        CCx_loc_sum.size() != feat_y.size()) {
        throw std::invalid_argument("All input vectors must have the same length.");
    }

    Eigen::VectorXd jaccard_indices(CCx_loc_sum.size());

    for (int i = 0; i < CCx_loc_sum.size(); ++i) {
        double x_loc = CCx_loc_sum[i];
        double y_loc = CCy_loc_sum[i];
        double x_feat = feat_x[i];
        double y_feat = feat_y[i];

        // Handle zero sums to avoid division by zero
        if (x_loc == 0.0 && y_loc == 0.0) {
            jaccard_indices[i] = 0.0;
            continue;
        }

        // If features are not provided (assuming zero indicates no feature)
        if (x_feat == 0.0 && y_feat == 0.0) {
            // Compute standard Jaccard index
            double intersection = std::min(x_loc, y_loc);
            double union_sum = x_loc + y_loc - intersection;
            jaccard_indices[i] = (union_sum != 0.0) ? (1.0 - (intersection / union_sum)) : 0.0;
        } else {
            // Compute weighted Jaccard index
            double min_feat = std::min(x_feat, y_feat);
            double max_feat = std::max(x_feat, y_feat);

            if (max_feat - min_feat == 0.0) {
                jaccard_indices[i] = 0.0;
                continue;
            }

            double norm_x_feat = (x_feat - min_feat) / (max_feat - min_feat);
            double norm_y_feat = (y_feat - min_feat) / (max_feat - min_feat);

            double min_sum = std::min(norm_x_feat, norm_y_feat);
            double max_sum = std::max(norm_x_feat, norm_y_feat);

            jaccard_indices[i] = (max_sum != 0.0) ? (min_sum / max_sum) : 0.0;
        }
    }

    return jaccard_indices;
}