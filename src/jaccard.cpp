#include "jaccard.h"
#include <algorithm> // For std::min, std::max
#include <iostream>  // Optional: For debugging or logging
#include <stdexcept> // For std::invalid_argument

// Function to calculate the Jaccard composite index from arrays of connected component locations
double jaccard_composite(const Eigen::VectorXd& CCx_loc_sum, 
                         const Eigen::VectorXd& CCy_loc_sum,
                         const Eigen::VectorXd& feat_x, 
                         const Eigen::VectorXd& feat_y) {
    // Ensure all inputs are the same size
    if (CCx_loc_sum.size() != CCy_loc_sum.size() ||
        CCx_loc_sum.size() != feat_x.size() ||
        CCx_loc_sum.size() != feat_y.size()) {
        throw std::invalid_argument("All input vectors must have the same length.");
    }

    double intersection = 0.0;
    double union_sum = 0.0;

    for (int i = 0; i < CCx_loc_sum.size(); ++i) {
        double x_loc = CCx_loc_sum[i];
        double y_loc = CCy_loc_sum[i];
        double x_feat = feat_x[i];
        double y_feat = feat_y[i];

        // Handle zero sums to avoid division by zero
        if (x_loc == 0.0 && y_loc == 0.0) {
            continue; // No contribution to intersection or union
        }

        // If features are not provided (assuming zero indicates no feature)
        if (x_feat == 0.0 && y_feat == 0.0) {
            // Compute standard Jaccard index for this element
            double min_val = std::min(x_loc, y_loc);
            double max_val = std::max(x_loc, y_loc);
            intersection += min_val;
            union_sum += max_val;
        } else {
            // Compute weighted Jaccard index for this element
            double min_feat = std::min(x_feat, y_feat);
            double max_feat = std::max(x_feat, y_feat);

            // Handle the case where all feature values are equal
            if (max_feat - min_feat == 0.0) {
                continue; // Cannot normalize, skip this element
            }

            double norm_x_feat = (x_feat - min_feat) / (max_feat - min_feat);
            double norm_y_feat = (y_feat - min_feat) / (max_feat - min_feat);

            double min_sum = std::min(norm_x_feat, norm_y_feat);
            double max_sum = std::max(norm_x_feat, norm_y_feat);

            if (max_sum != 0.0) {
                intersection += min_sum;
                union_sum += max_sum;
            }
        }
    }

    // Calculate and return Jaccard index
    if (union_sum == 0.0) {
        return 0.0; // Handle edge case where there's no union
    } else {
        return intersection / union_sum;
    }
}