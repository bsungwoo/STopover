#include "jaccard.h"
#include <algorithm> // For std::min, std::max

// Function to calculate the Jaccard composite index from connected component locations
double jaccard_composite(double CCx_loc_sum, double CCy_loc_sum, double feat_x, double feat_y) {
    // Check for zero sums to avoid division by zero
    if (CCx_loc_sum == 0.0 && CCy_loc_sum == 0.0) {
        return 0.0;
    }

    // If features are not provided (assuming zero indicates no feature)
    if (feat_x == 0.0 && feat_y == 0.0) {
        // Compute standard Jaccard index
        double intersection = std::min(CCx_loc_sum, CCy_loc_sum);
        double union_sum = CCx_loc_sum + CCy_loc_sum - intersection;
        return (union_sum != 0.0) ? (1.0 - (intersection / union_sum)) : 0.0;
    } else {
        // Compute weighted Jaccard index
        // Normalize feature values between 0 and 1 to prevent division by zero
        double min_feat = std::min(feat_x, feat_y);
        double max_feat = std::max(feat_x, feat_y);

        // Handle the case where all feature values are equal
        if (max_feat - min_feat == 0.0) {
            return 0.0;
        }

        double norm_feat_x = (feat_x - min_feat) / (max_feat - min_feat);
        double norm_feat_y = (feat_y - min_feat) / (max_feat - min_feat);

        double min_sum = std::min(norm_feat_x, norm_feat_y);
        double max_sum = std::max(norm_feat_x, norm_feat_y);

        return (max_sum != 0.0) ? (min_sum / max_sum) : 0.0;
    }
}