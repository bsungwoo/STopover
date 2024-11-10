#include "jaccard.h"
#include <iostream>  // Optional: For debugging or logging
#include <stdexcept> // For std::invalid_argument

// Function to calculate the Jaccard composite index
double jaccard_composite(const Eigen::VectorXd& CCx_loc_sum, 
                         const Eigen::VectorXd& CCy_loc_sum,
                         const Eigen::VectorXd* feat_x,
                         const Eigen::VectorXd* feat_y) {
    // Ensure input vectors have the same size
    if (CCx_loc_sum.size() != CCy_loc_sum.size()) {
        throw std::invalid_argument("CCx_loc_sum and CCy_loc_sum must have the same length.");
    }
    
    const int N = CCx_loc_sum.size();

    // Concatenate CCx_loc_sum and CCy_loc_sum into a matrix
    Eigen::MatrixXd CCxy_loc_sum(N, 2);
    CCxy_loc_sum.col(0) = CCx_loc_sum;
    CCxy_loc_sum.col(1) = CCy_loc_sum;

    // Check if all elements are zero
    if ((CCxy_loc_sum.array() != 0.0).count() == 0) {
        return 0.0;
    }

    // If feature vectors are not provided
    if (feat_x == nullptr && feat_y == nullptr) {
        // Convert CCxy_loc_sum to binary (non-zero elements are 1)
        Eigen::ArrayXXi CCxy_binary = (CCxy_loc_sum.array() != 0.0).cast<int>();

        // Extract the two binary columns
        Eigen::ArrayXi col1 = CCxy_binary.col(0);
        Eigen::ArrayXi col2 = CCxy_binary.col(1);

        // Compute intersection and union
        int intersection = ( (col1 == 1) && (col2 == 1) ).count();
        int union_count = ( (col1 == 1) || (col2 == 1) ).count();

        // Calculate Jaccard similarity
        if (union_count == 0) {
            return 0.0;
        } else {
            return static_cast<double>(intersection) / static_cast<double>(union_count);
        }
    } else {
        // Ensure both feature vectors are provided
        if (feat_x == nullptr || feat_y == nullptr) {
            throw std::invalid_argument("Both feat_x and feat_y must be provided.");
        }

        // Ensure feature vectors have the correct size
        if (feat_x->size() != N || feat_y->size() != N) {
            throw std::invalid_argument("Feature vectors must have the same length as CCx_loc_sum and CCy_loc_sum.");
        }

        // Concatenate feat_x and feat_y into a matrix
        Eigen::MatrixXd feat_val(N, 2);
        feat_val.col(0) = *feat_x;
        feat_val.col(1) = *feat_y;

        // Compute the minimum and maximum per column
        Eigen::RowVectorXd min_col = feat_val.colwise().minCoeff();
        Eigen::RowVectorXd max_col = feat_val.colwise().maxCoeff();

        // Compute the denominator for normalization
        Eigen::RowVectorXd denom_col = max_col - min_col;

        // Initialize normalized feature matrix
        Eigen::MatrixXd feat_val_normalized = Eigen::MatrixXd::Zero(N, 2);

        // Normalize features per column, handling zero denominators
        for (int i = 0; i < 2; ++i) {
            if (denom_col(i) == 0.0) {
                // Avoid division by zero; all values are the same
                feat_val_normalized.col(i).setZero();
            } else {
                feat_val_normalized.col(i) = (feat_val.col(i).array() - min_col(i)) / denom_col(i);
            }
        }

        // Create a mask for non-zero connected component locations
        Eigen::ArrayXXd mask = (CCxy_loc_sum.array() != 0.0).cast<double>();

        // Apply the mask to the normalized features
        feat_val_normalized = feat_val_normalized.array() * mask;

        // Compute the minimum and maximum per row
        Eigen::VectorXd feat_min = feat_val_normalized.rowwise().minCoeff();
        Eigen::VectorXd feat_max = feat_val_normalized.rowwise().maxCoeff();

        // Sum over all elements
        double sum_min = feat_min.sum();
        double sum_max = feat_max.sum();

        // Calculate Jaccard composite index
        if (sum_max == 0.0) {
            return 0.0;
        } else {
            return sum_min / sum_max;
        }
    }
}