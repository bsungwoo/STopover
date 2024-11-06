#include "jaccard.h"

// Function to calculate the Jaccard composite index from connected component locations
double jaccard_composite(const Eigen::MatrixXd& CCx_loc_sum, const Eigen::MatrixXd& CCy_loc_sum,
                         const Eigen::MatrixXd& feat_x = Eigen::MatrixXd(), const Eigen::MatrixXd& feat_y = Eigen::MatrixXd()) {
    Eigen::MatrixXd CCxy_loc_sum;

    if (CCx_loc_sum.rows() != CCy_loc_sum.rows()) {
        throw std::invalid_argument("'CCx_loc_sum' and 'CCy_loc_sum' must have the same number of rows");
    }

    // Combine the locations for features x and y
    CCxy_loc_sum.resize(CCx_loc_sum.rows(), 2);
    CCxy_loc_sum << CCx_loc_sum, CCy_loc_sum;

    if (CCxy_loc_sum.nonZeros() == 0) {
        return 0.0;
    }

    // If features are not provided, calculate the default Jaccard index
    if (feat_x.size() == 0 && feat_y.size() == 0) {
        // Compute Jaccard using basic logic
        // In C++, we don't have a built-in pdist function. You'll need to implement a custom pairwise Jaccard calculation.
        Eigen::MatrixXd binary_mat = (CCxy_loc_sum.array() != 0).cast<double>();
        double intersection = (binary_mat.col(0).array() * binary_mat.col(1).array()).sum();
        double union_sum = binary_mat.array().rowwise().any().cast<double>().sum();

        return 1.0 - (intersection / union_sum);
    } else {
        // Weighted Jaccard index
        Eigen::MatrixXd feat_val(CCx_loc_sum.rows(), 2);
        feat_val << feat_x, feat_y;

        // Normalize the feature values
        feat_val = (CCxy_loc_sum.array() != 0).select(feat_val.array(), 0);
        feat_val.col(0) = (feat_val.col(0).array() - feat_val.col(0).minCoeff()) / (feat_val.col(0).maxCoeff() - feat_val.col(0).minCoeff());
        feat_val.col(1) = (feat_val.col(1).array() - feat_val.col(1).minCoeff()) / (feat_val.col(1).maxCoeff() - feat_val.col(1).minCoeff());

        return (feat_val.rowwise().minCoeff().sum()) / (feat_val.rowwise().maxCoeff().sum());
    }
}