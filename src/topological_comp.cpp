#include "topological_comp.h"
#include "logging.h"  // For log_message
#include <limits>    // for std::numeric_limits
#include <algorithm> // for std::sort, std::find
#include <cmath>     // for M_PI
#include <stdexcept> // for std::invalid_argument
#include <tuple>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

// Function to compute adjacency matrix and Gaussian smoothing mask based on spatial locations
std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd> extract_adjacency_spatial(
    const Eigen::MatrixXd& loc, const std::string& spatial_type, double fwhm) {
    
    log_message("extract_adjacency_spatial: Starting with spatial_type=" + spatial_type + 
               ", fwhm=" + std::to_string(fwhm) + 
               ", loc shape=(" + std::to_string(loc.rows()) + "," + std::to_string(loc.cols()) + ")");
    
    int p = loc.rows();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd arr_mod = Eigen::MatrixXd::Zero(p, p);
    double sigma = fwhm / 2.355;

    try {
        if (spatial_type == "visium") {
            log_message("extract_adjacency_spatial: Processing visium data");
            // Calculate pairwise Euclidean distances
            for (int i = 0; i < p; ++i) {
                for (int j = i + 1; j < p; ++j) {
                    double dist = (loc.row(i) - loc.row(j)).norm();
                    A(i, j) = A(j, i) = dist;
                }
            }

            // Replace distances exceeding fwhm with infinity
            for (int i = 0; i < p; ++i) {
                for (int j = 0; j < p; ++j) {
                    if (A(i, j) > fwhm) {
                        A(i, j) = std::numeric_limits<double>::infinity();
                    }
                }
            }

            // Gaussian smoothing
            arr_mod = (1.0 / (2.0 * M_PI * sigma * sigma)) * (-A.array().square() / (2.0 * sigma * sigma)).exp();

            // Calculate minimum non-zero distance
            double min_distance = std::numeric_limits<double>::infinity();
            for (int i = 0; i < p; ++i) {
                for (int j = 0; j < p; ++j) {
                    if (A(i, j) > 0 && A(i, j) < min_distance && A(i, j) != std::numeric_limits<double>::infinity()) {
                        min_distance = A(i, j);
                    }
                }
            }

            // Create adjacency matrix based on minimum distances
            for (int i = 0; i < p; ++i) {
                for (int j = 0; j < p; ++j) {
                    A(i, j) = (A(i, j) > 0 && A(i, j) <= min_distance) ? 1.0 : 0.0;
                }
            }

            Eigen::SparseMatrix<double> A_sparse = A.sparseView();
            log_message("extract_adjacency_spatial: Completed visium processing");
            return std::make_tuple(A_sparse, arr_mod);
        } else if (spatial_type == "imageST") {
            log_message("extract_adjacency_spatial: Processing imageST data");
            // Determine grid size
            int rows = static_cast<int>(loc.col(1).maxCoeff()) + 1;
            int cols = static_cast<int>(loc.col(0).maxCoeff()) + 1;
            Eigen::SparseMatrix<double> adjacency(rows * cols, rows * cols);

            // Construct adjacency matrix for imageST
            std::vector<Eigen::Triplet<double>> tripletList;
            tripletList.reserve(loc.rows() * 4); // Estimate number of non-zeros

            for (int i = 0; i < loc.rows(); ++i) {
                int x = static_cast<int>(loc(i, 0));
                int y = static_cast<int>(loc(i, 1));
                int current = x * cols + y;

                // Connect to left neighbor
                if (x - 1 >= 0) {
                    int neighbor1 = (x - 1) * cols + y;
                    tripletList.emplace_back(current, neighbor1, 1.0);
                    tripletList.emplace_back(neighbor1, current, 1.0);
                }

                // Connect to top neighbor
                if (y - 1 >= 0) {
                    int neighbor2 = x * cols + (y - 1);
                    tripletList.emplace_back(current, neighbor2, 1.0);
                    tripletList.emplace_back(neighbor2, current, 1.0);
                }
            }

            adjacency.setFromTriplets(tripletList.begin(), tripletList.end());
            adjacency.makeCompressed(); // Optimize the sparse matrix

            // Subset the adjacency matrix to include only valid rows/cols
            std::vector<int> valid_indices;
            valid_indices.reserve(loc.rows());
            for (int i = 0; i < loc.rows(); ++i) {
                int index = static_cast<int>(loc(i, 1)) * cols + static_cast<int>(loc(i, 0));
                valid_indices.push_back(index);
            }

            // Create a mapping from old indices to new indices
            std::unordered_map<int, int> index_map;
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                index_map[valid_indices[i]] = static_cast<int>(i);
            }

            // Collect triplets for the subset adjacency matrix
            std::vector<Eigen::Triplet<double>> subset_triplets;
            subset_triplets.reserve(adjacency.nonZeros());

            for (int k = 0; k < adjacency.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency, k); it; ++it) {
                    int row = it.row();
                    int col = it.col();
                    if (index_map.find(row) != index_map.end() && index_map.find(col) != index_map.end()) {
                        subset_triplets.emplace_back(index_map[row], index_map[col], it.value());
                    }
                }
            }

            Eigen::SparseMatrix<double> adjacency_subset(valid_indices.size(), valid_indices.size());
            adjacency_subset.setFromTriplets(subset_triplets.begin(), subset_triplets.end());
            adjacency_subset.makeCompressed();

            log_message("extract_adjacency_spatial: Completed imageST processing");
            return std::make_tuple(adjacency_subset, Eigen::MatrixXd());
        } else {
            log_message("ERROR: Invalid spatial_type: " + spatial_type);
            throw std::invalid_argument("'spatial_type' not among ['visium', 'imageST']");
        }
    } catch (const std::exception& e) {
        log_message("ERROR in extract_adjacency_spatial: " + std::string(e.what()));
        throw; // Re-throw to be caught by the caller
    }
    
    // This should never be reached, but added to avoid compiler warnings
    return std::make_tuple(Eigen::SparseMatrix<double>(), Eigen::MatrixXd());
}

// Placeholder function definitions - Implement these properly
std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>, std::vector<int>> make_original_dendrogram_cc(
    const Eigen::VectorXd& tx, const Eigen::SparseMatrix<double>& A_sparse, const std::vector<double>& threshold_x) {
    // Implement the actual logic here
    return std::make_tuple(std::vector<std::vector<int>>(), std::vector<int>(), std::vector<int>(), std::vector<int>());
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> make_smoothed_dendrogram(
    const std::vector<std::vector<int>>& cCC_x, const std::vector<int>& cE_x, const std::vector<int>& cduration_x,
    const std::vector<int>& chistory_x, const Eigen::ArrayXd& linspaced) {
    // Implement the actual logic here
    return std::make_tuple(std::vector<int>(), std::vector<int>(), std::vector<int>());
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> make_dendrogram_bar(
    const std::vector<int>& chistory_x, const std::vector<int>& cduration_x) {
    // Implement the actual logic here
    return std::make_tuple(std::vector<int>(), std::vector<int>(), std::vector<int>(), std::vector<int>(), std::vector<int>(), std::vector<int>());
}

// Corrected function to extract connected components
std::vector<std::vector<int>> extract_connected_comp(
    const Eigen::VectorXd& tx, const Eigen::SparseMatrix<double>& A_sparse, 
    const std::vector<double>& threshold_x, int num_spots, int min_size) {
    
    auto [cCC_x, cE_x, cduration_x, chistory_x] = make_original_dendrogram_cc(tx, A_sparse, threshold_x);
    auto [nCC_x, nduration_x, nhistory_x] = make_smoothed_dendrogram(cCC_x, cE_x, cduration_x, chistory_x, Eigen::ArrayXd::LinSpaced(2, min_size, num_spots));
    auto [cvertical_x_x, cvertical_y_x, chorizontal_x_x, chorizontal_y_x, cdots_x, nlayer_x] = make_dendrogram_bar(chistory_x, cduration_x);

    std::vector<std::vector<int>> CCx;
    for (size_t i = 0; i < nlayer_x.size(); ++i) {
        CCx.emplace_back(std::vector<int>{nCC_x[i]});
    }
    return CCx;
}

// Function to extract the connected location matrix
Eigen::SparseMatrix<int> extract_connected_loc_mat(
    const std::vector<std::vector<int>>& CC, int num_spots, const std::string& format) {
    
    Eigen::MatrixXi CC_loc_arr = Eigen::MatrixXi::Zero(num_spots, CC.size());

    for (size_t num = 0; num < CC.size(); ++num) {
        const auto& element = CC[num];
        for (int idx : element) {
            if (idx >= 0 && idx < num_spots) { // Safety check
                CC_loc_arr(idx, num) = static_cast<int>(num) + 1;
            }
        }
    }

    if (format == "sparse") {
        return CC_loc_arr.sparseView();
    } else {
        throw std::invalid_argument("Sparse format required for compatibility.");
    }
}

// Adjusted function to filter connected component locations based on expression values
Eigen::SparseMatrix<int> filter_connected_loc_exp(
    const Eigen::SparseMatrix<int>& CC_loc_mat, const Eigen::VectorXd& feat_data, int thres_per) {
    
    // Cast CC_loc_mat to double for multiplication
    Eigen::VectorXd CC_mat_sum = CC_loc_mat.cast<double>() * Eigen::VectorXd::Ones(CC_loc_mat.cols());

    std::vector<std::pair<int, double>> CC_mean;
    for (int i = 0; i < CC_loc_mat.cols(); ++i) {
        double sum = CC_mat_sum(i);
        if (sum != 0) {
            CC_mean.emplace_back(i, feat_data(i));
        }
    }

    // Sort components based on expression values in descending order
    std::sort(CC_mean.begin(), CC_mean.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
    });

    // Determine cutoff based on threshold percentage
    int cutoff = static_cast<int>(CC_mean.size() * (1.0 - static_cast<double>(thres_per) / 100.0));
    if (cutoff < 0) cutoff = 0;
    if (cutoff > static_cast<int>(CC_mean.size())) cutoff = static_cast<int>(CC_mean.size());
    CC_mean.resize(cutoff);

    // Create a new sparse matrix with filtered components
    Eigen::SparseMatrix<int> CC_loc_mat_fin(CC_loc_mat.rows(), CC_mean.size());
    std::vector<Eigen::Triplet<int>> tripletList;
    tripletList.reserve(CC_loc_mat.nonZeros());

    for (size_t idx = 0; idx < CC_mean.size(); ++idx) {
        int original_col = CC_mean[idx].first;
        for (Eigen::SparseMatrix<int>::InnerIterator it(CC_loc_mat, original_col); it; ++it) {
            tripletList.emplace_back(it.row(), static_cast<int>(idx), it.value());
        }
    }

    CC_loc_mat_fin.setFromTriplets(tripletList.begin(), tripletList.end());
    CC_loc_mat_fin.makeCompressed();

    return CC_loc_mat_fin;
}

// Function for topological connected component analysis
std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>> topological_comp_res(
    const Eigen::VectorXd& feat, const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& mask,
    const std::string& spatial_type, int min_size, int thres_per, const std::string& return_mode) {

    if (return_mode != "all" && return_mode != "cc_loc" && return_mode != "jaccard_cc_list") {
        throw std::invalid_argument("'return_mode' should be among 'all', 'cc_loc', or 'jaccard_cc_list'");
    }

    int p = feat.size();

    Eigen::VectorXd smooth;
    if (spatial_type == "visium") {
        double feat_sum = feat.sum();
        if (feat_sum == 0) {
            throw std::invalid_argument("Sum of 'feat' vector is zero, cannot divide by zero.");
        }
        smooth = (mask * feat).array() / feat_sum;
    } else {
        smooth = feat;
    }

    Eigen::VectorXd t = smooth.cwiseMax(0);
    std::vector<double> threshold(t.data(), t.data() + t.size());
    std::sort(threshold.begin(), threshold.end(), std::greater<double>());
    threshold.erase(std::unique(threshold.begin(), threshold.end()), threshold.end());

    auto CC_list = extract_connected_comp(t, A, threshold, p, min_size);

    Eigen::SparseMatrix<int> CC_loc_mat = extract_connected_loc_mat(CC_list, p, "sparse");
    CC_loc_mat = filter_connected_loc_exp(CC_loc_mat, feat, thres_per);

    return std::make_tuple(CC_list, CC_loc_mat);
}