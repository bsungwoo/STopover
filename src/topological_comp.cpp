#include "topological_comp.h"
#include "make_original_dendrogram.h"
#include "make_smoothed_dendrogram.h"
#include "make_dendrogram_bar.h"
#include <limits>    // for std::numeric_limits
#include <algorithm> // for std::sort, std::find
#include <cmath>     // for M_PI
#include <stdexcept> // for std::invalid_argument
#include <tuple>
#include <vector>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>


// Function to compute adjacency matrix and Gaussian smoothing mask based on spatial locations
std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd> extract_adjacency_spatial(
    const Eigen::MatrixXd& loc, const std::string& spatial_type, double fwhm) {
    
    int p = loc.rows();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd arr_mod = Eigen::MatrixXd::Zero(p, p);
    double sigma = fwhm / 2.355;

    if (spatial_type == "visium") {
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
        return std::make_tuple(A_sparse, arr_mod);
    } else if (spatial_type == "imageST") {
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

        return std::make_tuple(adjacency_subset, Eigen::MatrixXd());
    } else {
        throw std::invalid_argument("'spatial_type' not among ['visium', 'imageST']");
    }
}

std::vector<std::vector<int>> extract_connected_comp(
    const Eigen::VectorXd& tx, 
    const Eigen::SparseMatrix<double>& A_sparse, 
    const std::vector<double>& threshold_x, 
    int num_spots, 
    int min_size) {
    
    // Step 1: Compute Connected Components
    auto [cCC_x, cE_x, cduration_x, chistory_x] = make_original_dendrogram_cc(tx, A_sparse, threshold_x);

    // Step 2: Smooth the Dendrogram
    auto [nCC_x, nE_x, nduration_x, nhistory_x] = make_smoothed_dendrogram(
        cCC_x, 
        cE_x, 
        cduration_x, 
        chistory_x, 
        Eigen::Vector2d(min_size, num_spots)
    );
    
    // Step 3a: Estimate Initial Dendrogram Bars for Plotting
    // Call make_dendrogram_bar with original history and duration
    Eigen::MatrixXd cvertical_x_x, cvertical_y_x, chorizontal_x_x, chorizontal_y_x, cdots_x;
    std::vector<std::vector<int>> clayer_x;
    std::tie(cvertical_x_x, cvertical_y_x, chorizontal_x_x, chorizontal_y_x, cdots_x, clayer_x) = 
        make_dendrogram_bar(chistory_x, cduration_x, Eigen::MatrixXd(), Eigen::MatrixXd(), Eigen::MatrixXd(), Eigen::MatrixXd(), Eigen::MatrixXd());
    
    // Step 3b: Estimate Smoothed Dendrogram Bars for Plotting
    // Call make_dendrogram_bar with smoothed history and duration, along with initial bar coordinates
    Eigen::MatrixXd cvertical_x_new, cvertical_y_new, chorizontal_x_new, chorizontal_y_new, cdots_new;
    std::vector<std::vector<int>> nlayer_x;
    std::tie(cvertical_x_new, cvertical_y_new, chorizontal_x_new, chorizontal_y_new, cdots_new, nlayer_x) = 
        make_dendrogram_bar(
            nhistory_x, 
            nduration_x, 
            cvertical_x_x, 
            cvertical_y_x, 
            chorizontal_x_x, 
            chorizontal_y_x, 
            cdots_x
        );

    // Step 4: Extract Connected Components Based on Layer Information
    // Ensure that nlayer_x has at least one layer
    if (nlayer_x.empty() || nlayer_x[0].empty()) {
        // No connected components found; return an empty vector
        return {};
    }
    
    // Extract the first layer indices
    std::vector<int> sind = nlayer_x[0];
    std::vector<std::vector<int>> CCx;
    
    // Populate CCx with the connected components corresponding to sind
    for (const auto& i : sind) {
        if (i >= 0 && i < static_cast<int>(nCC_x.size())) { // Validate index
            CCx.emplace_back(nCC_x[i]);
        } else {
            // Handle invalid indices if necessary
            std::cerr << "Warning: Index " << i << " is out of bounds for nCC_x with size " << nCC_x.size() << ". Skipping.\n";
        }
    }
    
    return CCx;
}

// Function to extract the connected location matrix
Eigen::SparseMatrix<double> extract_connected_loc_mat(
    const std::vector<std::vector<int>>& CC, int num_spots, const std::string& format) {
    
    Eigen::MatrixXd CC_loc_arr = Eigen::MatrixXd::Zero(num_spots, CC.size());

    for (size_t num = 0; num < CC.size(); ++num) {
        const auto& element = CC[num];
        for (int idx : element) {
            if (idx >= 0 && idx < num_spots) { // Safety check
                CC_loc_arr(idx, num) = static_cast<double>(num) + 1.0; // Use double
            }
        }
    }

    if (format == "sparse") {
        return CC_loc_arr.sparseView();
    } else {
        throw std::invalid_argument("Sparse format required for compatibility.");
    }
}

Eigen::SparseMatrix<double> filter_connected_loc_exp(
    const Eigen::SparseMatrix<double>& CC_loc_mat, 
    const Eigen::VectorXd& feat_data, 
    int thres_per) {
    
    // Vector to store the mean expression value of each connected component
    std::vector<std::pair<int, double>> CC_mean;
    
    for (int i = 0; i < CC_loc_mat.cols(); ++i) {
        // Get the indices of the spots in this connected component
        std::vector<int> indices;
        for (Eigen::SparseMatrix<double>::InnerIterator it(CC_loc_mat, i); it; ++it) {
            indices.push_back(it.row());
        }
        if (!indices.empty()) {
            // Compute the mean of feat_data over these indices
            double sum = 0.0;
            for (int idx : indices) {
                sum += feat_data(idx);
            }
            double mean_value = sum / indices.size();
            CC_mean.emplace_back(i, mean_value);
        }
    }

    // Sort components based on expression values in descending order
    std::sort(CC_mean.begin(), CC_mean.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) -> bool {
        return a.second > b.second;
    });

    // Determine cutoff based on threshold percentage
    int cutoff = static_cast<int>(CC_mean.size() * (static_cast<double>(thres_per) / 100.0));
    if (cutoff < 0) cutoff = 0;
    if (cutoff > static_cast<int>(CC_mean.size())) cutoff = static_cast<int>(CC_mean.size());
    CC_mean.resize(cutoff);

    // Create a new sparse matrix with filtered components
    Eigen::SparseMatrix<double> CC_loc_mat_fin(CC_loc_mat.rows(), cutoff);
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(CC_loc_mat.nonZeros());

    for (size_t idx = 0; idx < CC_mean.size(); ++idx) {
        int original_col = CC_mean[idx].first;
        for (Eigen::SparseMatrix<double>::InnerIterator it(CC_loc_mat, original_col); it; ++it) {
            tripletList.emplace_back(it.row(), static_cast<int>(idx), it.value());
        }
    }

    CC_loc_mat_fin.setFromTriplets(tripletList.begin(), tripletList.end());
    CC_loc_mat_fin.makeCompressed();

    return CC_loc_mat_fin;
}

// Function for topological connected component analysis
Eigen::VectorXd topological_comp_res(
    const Eigen::MatrixXd& loc, const std::string& spatial_type, double fwhm,
    const Eigen::VectorXd& feat, int min_size, int thres_per, const std::string& return_mode) {

    if (return_mode != "all" && return_mode != "cc_loc" && return_mode != "jaccard_cc_list") {
        throw std::invalid_argument("'return_mode' should be among 'all', 'cc_loc', or 'jaccard_cc_list'");
    }

    // Extract adjacency matrix and mask
    auto [A, mask] = extract_adjacency_spatial(loc, spatial_type, fwhm);

    // Smooth the feature values with given mask
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

    Eigen::SparseMatrix<double> CC_loc_mat = extract_connected_loc_mat(CC_list, p, "sparse");
    CC_loc_mat = filter_connected_loc_exp(CC_loc_mat, feat, thres_per);
    
    Eigen::VectorXd row_sums = CC_loc_mat * Eigen::VectorXd::Ones(CC_loc_mat.cols());

    return row_sums;
}