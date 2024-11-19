#include "topological_comp.h"
#include "make_original_dendrogram.h"
#include "make_smoothed_dendrogram.h"
#include "make_dendrogram_bar.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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
                // Round to 4 decimal places
                double dist_rounded = std::round(dist * 10000.0) / 10000.0;
                A(i, j) = A(j, i) = dist_rounded;
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
        arr_mod = (1.0 / (2.0 * M_PI * sigma * sigma)) *
                  (-A.array().square() / (2.0 * sigma * sigma)).exp();

        // Find minimum non-zero finite distance
        double min_distance = std::numeric_limits<double>::max();
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < p; ++j) {
                double value = A(i, j);
                if (value > 0 && std::isfinite(value) && value < min_distance) {
                    min_distance = value;
                }
            }
        }

        // Create adjacency matrix based on minimum distances
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < p; ++j) {
                double value = A(i, j);
                if (value > 0 && value <= min_distance && std::isfinite(value)) {
                    A(i, j) = 1.0;
                } else {
                    A(i, j) = 0.0;
                }
            }
        }

        Eigen::SparseMatrix<double> A_sparse = A.sparseView();
        return std::make_tuple(A_sparse, arr_mod);
    } else if (spatial_type == "imageST" || spatial_type == "visiumHD") {
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
        throw std::invalid_argument("'spatial_type' not among ['visium', 'imageST', 'visiumHD']");
    }
}

// Function to smooth feature vector following Python's smoothing logic
Eigen::VectorXd smooth_feature_vector_python_style(const Eigen::VectorXd& feat, const Eigen::MatrixXd& mask) {
    int p = feat.size();
    Eigen::MatrixXd feat_tiled = feat.replicate(1, p); // p x p
    Eigen::MatrixXd multiplied = mask.array() * feat_tiled.array(); // p x p
    Eigen::VectorXd sum_per_column = multiplied.colwise().sum(); // p x 1

    double sum_smooth = sum_per_column.sum();
    double sum_feat = feat.sum();

    if (sum_smooth == 0) {
        throw std::invalid_argument("Sum of smoothed feature vector is zero, cannot divide by zero.");
    }

    Eigen::VectorXd smooth = sum_per_column.array() / sum_smooth * sum_feat;
    return smooth;
}

// Function to compute thresholds similar to np.setdiff1d(t, 0) sorted descendingly
std::vector<double> compute_thresholds_python_style(const Eigen::VectorXd& t) {
    const double EPSILON = 1e-9;  // Define a small threshold value
    std::vector<double> threshold;
    for (int i = 0; i < t.size(); ++i) {
        if (t[i] > EPSILON) {  // Only add values significantly greater than zero
            threshold.push_back(t[i]);
        }
    }
    std::sort(threshold.begin(), threshold.end(), std::greater<double>());
    // Remove duplicates
    threshold.erase(std::unique(threshold.begin(), threshold.end()), threshold.end());
    return threshold;
}

// Function to extract connected components similar to Python's extract_connected_comp
std::vector<std::vector<int>> extract_connected_comp_python_style(
    const Eigen::VectorXd& tx, 
    const Eigen::SparseMatrix<double>& A_sparse, 
    const std::vector<double>& threshold_x, 
    int num_spots, 
    int min_size) {

    // Compute connected components using make_original_dendrogram_cc
    auto [cCC_x, cE_x, cduration_x, chistory_x] = make_original_dendrogram_cc(tx, A_sparse, threshold_x);

    // Smooth the dendrogram
    auto [nCC_x, nE_x, nduration_x, nhistory_x] = make_smoothed_dendrogram(cCC_x, cE_x, cduration_x, chistory_x, Eigen::Vector2d(min_size, num_spots));

    // Estimate dendrogram bars for plotting
    Eigen::MatrixXd cvertical_x_x, cvertical_y_x, chorizontal_x_x, chorizontal_y_x, cdots_x;
    std::vector<std::vector<int>> clayer_x;
    std::tie(cvertical_x_x, cvertical_y_x, chorizontal_x_x, chorizontal_y_x, cdots_x, clayer_x) = 
        make_dendrogram_bar(chistory_x, cduration_x);

    // Estimate smoothed dendrogram bars
    Eigen::MatrixXd cvertical_x_new, cvertical_y_new, chorizontal_x_new, chorizontal_y_new, cdots_new;
    std::vector<std::vector<int>> nlayer_x;
    std::tie(cvertical_x_new, cvertical_y_new, chorizontal_x_new, chorizontal_y_new, cdots_new, nlayer_x) = 
        make_dendrogram_bar(nhistory_x, nduration_x, cvertical_x_x, cvertical_y_x, chorizontal_x_x, chorizontal_y_x, cdots_x);

    // Extract connected components based on layer information
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
Eigen::SparseMatrix<double> extract_connected_loc_mat_python_style(
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

// Function to filter connected components based on expression percentile
Eigen::SparseMatrix<double> filter_connected_loc_exp_python_style(
    const Eigen::SparseMatrix<double>& CC_loc_mat,
    const Eigen::VectorXd& feat_data,
    int thres_per) {

    // Step 1: Sum CC_loc_mat across columns to get CC_mat_sum (p x 1 vector)
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(CC_loc_mat.cols());
    Eigen::VectorXd CC_mat_sum = CC_loc_mat * ones;

    // Step 2: Combine CC_mat_sum and feat_data
    // Not necessary to create a combined matrix; we can process them directly

    // Step 3: Group by CC_mat_sum (connected component labels) and compute mean of feat_data
    std::unordered_map<int, std::vector<double>> CC_groups;
    CC_groups.reserve(CC_loc_mat.cols()); // Reserve space based on number of components

    for (int i = 0; i < CC_mat_sum.size(); ++i) {
        int label = static_cast<int>(CC_mat_sum(i));
        if (label != 0) { // Exclude label 0
            CC_groups[label].emplace_back(feat_data(i));
        }
    }

    // Step 4: Compute mean feature value per connected component
    std::vector<std::pair<int, double>> CC_mean;
    CC_mean.reserve(CC_groups.size());

    for (const auto& pair : CC_groups) {
        int label = pair.first;
        const std::vector<double>& values = pair.second;
        double sum = std::accumulate(values.begin(), values.end(), 0.0);
        double mean_value = sum / values.size();
        CC_mean.emplace_back(label, mean_value);
    }

    // Step 5: Sort connected components by mean values in descending order
    std::sort(CC_mean.begin(), CC_mean.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second > b.second;
              });

    // Step 6: Determine number of components to keep based on threshold
    int num_components_to_keep = static_cast<int>(CC_mean.size() * (1.0 - (thres_per / 100.0)));
    num_components_to_keep = std::max(0, std::min(num_components_to_keep, static_cast<int>(CC_mean.size())));
    CC_mean.resize(num_components_to_keep);

    // Step 7: Get indices of components to keep (zero-based)
    std::vector<int> components_to_keep;
    components_to_keep.reserve(CC_mean.size());

    for (const auto& pair : CC_mean) {
        int label = pair.first;
        components_to_keep.emplace_back(label - 1); // Convert to zero-based index
    }

    std::sort(components_to_keep.begin(), components_to_keep.end());

    // Step 8: Filter CC_loc_mat accordingly
    Eigen::SparseMatrix<double> CC_loc_mat_fin(CC_loc_mat.rows(), components_to_keep.size());
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(CC_loc_mat.nonZeros());

    for (size_t idx = 0; idx < components_to_keep.size(); ++idx) {
        int col_idx = components_to_keep[idx];
        for (Eigen::SparseMatrix<double>::InnerIterator it(CC_loc_mat, col_idx); it; ++it) {
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

    // Smooth the feature values with given mask, following Python logic
    Eigen::VectorXd smooth = smooth_feature_vector_python_style(feat, mask);

    // Apply t = smooth*(smooth > 0)
    Eigen::VectorXd t = smooth.array() * (smooth.array() > 0).cast<double>();

    // Compute unique nonzero thresholds in descending order
    std::vector<double> threshold = compute_thresholds_python_style(t);

    // Compute connected components
    std::vector<std::vector<int>> CC_list = extract_connected_comp_python_style(t, A, threshold, feat.size(), min_size);

    // Extract location of connected components as sparse matrix
    Eigen::SparseMatrix<double> CC_loc_mat = extract_connected_loc_mat_python_style(CC_list, feat.size(), "sparse");

    // Filter connected components based on feature expression percentile
    CC_loc_mat = filter_connected_loc_exp_python_style(CC_loc_mat, feat, thres_per);

    // Compute row sums
    Eigen::VectorXd row_sums = CC_loc_mat * Eigen::VectorXd::Ones(CC_loc_mat.cols());

    return row_sums;
}