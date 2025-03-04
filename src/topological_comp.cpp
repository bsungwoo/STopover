#include "topological_comp.h"
#include "make_original_dendrogram.h"
#include "make_smoothed_dendrogram.h"
#include "make_dendrogram_bar.h"
#include "logging.h"
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
    const double EPSILON = 1e-8;  // Define a small threshold value
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
Eigen::SparseMatrix<double> extract_connected_loc_mat(
    const std::vector<std::vector<int>>& CC, int num_spots, const std::string& format) {
    
    log_message("extract_connected_loc_mat: Starting with " + std::to_string(CC.size()) + 
               " connected components, num_spots=" + std::to_string(num_spots));
    
    try {
        // Create triplet list for sparse matrix
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        
        // Estimate number of non-zeros
        int estimated_nnz = 0;
        for (const auto& cc : CC) {
            estimated_nnz += cc.size();
        }
        tripletList.reserve(estimated_nnz);
        
        // Fill triplet list
        for (size_t i = 0; i < CC.size(); ++i) {
            for (const auto& spot : CC[i]) {
                if (spot >= 0 && spot < num_spots) {
                    tripletList.push_back(T(spot, i, 1.0)); // Use 1.0 (double) not 1 (int)
                }
            }
        }
        
        // Create sparse matrix
        Eigen::SparseMatrix<double> CC_loc_mat(num_spots, CC.size());
        CC_loc_mat.setFromTriplets(tripletList.begin(), tripletList.end());
        
        log_message("extract_connected_loc_mat: Created matrix with " + 
                   std::to_string(CC_loc_mat.nonZeros()) + " non-zeros");
        
        return CC_loc_mat;
    } catch (const std::exception& e) {
        log_message("ERROR in extract_connected_loc_mat: " + std::string(e.what()));
        throw; // Re-throw to be caught by the caller
    }
}

// Function to filter connected components based on expression percentile
Eigen::SparseMatrix<double> filter_connected_loc_exp(
    const Eigen::SparseMatrix<double>& CC_loc_mat, const Eigen::VectorXd& feat_data, double thres_per) {
    
    log_message("filter_connected_loc_exp: Starting with matrix size=" + 
               std::to_string(CC_loc_mat.rows()) + "x" + std::to_string(CC_loc_mat.cols()) + 
               ", thres_per=" + std::to_string(thres_per));
    
    try {
        // Calculate percentile threshold
        std::vector<double> feat_vec(feat_data.data(), feat_data.data() + feat_data.size());
        std::sort(feat_vec.begin(), feat_vec.end());
        int idx = static_cast<int>(feat_vec.size() * thres_per / 100.0);
        double threshold = feat_vec[idx];
        
        log_message("filter_connected_loc_exp: Threshold at " + std::to_string(thres_per) + 
                   "% percentile = " + std::to_string(threshold));
        
        // Create filtered matrix
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        
        // Estimate number of non-zeros
        tripletList.reserve(CC_loc_mat.nonZeros());
        
        // Filter components based on feature expression
        for (int k = 0; k < CC_loc_mat.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(CC_loc_mat, k); it; ++it) {
                int row = it.row();
                int col = it.col();
                
                if (feat_data(row) >= threshold) {
                    tripletList.push_back(T(row, col, 1.0)); // Use 1.0 (double) not 1 (int)
                }
            }
        }
        
        // Create sparse matrix
        Eigen::SparseMatrix<double> filtered_mat(CC_loc_mat.rows(), CC_loc_mat.cols());
        filtered_mat.setFromTriplets(tripletList.begin(), tripletList.end());
        
        log_message("filter_connected_loc_exp: Created filtered matrix with " + 
                   std::to_string(filtered_mat.nonZeros()) + " non-zeros");
        
        return filtered_mat;
    } catch (const std::exception& e) {
        log_message("ERROR in filter_connected_loc_exp: " + std::string(e.what()));
        throw; // Re-throw to be caught by the caller
    }
}

// Function for topological connected component analysis
Eigen::VectorXd topological_comp_res(
    const Eigen::MatrixXd& loc, const std::string& spatial_type, double fwhm,
    const Eigen::VectorXd& feat, int min_size, double thres_per, const std::string& return_mode) {
    
    log_message("topological_comp_res: Starting with spatial_type=" + spatial_type + 
               ", fwhm=" + std::to_string(fwhm) + 
               ", min_size=" + std::to_string(min_size) + 
               ", thres_per=" + std::to_string(thres_per) + 
               ", return_mode=" + return_mode);
    log_message("Input dimensions: loc=" + std::to_string(loc.rows()) + "x" + std::to_string(loc.cols()) + 
               ", feat=" + std::to_string(feat.size()));
    
    try {
        // Extract adjacency matrix and mask
        log_message("Calling extract_adjacency_spatial...");
        auto [A, mask] = extract_adjacency_spatial(loc, spatial_type, fwhm);
        log_message("extract_adjacency_spatial completed. A has " + std::to_string(A.nonZeros()) + " non-zeros");
        
        // Check if adjacency matrix is valid
        if (A.nonZeros() == 0) {
            log_message("topological_comp_res: Empty adjacency matrix, returning empty result");
            return Eigen::VectorXd::Zero(loc.rows());
        }
        
        // Smooth the feature values
        Eigen::VectorXd smooth;
        try {
            log_message("Starting feature smoothing...");
            // Apply Gaussian smoothing using the mask
            if (mask.size() > 0) {
                // Following the original algorithm's smoothing approach
                Eigen::MatrixXd feat_tiled = feat.replicate(1, feat.size());
                Eigen::MatrixXd multiplied = mask.array() * feat_tiled.array();
                Eigen::VectorXd sum_per_column = multiplied.colwise().sum();
                
                double sum_smooth = sum_per_column.sum();
                double sum_feat = feat.sum();
                
                log_message("Smoothing stats: sum_smooth=" + std::to_string(sum_smooth) + 
                           ", sum_feat=" + std::to_string(sum_feat));
                
                if (sum_smooth > 0) {
                    smooth = sum_per_column.array() / sum_smooth * sum_feat;
                } else {
                    log_message("sum_smooth is zero, using original features");
                    smooth = feat; // Fallback to original features if smoothing fails
                }
            } else {
                // If no mask, use original feature values
                log_message("No mask available, using original features");
                smooth = feat;
            }
            
            log_message("Smoothing completed, smooth vector sum = " + std::to_string(smooth.sum()));
        } catch (const std::exception& e) {
            log_message("ERROR in smoothing: " + std::string(e.what()) + ", using original features");
            smooth = feat;
        }
        
        // Apply t = smooth*(smooth > 0)
        log_message("Creating thresholded vector t...");
        Eigen::VectorXd t = Eigen::VectorXd::Zero(smooth.size());
        for (int i = 0; i < smooth.size(); ++i) {
            t(i) = smooth(i) > 0 ? smooth(i) : 0;
        }
        
        // Compute unique nonzero thresholds in descending order
        log_message("Computing unique thresholds...");
        std::vector<double> threshold;
        for (int i = 0; i < t.size(); ++i) {
            if (t(i) > 0) {
                threshold.push_back(t(i));
            }
        }
        
        // Sort thresholds in descending order
        std::sort(threshold.begin(), threshold.end(), std::greater<double>());
        
        // Remove duplicates
        threshold.erase(std::unique(threshold.begin(), threshold.end()), threshold.end());
        
        log_message("Found " + std::to_string(threshold.size()) + " unique thresholds");
        
        // If no thresholds, return empty result
        if (threshold.empty()) {
            log_message("No positive thresholds found, returning empty result");
            return Eigen::VectorXd::Zero(loc.rows());
        }
        
        // Extract connected components
        std::vector<std::vector<int>> CC_list;
        try {
            log_message("Calling make_original_dendrogram_cc...");
            auto [cCC_x, cE_x, cduration_x, chistory_x] = make_original_dendrogram_cc(t, A, threshold);
            log_message("make_original_dendrogram_cc completed successfully");
            
            // Smooth the dendrogram
            auto [nCC_x, nE_x, nduration_x, nhistory_x] = make_smoothed_dendrogram(cCC_x, cE_x, cduration_x, chistory_x, Eigen::Vector2d(min_size, loc.rows()));

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
                return Eigen::VectorXd::Zero(loc.rows());
            }

            // Extract the first layer indices
            std::vector<int> sind = nlayer_x[0];
            CC_list.clear();
            CC_list.reserve(sind.size());

            // Populate CC_list with the connected components corresponding to sind
            for (const auto& i : sind) {
                if (i >= 0 && i < static_cast<int>(nCC_x.size())) { // Validate index
                    CC_list.emplace_back(nCC_x[i]);
                } else {
                    // Handle invalid indices if necessary
                    std::cerr << "Warning: Index " << i << " is out of bounds for nCC_x with size " << nCC_x.size() << ". Skipping.\n";
                }
            }

            log_message("topological_comp_res: Extracted " + std::to_string(CC_list.size()) + " connected components");
        } catch (const std::exception& e) {
            log_message("ERROR in make_original_dendrogram_cc: " + std::string(e.what()));
            return Eigen::VectorXd::Zero(loc.rows());
        }
        
        // If no components, return empty result
        if (CC_list.empty()) {
            log_message("topological_comp_res: No connected components found, returning empty result");
            return Eigen::VectorXd::Zero(loc.rows());
        }
        
        // Create connected component location matrix
        Eigen::SparseMatrix<double> CC_loc_mat_double;
        try {
            // First get the matrix as double type
            CC_loc_mat_double = extract_connected_loc_mat(CC_list, loc.rows(), "sparse");
            log_message("topological_comp_res: Created CC_loc_mat with " + 
                       std::to_string(CC_loc_mat_double.nonZeros()) + " non-zeros");
        } catch (const std::exception& e) {
            log_message("ERROR in extract_connected_loc_mat: " + std::string(e.what()));
            return Eigen::VectorXd::Zero(loc.rows());
        }
        
        // Filter connected components based on feature expression percentile
        Eigen::SparseMatrix<double> filtered_CC_loc_mat;
        try {
            filtered_CC_loc_mat = filter_connected_loc_exp(CC_loc_mat_double, feat, thres_per);
            log_message("topological_comp_res: Filtered CC_loc_mat, now has " + 
                       std::to_string(filtered_CC_loc_mat.nonZeros()) + " non-zeros");
        } catch (const std::exception& e) {
            log_message("ERROR in filter_connected_loc_exp: " + std::string(e.what()) + ", using unfiltered matrix");
            filtered_CC_loc_mat = CC_loc_mat_double;
        }
        
        // Compute row sums
        Eigen::VectorXd row_sums = filtered_CC_loc_mat * Eigen::VectorXd::Ones(filtered_CC_loc_mat.cols());
        log_message("topological_comp_res: Computed row sums, sum = " + std::to_string(row_sums.sum()));
        
        return row_sums;
    }
    catch (const std::exception& e) {
        log_message("ERROR in topological_comp_res: " + std::string(e.what()));
        return Eigen::VectorXd::Zero(loc.rows());
    }
}