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
#include <map>
#include <queue>     // Added missing include for std::queue

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
    
    log_message("extract_connected_comp: Starting with " + std::to_string(threshold_x.size()) + 
               " thresholds, min_size=" + std::to_string(min_size));
    
    // Log adjacency matrix info
    log_message("Adjacency matrix: " + std::to_string(A_sparse.rows()) + "x" + 
               std::to_string(A_sparse.cols()) + " with " + 
               std::to_string(A_sparse.nonZeros()) + " non-zeros");
    
    // Check if adjacency matrix is empty
    if (A_sparse.nonZeros() == 0) {
        log_message("WARNING: Adjacency matrix has no non-zero entries!");
    }
    
    std::vector<std::vector<int>> CC_list;
    
    if (threshold_x.empty()) {
        log_message("extract_connected_comp: Empty threshold list, returning empty CC_list");
        return CC_list;
    }
    
    // Log threshold values
    std::string thresholds_str = "Thresholds: ";
    for (size_t i = 0; i < std::min(threshold_x.size(), size_t(10)); ++i) {
        thresholds_str += std::to_string(threshold_x[i]) + " ";
    }
    if (threshold_x.size() > 10) {
        thresholds_str += "... (" + std::to_string(threshold_x.size()) + " total)";
    }
    log_message(thresholds_str);
    
    // For each threshold value
    for (size_t i = 0; i < threshold_x.size(); ++i) {
        double threshold = threshold_x[i];
        
        // Find spots that meet the threshold
        std::vector<int> selected_spots;
        for (int j = 0; j < tx.size(); ++j) {
            if (tx(j) >= threshold) {
                selected_spots.push_back(j);
            }
        }
        
        if (selected_spots.empty()) {
            log_message("extract_connected_comp: No spots meet threshold " + std::to_string(threshold));
            continue;
        }
        
        log_message("extract_connected_comp: Found " + std::to_string(selected_spots.size()) + 
                   " spots meeting threshold " + std::to_string(threshold));
        
        // Create adjacency list for selected spots
        std::vector<std::vector<int>> adj_list(selected_spots.size());
        int total_edges = 0;
        
        // Create mapping from original indices to compressed indices
        std::map<int, int> index_map;
        for (size_t j = 0; j < selected_spots.size(); ++j) {
            index_map[selected_spots[j]] = j;
        }
        
        // Populate adjacency list
        for (size_t j = 0; j < selected_spots.size(); ++j) {
            int orig_j = selected_spots[j];
            for (Eigen::SparseMatrix<double>::InnerIterator it(A_sparse, orig_j); it; ++it) {
                int orig_k = it.row();
                if (it.value() > 0 && index_map.find(orig_k) != index_map.end()) {
                    adj_list[j].push_back(index_map[orig_k]);
                    total_edges++;
                }
            }
        }
        
        log_message("extract_connected_comp: Created adjacency list with " + 
                   std::to_string(total_edges) + " edges for threshold " + 
                   std::to_string(threshold));
        
        if (total_edges == 0) {
            log_message("WARNING: No edges in adjacency list for threshold " + 
                       std::to_string(threshold) + ", skipping component extraction");
            continue;
        }
        
        // Find connected components using BFS
        std::vector<bool> visited(selected_spots.size(), false);
        int components_found = 0;
        
        for (size_t j = 0; j < selected_spots.size(); ++j) {
            if (!visited[j]) {
                std::vector<int> component;
                std::queue<int> queue;
                queue.push(j);
                
                while (!queue.empty()) {
                    int current = queue.front();
                    queue.pop();
                    
                    if (!visited[current]) {
                        visited[current] = true;
                        component.push_back(selected_spots[current]);
                        
                        // Add neighbors to queue
                        for (int neighbor : adj_list[current]) {
                            if (!visited[neighbor]) {
                                queue.push(neighbor);
                            }
                        }
                    }
                }
                
                // Add component if it meets minimum size
                if (static_cast<int>(component.size()) >= min_size) {
                    CC_list.push_back(component);
                    components_found++;
                    log_message("extract_connected_comp: Found component with " + 
                               std::to_string(component.size()) + " spots");
                }
            }
        }
        
        log_message("extract_connected_comp: Found " + std::to_string(components_found) + 
                   " components for threshold " + std::to_string(threshold));
    }
    
    log_message("extract_connected_comp: Returning " + std::to_string(CC_list.size()) + " components");
    return CC_list;
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

// Function to extract connected components
std::vector<std::vector<int>> extract_connected_comp_python_style(
    const Eigen::VectorXd& tx, 
    const Eigen::SparseMatrix<double>& A_sparse, 
    const std::vector<double>& threshold_x, 
    int num_spots, 
    int min_size) {

    try {
        log_message("extract_connected_comp_python_style: Starting with " + std::to_string(threshold_x.size()) + 
                   " thresholds, min_size=" + std::to_string(min_size));
        
        // Log adjacency matrix info
        log_message("Adjacency matrix: " + std::to_string(A_sparse.rows()) + "x" + 
                   std::to_string(A_sparse.cols()) + " with " + 
                   std::to_string(A_sparse.nonZeros()) + " non-zeros");
        
        // Check if adjacency matrix is empty
        if (A_sparse.nonZeros() == 0) {
            log_message("WARNING: Adjacency matrix has no non-zero entries!");
            return {};
        }
        
        if (threshold_x.empty()) {
            log_message("WARNING: Empty threshold list, returning empty CC_list");
            return {};
        }
        
        // Direct implementation of connected components using BFS
        std::vector<std::vector<int>> CC_list;
        std::vector<bool> visited(num_spots, false);
        
        // For each threshold value (in descending order)
        for (double threshold : threshold_x) {
            log_message("Processing threshold: " + std::to_string(threshold));
            
            // Create a mask for spots above the threshold
            std::vector<bool> above_threshold(num_spots, false);
            for (int i = 0; i < tx.size(); ++i) {
                if (tx(i) >= threshold) {
                    above_threshold[i] = true;
                }
            }
            
            // Find connected components using BFS
            for (int start = 0; start < num_spots; ++start) {
                if (above_threshold[start] && !visited[start]) {
                    // Start a new component
                    std::vector<int> component;
                    std::queue<int> queue;
                    
                    queue.push(start);
                    visited[start] = true;
                    
                    while (!queue.empty()) {
                        int current = queue.front();
                        queue.pop();
                        component.push_back(current);
                        
                        // Check neighbors using adjacency matrix
                        for (Eigen::SparseMatrix<double>::InnerIterator it(A_sparse, current); it; ++it) {
                            int neighbor = it.row();
                            if (above_threshold[neighbor] && !visited[neighbor]) {
                                queue.push(neighbor);
                                visited[neighbor] = true;
                            }
                        }
                    }
                    
                    // Add component if it meets minimum size
                    if (static_cast<int>(component.size()) >= min_size) {
                        CC_list.push_back(component);
                        log_message("Found component with " + std::to_string(component.size()) + " spots");
                    }
                }
            }
        }
        
        log_message("extract_connected_comp_python_style: Found " + std::to_string(CC_list.size()) + " components");
        return CC_list;
        
    } catch (const std::exception& e) {
        log_message("ERROR in extract_connected_comp_python_style: " + std::string(e.what()));
        return {};
    }
}

// Update the function signature to match what parallelize.cpp expects
std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>> topological_comp_res(
    const Eigen::MatrixXd& loc, 
    const Eigen::SparseMatrix<double>& A, 
    const Eigen::MatrixXd& mask,
    const std::string& spatial_type, 
    int min_size, 
    int thres_per, 
    const std::string& return_mode) {

    try {
        // Define p at the beginning of the function
        int p = loc.rows();
        
        if (return_mode != "all" && return_mode != "cc_loc" && return_mode != "jaccard_cc_list") {
            throw std::invalid_argument("'return_mode' should be among 'all', 'cc_loc', or 'jaccard_cc_list'");
        }

        log_message("topological_comp_res: Starting with " + std::to_string(p) + " spots");
        
        // Extract feature vector from the first column of loc
        Eigen::VectorXd feat = loc.col(0);
        
        // Smooth the feature values with given mask
        Eigen::VectorXd smooth;
        try {
            smooth = smooth_feature_vector_python_style(feat, mask);
            
            // Log smoothed feature stats
            double min_smooth = smooth.minCoeff();
            double max_smooth = smooth.maxCoeff();
            double mean_smooth = smooth.mean();
            int positive_count = (smooth.array() > 0).count();
            
            log_message("Smoothed feature stats: min=" + std::to_string(min_smooth) + 
                       ", max=" + std::to_string(max_smooth) + 
                       ", mean=" + std::to_string(mean_smooth) + 
                       ", positive values=" + std::to_string(positive_count) + 
                       " out of " + std::to_string(smooth.size()));
        } catch (const std::exception& e) {
            log_message("Error during smoothing: " + std::string(e.what()));
            throw;
        }
        
        // Create thresholds from positive values in smoothed features
        Eigen::VectorXd t = smooth.cwiseMax(0);
                
        // Convert to vector and sort in descending order
        std::vector<double> threshold_values;
        threshold_values.reserve(t.size());
        for (int i = 0; i < t.size(); i++) {
            if (t(i) > 0) {  // Only include positive values
                threshold_values.push_back(t(i));
            }
        }
        std::sort(threshold_values.begin(), threshold_values.end(), std::greater<double>());
        
        log_message("Created " + std::to_string(threshold_values.size()) + " threshold values");
        
        // Extract connected components
        std::vector<std::vector<int>> CC_list = extract_connected_comp_python_style(t, A, threshold_values, p, min_size);
        
        log_message("Found " + std::to_string(CC_list.size()) + " connected components");
        
        // Create connected location matrix
        Eigen::SparseMatrix<int> CC_loc_mat(p, CC_list.size());
        std::vector<Eigen::Triplet<int>> triplets;
        
        for (size_t j = 0; j < CC_list.size(); ++j) {
            const auto& component = CC_list[j];
            for (int idx : component) {
                if (idx >= 0 && idx < p) {  // Safety check
                    triplets.emplace_back(idx, j, j + 1);  // 1-indexed component IDs
                }
            }
        }
        
        CC_loc_mat.setFromTriplets(triplets.begin(), triplets.end());
        CC_loc_mat.makeCompressed();
        
        log_message("Created CC_loc_mat with shape (" + 
                   std::to_string(CC_loc_mat.rows()) + ", " + 
                   std::to_string(CC_loc_mat.cols()) + ") and " + 
                   std::to_string(CC_loc_mat.nonZeros()) + " non-zeros");
        
        // If thres_per is greater than 0, filter components based on expression percentile
        if (thres_per > 0) {
            // Compute mean expression for each component
            std::vector<std::pair<int, double>> component_means;
            for (size_t i = 0; i < CC_list.size(); ++i) {
                const auto& component = CC_list[i];
                double sum = 0.0;
                for (int idx : component) {
                    sum += feat(idx);
                }
                double mean = sum / component.size();
                component_means.emplace_back(i, mean);
            }
            
            // Sort components by mean expression in descending order
            std::sort(component_means.begin(), component_means.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            // Keep top thres_per% of components
            int keep_count = std::max(1, static_cast<int>(CC_list.size() * thres_per / 100.0));
            keep_count = std::min(keep_count, static_cast<int>(CC_list.size()));
            
            log_message("Keeping top " + std::to_string(keep_count) + " components out of " + 
                       std::to_string(CC_list.size()) + 
                       " based on thres_per=" + std::to_string(thres_per));
            
            // Get indices of components to keep
            std::vector<int> keep_indices;
            for (int i = 0; i < keep_count; ++i) {
                keep_indices.push_back(component_means[i].first);
            }
            
            // Create filtered CC_list
            std::vector<std::vector<int>> filtered_CC_list;
            filtered_CC_list.reserve(keep_count);
            
            for (int idx : keep_indices) {
                filtered_CC_list.push_back(CC_list[idx]);
            }
            
            // Create filtered CC_loc_mat
            Eigen::SparseMatrix<int> filtered_CC_loc_mat(p, keep_count);
            std::vector<Eigen::Triplet<int>> filtered_triplets;
            
            for (size_t j = 0; j < keep_indices.size(); ++j) {
                int orig_j = keep_indices[j];
                const auto& component = CC_list[orig_j];
                for (int idx : component) {
                    if (idx >= 0 && idx < p) {  // Safety check
                        filtered_triplets.emplace_back(idx, j, j + 1);  // 1-indexed component IDs
                    }
                }
            }
            
            filtered_CC_loc_mat.setFromTriplets(filtered_triplets.begin(), filtered_triplets.end());
            filtered_CC_loc_mat.makeCompressed();
            
            log_message("Created filtered CC_loc_mat with shape (" + 
                       std::to_string(filtered_CC_loc_mat.rows()) + ", " + 
                       std::to_string(filtered_CC_loc_mat.cols()) + ") and " + 
                       std::to_string(filtered_CC_loc_mat.nonZeros()) + " non-zeros");
            
            return std::make_tuple(filtered_CC_list, filtered_CC_loc_mat);
        }
        
        return std::make_tuple(CC_list, CC_loc_mat);
    }
    catch (const std::exception& e) {
        log_message("ERROR in topological_comp_res: " + std::string(e.what()));
        std::vector<std::vector<int>> empty_CC_list;
        Eigen::SparseMatrix<int> empty_CC_loc_mat(loc.rows(), 0);
        return std::make_tuple(empty_CC_list, empty_CC_loc_mat);
    }
}