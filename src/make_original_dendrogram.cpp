#include "make_original_dendrogram.h"

// Function to extract connected nodes using breadth-first search (BFS)
std::set<int> extract_connected_nodes(const std::vector<std::vector<int>>& edge_list, int sel_node_idx) {
    std::set<int> cc_set;
    std::queue<int> next_neighbor;
    next_neighbor.push(sel_node_idx);
    
    while (!next_neighbor.empty()) {
        int curr_neighbor = next_neighbor.front();
        next_neighbor.pop();
        
        if (cc_set.find(curr_neighbor) == cc_set.end()) {
            cc_set.insert(curr_neighbor);
            for (const int& neighbor : edge_list[curr_neighbor]) {
                next_neighbor.push(neighbor);
            }
        }
    }
    
    return cc_set;
}

// Function to generate connected components from a sparse adjacency matrix
std::vector<std::set<int>> connected_components_generator(const Eigen::SparseMatrix<double>& A) {
    log_message("connected_components_generator: Starting with matrix size=" + 
               std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
    
    try {
        std::vector<std::set<int>> components;
        std::vector<bool> visited(A.rows(), false);
        
        for (int i = 0; i < A.rows(); ++i) {
            if (!visited[i]) {
                std::set<int> component;
                std::queue<int> queue;
                
                queue.push(i);
                visited[i] = true;
                
                while (!queue.empty()) {
                    int node = queue.front();
                    queue.pop();
                    component.insert(node);
                    
                    // Process neighbors
                    for (Eigen::SparseMatrix<double>::InnerIterator it(A, node); it; ++it) {
                        int neighbor = it.col();
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            queue.push(neighbor);
                        }
                    }
                }
                
                components.push_back(component);
            }
        }
        
        log_message("connected_components_generator: Found " + std::to_string(components.size()) + " components");
        return components;
    }
    catch (const std::exception& e) {
        log_message("ERROR in connected_components_generator: " + std::string(e.what()));
        throw; // Re-throw to be caught by the caller
    }
}

// Function to create the original dendrogram with connected components
std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<double>, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_original_dendrogram_cc(const Eigen::VectorXd& U, const Eigen::SparseMatrix<double>& A, const std::vector<double>& threshold) {
    log_message("make_original_dendrogram_cc: Starting with U size=" + std::to_string(U.size()) + 
               ", A size=" + std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + 
               ", threshold size=" + std::to_string(threshold.size()));
    
    try {
        // Initialize result containers
        std::vector<std::vector<int>> CC;
        Eigen::SparseMatrix<double> E(U.size(), U.size());
        Eigen::MatrixXd duration = Eigen::MatrixXd::Zero(U.size(), 2);
        std::vector<std::vector<int>> history(U.size());
        
        // Process each threshold
        for (size_t i = 0; i < threshold.size(); ++i) {
            log_message("Processing threshold " + std::to_string(i+1) + "/" + 
                       std::to_string(threshold.size()) + ": " + std::to_string(threshold[i]));
            
            // Create binary vector for spots above threshold
            std::vector<int> binary_vec(U.size(), 0);
            for (int j = 0; j < U.size(); ++j) {
                if (U(j) >= threshold[i]) {
                    binary_vec[j] = 1;
                }
            }
            
            // Create adjacency matrix for spots above threshold
            typedef Eigen::Triplet<double> T;
            std::vector<T> tripletList;
            
            log_message("Creating thresholded adjacency matrix...");
            for (int k = 0; k < A.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
                    int row = it.row();
                    int col = it.col();
                    
                    if (binary_vec[row] == 1 && binary_vec[col] == 1) {
                        tripletList.push_back(T(row, col, 1.0));
                    }
                }
            }
            
            // Create sparse matrix for connected components
            Eigen::SparseMatrix<double> A_cc(U.size(), U.size());
            A_cc.setFromTriplets(tripletList.begin(), tripletList.end());
            
            log_message("Finding connected components...");
            // Find connected components
            std::vector<std::set<int>> CC_profiles = connected_components_generator(A_cc);
            log_message("Found " + std::to_string(CC_profiles.size()) + " connected components");
            
            // Process connected components
            for (const auto& component : CC_profiles) {
                std::vector<int> current_cc(component.begin(), component.end());
                CC.push_back(current_cc);
                E.coeffRef(current_cc[0], current_cc[0]) = threshold[i];
                duration(current_cc[0], 0) = threshold[i];
                history[current_cc[0]] = {};  // Populate the history
            }
        }
        
        log_message("make_original_dendrogram_cc completed successfully");
        return std::make_tuple(CC, E, duration, history);
    }
    catch (const std::exception& e) {
        log_message("ERROR in make_original_dendrogram_cc: " + std::string(e.what()));
        throw; // Re-throw to be caught by the caller
    }
}