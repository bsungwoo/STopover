#include "make_original_dendrogram.h"
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <set>
#include <vector>
#include <tuple>
#include <numeric> // For std::accumulate
#include <iterator> // For std::set_difference

/**
 * @brief Extracts connected nodes using breadth-first search (BFS).
 *
 * @param edge_list Adjacency list representation of the graph.
 * @param sel_node_idx Index of the selected node to start BFS.
 * @return A set of connected node indices.
 */
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

/**
 * @brief Generates connected components from a sparse adjacency matrix.
 *
 * @param A Sparse adjacency matrix (Eigen::SparseMatrix<double>).
 * @return A vector of sets, each representing a connected component.
 */
std::vector<std::set<int>> connected_components_generator(const Eigen::SparseMatrix<double>& A) {
    std::vector<std::set<int>> components;
    std::vector<bool> visited(A.rows(), false);

    // Convert sparse matrix to adjacency list for efficient traversal
    std::vector<std::vector<int>> edge_list(A.rows(), std::vector<int>());
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            edge_list[k].push_back(it.col());
        }
    }

    for (int vertex = 0; vertex < A.rows(); ++vertex) {
        if (!visited[vertex]) {
            std::set<int> cc_set;
            std::queue<int> queue;
            queue.push(vertex);

            while (!queue.empty()) {
                int curr = queue.front();
                queue.pop();
                if (!visited[curr]) {
                    visited[curr] = true;
                    cc_set.insert(curr);
                    for (const int& neighbor : edge_list[curr]) {
                        if (!visited[neighbor]) {
                            queue.push(neighbor);
                        }
                    }
                }
            }
            components.push_back(cc_set);
        }
    }
    return components;
}

/**
 * @brief Creates the original dendrogram with connected components.
 *
 * @param U Eigen::VectorXd containing some data (e.g., expression values).
 * @param A Sparse adjacency matrix representing connections (Eigen::SparseMatrix<double>).
 * @param threshold Vector of thresholds to define layers.
 * @return A tuple containing:
 *         - CC: Vector of connected components (each component is a vector of integers).
 *         - E: Sparse adjacency matrix (Eigen::SparseMatrix<double>).
 *         - duration: Duration matrix (Eigen::MatrixXd).
 *         - history: Vector of connected components history (each component is a vector of integers).
 */
std::tuple<std::vector<std::vector<int>>, 
           Eigen::SparseMatrix<double>, 
           Eigen::MatrixXd, 
           std::vector<std::vector<int>>>
make_original_dendrogram_cc(const Eigen::VectorXd& U, 
                            const Eigen::SparseMatrix<double>& A, 
                            const std::vector<double>& threshold) {
    int p = U.size();
    std::vector<std::vector<int>> CC(p, std::vector<int>());
    Eigen::SparseMatrix<double> E(p, p);
    E.setZero();
    Eigen::MatrixXd duration = Eigen::MatrixXd::Zero(p, 2);
    std::vector<std::vector<int>> history(p, std::vector<int>());

    // Initialize variables
    std::vector<int> ck_cc(p, -1);
    int ncc = -1;

    // Precompute adjacency list from A
    std::vector<std::vector<int>> edge_list(A.rows(), std::vector<int>());
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            edge_list[k].push_back(it.col());
        }
    }

    for (size_t i = 0; i < threshold.size(); ++i) {
        // Choose current voxels that satisfy threshold interval
        std::vector<int> cvoxels;
        if (i == 0) {
            for(int idx = 0; idx < p; ++idx){
                if(U(idx) >= threshold[i]){
                    cvoxels.push_back(idx);
                }
            }
        }
        else {
            for(int idx = 0; idx < p; ++idx){
                if(U(idx) >= threshold[i] && U(idx) < threshold[i-1]){
                    cvoxels.push_back(idx);
                }
            }
        }

        // Create submatrix for current cvoxels
        // Since A is sparse and binary, we can create a submatrix where rows and cols are cvoxels
        std::vector<Eigen::Triplet<int>> tripletList;
        for(auto & voxel : cvoxels){
            for(auto & neighbor : edge_list[voxel]){
                if(std::find(cvoxels.begin(), cvoxels.end(), neighbor) != cvoxels.end()){
                    tripletList.emplace_back(voxel, neighbor, 1);
                }
            }
        }
        Eigen::SparseMatrix<double> A_sub(p, p);
        A_sub.setFromTriplets(tripletList.begin(), tripletList.end());

        // Extract connected components for the adjacency matrix
        std::vector<std::set<int>> CC_profiles = connected_components_generator(A_sub);

        // Filter CC_profiles to include only those with voxels in cvoxels
        std::vector<std::set<int>> filtered_CC_profiles;
        for(const auto& cc : CC_profiles){
            std::set<int> intersection;
            std::set_intersection(cc.begin(), cc.end(),
                                  cvoxels.begin(), cvoxels.end(),
                                  std::inserter(intersection, intersection.begin()));
            if(!intersection.empty()){
                filtered_CC_profiles.push_back(intersection);
            }
        }

        size_t S = filtered_CC_profiles.size();

        // Initialize new connected components
        std::vector<std::vector<int>> nCC(S, std::vector<int>());
        Eigen::SparseMatrix<double> nA;
        // Note: The Python code uses a DOK matrix, which can be emulated using Eigen's SparseMatrix with triplets
        std::vector<Eigen::Triplet<int>> nA_triplets;
        nA.resize(p, S);
        nA.reserve(Eigen::VectorXi::Constant(S, 10)); // Estimate non-zeros per column

        std::vector<int> neighbor_cc_indices;

        for(size_t j = 0; j < S; ++j){
            nCC[j].assign(filtered_CC_profiles[j].begin(), filtered_CC_profiles[j].end());
            // Find neighbors of current voxels
            std::set<int> neighbor_voxels;
            for(const auto & voxel : nCC[j]){
                for(const auto & neighbor : edge_list[voxel]){
                    neighbor_voxels.insert(neighbor);
                }
            }
            // Remove current voxels from neighbors
            for(const auto & voxel : nCC[j]){
                neighbor_voxels.erase(voxel);
            }
            // Find unique connected component indices of neighbors
            std::unordered_set<int> unique_tcc;
            for(const auto & voxel : neighbor_voxels){
                if(ck_cc[voxel] != -1){
                    unique_tcc.insert(ck_cc[voxel]);
                }
            }
            for(const auto & tcc : unique_tcc){
                nA_triplets.emplace_back(tcc, j, 1);
                neighbor_cc_indices.push_back(tcc);
            }
        }
        nA.setFromTriplets(nA_triplets.begin(), nA_triplets.end());

        // Remove duplicates from neighbor_cc_indices
        std::sort(neighbor_cc_indices.begin(), neighbor_cc_indices.end());
        neighbor_cc_indices.erase(std::unique(neighbor_cc_indices.begin(), neighbor_cc_indices.end()), neighbor_cc_indices.end());

        if(neighbor_cc_indices.empty()){
            // No existing neighbors, create new connected components
            for(size_t j = 0; j < S; ++j){
                ncc += 1;
                CC[ncc] = nCC[j];
                std::sort(CC[ncc].begin(), CC[ncc].end());
                for(const auto & voxel : CC[ncc]){
                    ck_cc[voxel] = ncc;
                }
                duration(ncc, 0) = threshold[i];
                E.coeffRef(ncc, ncc) = threshold[i];
                history[ncc] = {}; // No history
            }
        }
        else{
            // Create a temporary adjacency matrix including existing neighbors and new components
            int existing_cc = neighbor_cc_indices.size();
            int total_cc = existing_cc + S;
            Eigen::SparseMatrix<double> nA_tmp(total_cc, total_cc);
            std::vector<Eigen::Triplet<int>> nA_tmp_triplets;
            // Set diagonal to 1
            for(int k = 0; k < total_cc; ++k){
                nA_tmp_triplets.emplace_back(k, k, 1);
            }
            // Add connections from nA
            for(int k = 0; k < nA.outerSize(); ++k){
                for(Eigen::SparseMatrix<double>::InnerIterator it(nA, k); it; ++it){
                    nA_tmp_triplets.emplace_back(it.row(), it.col() + existing_cc, it.value());
                }
            }
            nA_tmp.setFromTriplets(nA_tmp_triplets.begin(), nA_tmp_triplets.end());

            // Estimate connected components of clusters
            std::vector<std::set<int>> combined_CC_profiles = connected_components_generator(nA_tmp);
            size_t combined_S = combined_CC_profiles.size();

            for(size_t j = 0; j < combined_S; ++j){
                std::set<int> tind = combined_CC_profiles[j];
                std::vector<int> tind_vector(tind.begin(), tind.end());

                // Split tind into tind1 and tind2 based on existing_cc
                std::vector<int> tind1;
                std::vector<int> tind2;
                for(const auto & idx : tind_vector){
                    if(idx < existing_cc){
                        tind1.push_back(neighbor_cc_indices[idx]);
                    }
                    else{
                        tind2.push_back(idx - existing_cc);
                    }
                }

                if(tind1.size() == 1){
                    // Merge into existing connected component
                    int existing_index = tind1[0];
                    for(const auto & e : tind2){
                        CC[existing_index].insert(CC[existing_index].end(), nCC[e].begin(), nCC[e].end());
                        std::sort(CC[existing_index].begin(), CC[existing_index].end());
                        CC[existing_index].erase(std::unique(CC[existing_index].begin(), CC[existing_index].end()), CC[existing_index].end());
                        for(const auto & voxel : nCC[e]){
                            ck_cc[voxel] = existing_index;
                        }
                    }
                }
                else{
                    // Create a new connected component
                    ncc += 1;
                    std::set<int> new_cc;
                    for(const auto & existing_index : tind1){
                        new_cc.insert(CC[existing_index].begin(), CC[existing_index].end());
                    }
                    for(const auto & e : tind2){
                        new_cc.insert(nCC[e].begin(), nCC[e].end());
                    }
                    CC[ncc] = std::vector<int>(new_cc.begin(), new_cc.end());
                    std::sort(CC[ncc].begin(), CC[ncc].end());
                    for(const auto & voxel : CC[ncc]){
                        ck_cc[voxel] = ncc;
                    }
                    duration(ncc, 0) = threshold[i];

                    // Update E matrix
                    for(const auto & existing_index : tind1){
                        // Assuming E is being updated similarly to Python's behavior
                        // This part may need more detailed implementation based on specific requirements
                        E.coeffRef(ncc, existing_index) = threshold[i];
                        E.coeffRef(existing_index, ncc) = threshold[i];
                    }
                    E.coeffRef(ncc, ncc) = threshold[i];

                    // Update history
                    for(const auto & existing_index : tind1){
                        history[ncc].push_back(existing_index);
                    }
                }
            }
        }
    }

    // Remove empty lists from the end
    int rev_count = p;
    for(int index = p-1; index >=0; --index){
        if(CC[index].empty()){
            rev_count -=1;
        }
        else{
            break;
        }
    }

    CC.resize(rev_count);
    history.resize(rev_count);
    E.conservativeResize(rev_count, rev_count);
    duration.conservativeResize(rev_count, 2);

    return std::make_tuple(CC, E, duration, history);
}