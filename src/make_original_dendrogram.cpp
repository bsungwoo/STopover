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
#include <unordered_map>

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
    std::vector<std::vector<int>> CC;
    Eigen::SparseMatrix<double> E(p, p);
    E.setZero();
    Eigen::MatrixXd duration = Eigen::MatrixXd::Zero(p * threshold.size(), 2);
    std::vector<std::vector<int>> history;

    // Initialize variables
    std::vector<int> ck_cc(p, -1);
    int ncc = -1;

    // Convert sparse matrix to adjacency list
    std::vector<std::vector<int>> edge_list(p);
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            edge_list[it.row()].push_back(it.col());
        }
    }

    for (size_t i = 0; i < threshold.size(); ++i) {
        // Choose current voxels that satisfy threshold interval
        std::vector<int> cvoxels;
        if (i == 0) {
            for (int idx = 0; idx < p; ++idx) {
                if (U(idx) >= threshold[i]) {
                    cvoxels.push_back(idx);
                }
            }
        } else {
            for (int idx = 0; idx < p; ++idx) {
                if (U(idx) >= threshold[i] && U(idx) < threshold[i - 1]) {
                    cvoxels.push_back(idx);
                }
            }
        }

        if (cvoxels.empty()) {
            continue;
        }

        // Map cvoxels to new indices
        std::unordered_map<int, int> voxel_to_index;
        for (size_t idx = 0; idx < cvoxels.size(); ++idx) {
            voxel_to_index[cvoxels[idx]] = idx;
        }

        // Create submatrix A_sub
        Eigen::SparseMatrix<double> A_sub(cvoxels.size(), cvoxels.size());
        std::vector<Eigen::Triplet<double>> tripletList;

        for (size_t idx = 0; idx < cvoxels.size(); ++idx) {
            int voxel = cvoxels[idx];
            for (const auto& neighbor : edge_list[voxel]) {
                auto it = voxel_to_index.find(neighbor);
                if (it != voxel_to_index.end()) {
                    int neighbor_idx = it->second;
                    tripletList.emplace_back(idx, neighbor_idx, 1.0);
                }
            }
        }

        A_sub.setFromTriplets(tripletList.begin(), tripletList.end());

        // Extract connected components
        std::vector<std::set<int>> CC_profiles = connected_components_generator(A_sub);

        // Map back to original voxel indices
        std::vector<std::vector<int>> nCC;
        for (const auto& cc : CC_profiles) {
            std::vector<int> mapped_cc;
            for (const auto& idx : cc) {
                mapped_cc.push_back(cvoxels[idx]);
            }
            nCC.push_back(mapped_cc);
        }

        size_t S = nCC.size();
        std::vector<std::vector<int>> nA_rows(S);
        std::vector<int> neighbor_cc_indices;

        for (size_t j = 0; j < S; ++j) {
            // Find neighbors of current voxels
            std::set<int> neighbor_voxels;
            for (const auto& voxel : nCC[j]) {
                for (const auto& neighbor : edge_list[voxel]) {
                    neighbor_voxels.insert(neighbor);
                }
            }
            // Remove current voxels from neighbors
            for (const auto& voxel : nCC[j]) {
                neighbor_voxels.erase(voxel);
            }

            // Find unique connected component indices of neighbors
            std::unordered_set<int> unique_tcc;
            for (const auto& voxel : neighbor_voxels) {
                int cc_idx = ck_cc[voxel];
                if (cc_idx != -1) {
                    unique_tcc.insert(cc_idx);
                }
            }

            // Record connections to existing connected components
            nA_rows[j].assign(unique_tcc.begin(), unique_tcc.end());
            neighbor_cc_indices.insert(neighbor_cc_indices.end(), unique_tcc.begin(), unique_tcc.end());
        }

        // Remove duplicates from neighbor_cc_indices
        std::sort(neighbor_cc_indices.begin(), neighbor_cc_indices.end());
        neighbor_cc_indices.erase(std::unique(neighbor_cc_indices.begin(), neighbor_cc_indices.end()), neighbor_cc_indices.end());

        if (neighbor_cc_indices.empty()) {
            // No existing neighbors, create new connected components
            for (size_t j = 0; j < S; ++j) {
                ncc += 1;
                CC.push_back(nCC[j]);
                std::sort(CC[ncc].begin(), CC[ncc].end());
                for (const auto& voxel : CC[ncc]) {
                    ck_cc[voxel] = ncc;
                }
                duration(ncc, 0) = threshold[i];
                E.insert(ncc, ncc) = threshold[i];
                history.emplace_back(); // Empty history
            }
        } else {
            // Merge with existing connected components
            // Build adjacency matrix for connected components
            int total_cc = neighbor_cc_indices.size() + S;
            Eigen::SparseMatrix<double> nA(total_cc, total_cc);
            std::vector<Eigen::Triplet<double>> nA_triplets;

            // Set diagonal to 1
            for (int k = 0; k < total_cc; ++k) {
                nA_triplets.emplace_back(k, k, 1.0);
            }

            // Add connections from nA_rows
            for (size_t j = 0; j < S; ++j) {
                int row_idx = neighbor_cc_indices.size() + j;
                for (const auto& col_cc_idx : nA_rows[j]) {
                    auto it = std::find(neighbor_cc_indices.begin(), neighbor_cc_indices.end(), col_cc_idx);
                    if (it != neighbor_cc_indices.end()) {
                        int col_idx = std::distance(neighbor_cc_indices.begin(), it);
                        nA_triplets.emplace_back(row_idx, col_idx, 1.0);
                        nA_triplets.emplace_back(col_idx, row_idx, 1.0);
                    }
                }
            }

            nA.setFromTriplets(nA_triplets.begin(), nA_triplets.end());

            // Extract connected components
            std::vector<std::set<int>> combined_CC_profiles = connected_components_generator(nA);

            for (const auto& cc : combined_CC_profiles) {
                std::vector<int> tind1;
                std::vector<int> tind2;
                for (const auto& idx : cc) {
                    if (idx < static_cast<int>(neighbor_cc_indices.size())) {
                        tind1.push_back(neighbor_cc_indices[idx]);
                    } else {
                        tind2.push_back(idx - neighbor_cc_indices.size());
                    }
                }

                if (tind1.size() == 1) {
                    // Merge into existing connected component
                    int existing_idx = tind1[0];
                    for (const auto& e : tind2) {
                        CC[existing_idx].insert(CC[existing_idx].end(), nCC[e].begin(), nCC[e].end());
                        std::sort(CC[existing_idx].begin(), CC[existing_idx].end());
                        CC[existing_idx].erase(std::unique(CC[existing_idx].begin(), CC[existing_idx].end()), CC[existing_idx].end());
                        for (const auto& voxel : nCC[e]) {
                            ck_cc[voxel] = existing_idx;
                        }
                    }
                } else {
                    // Create a new connected component
                    ncc += 1;
                    std::set<int> new_cc;
                    for (const auto& existing_idx : tind1) {
                        new_cc.insert(CC[existing_idx].begin(), CC[existing_idx].end());
                    }
                    for (const auto& e : tind2) {
                        new_cc.insert(nCC[e].begin(), nCC[e].end());
                    }
                    CC.emplace_back(new_cc.begin(), new_cc.end());
                    std::sort(CC[ncc].begin(), CC[ncc].end());
                    for (const auto& voxel : CC[ncc]) {
                        ck_cc[voxel] = ncc;
                    }
                    duration(ncc, 0) = threshold[i];

                    // Update E matrix
                    for (const auto& existing_idx : tind1) {
                        E.insert(ncc, existing_idx) = threshold[i];
                        E.insert(existing_idx, ncc) = threshold[i];
                    }
                    E.insert(ncc, ncc) = threshold[i];

                    // Update history
                    std::vector<int> hist;
                    for (const auto& existing_idx : tind1) {
                        hist.push_back(existing_idx);
                    }
                    history.push_back(hist);
                }
            }
        }
    }

    // Remove empty lists from the end
    int rev_count = CC.size();
    for (int index = CC.size() - 1; index >= 0; --index) {
        if (CC[index].empty()) {
            rev_count -= 1;
        } else {
            break;
        }
    }

    CC.resize(rev_count);
    history.resize(rev_count);
    E.conservativeResize(rev_count, rev_count);
    duration.conservativeResize(rev_count, 2);

    return std::make_tuple(CC, E, duration, history);
}