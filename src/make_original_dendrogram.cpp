#include "make_original_dendrogram.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// =================== Helper Functions ===================

// Helper function to extract connected nodes using BFS with ordered traversal
std::vector<int> extract_connected_nodes(const std::vector<std::vector<int>>& edge_list, int sel_node_idx) {
    std::vector<int> cc_list; // To store the connected component nodes in order
    std::queue<int> to_visit;
    std::vector<bool> visited(edge_list.size(), false);

    to_visit.push(sel_node_idx);
    visited[sel_node_idx] = true;

    while (!to_visit.empty()) {
        int vertex = to_visit.front();
        to_visit.pop();
        cc_list.push_back(vertex);

        const auto& neighbors = edge_list[vertex];
        for (int neighbor : neighbors) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                to_visit.push(neighbor);
            }
        }
    }
    return cc_list;
}

// Function to find connected components in a graph represented by an edge list
std::vector<std::vector<int>> connected_components(const std::vector<std::vector<int>>& edge_list) {
    size_t n = edge_list.size();
    std::vector<bool> visited(n, false);
    std::vector<std::vector<int>> connected_components_list;

    for (size_t vertex = 0; vertex < n; ++vertex) {
        if (!visited[vertex]) {
            std::vector<int> cc_list = extract_connected_nodes(edge_list, vertex);
            // Mark nodes as visited
            for (int idx : cc_list) {
                visited[idx] = true;
            }
            connected_components_list.push_back(cc_list);
        }
    }
    return connected_components_list;
}

// =================== Main Function ===================

// Main function to compute original dendrogram with connected components
std::tuple<
    std::vector<std::vector<int>>,
    Eigen::SparseMatrix<double, Eigen::RowMajor>,
    Eigen::MatrixXd,
    std::vector<std::vector<int>>
>
make_original_dendrogram_cc(
    const Eigen::VectorXd& U,
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
    const std::vector<double>& threshold
)
{
    int p = U.size();
    std::vector<std::vector<int>> CC(p);  // Initialize CC with size p
    Eigen::SparseMatrix<double, Eigen::RowMajor> E(p, p);  // Initialize E with size p x p
    int ncc = -1;
    Eigen::VectorXi ck_cc = Eigen::VectorXi::Constant(p, -1);
    Eigen::MatrixXd duration = Eigen::MatrixXd::Zero(p, 2);  // Initialize duration with size p x 2
    std::vector<std::vector<int>> history(p);  // Initialize history with size p

    for (size_t i = 0; i < threshold.size(); ++i) {
        // Choose current voxels that satisfy threshold interval
        std::vector<int> cvoxels;
        if (i == 0) {
            for (int idx = 0; idx < U.size(); ++idx) {
                if (U[idx] >= threshold[i]) {
                    cvoxels.push_back(idx);
                }
            }
        } else {
            for (int idx = 0; idx < U.size(); ++idx) {
                if (U[idx] >= threshold[i] && U[idx] < threshold[i - 1]) {
                    cvoxels.push_back(idx);
                }
            }
        }

        // Create a mapping from original indices to new indices
        std::unordered_map<int, int> voxel_to_index;
        for (size_t idx = 0; idx < cvoxels.size(); ++idx) {
            voxel_to_index[cvoxels[idx]] = idx;
        }

        // Build sub adjacency matrix as edge list
        size_t n_cvoxels = cvoxels.size();
        std::vector<std::vector<int>> edge_list(n_cvoxels);

        for (size_t idx = 0; idx < cvoxels.size(); ++idx) {
            int original_idx = cvoxels[idx];
            std::vector<int> neighbors;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, original_idx); it; ++it) {
                int col = it.col();
                if (voxel_to_index.find(col) != voxel_to_index.end()) {
                    neighbors.push_back(voxel_to_index[col]);
                }
            }
            std::sort(neighbors.begin(), neighbors.end()); // Sort the neighbor list
            edge_list[idx] = neighbors;
        }

        // Extract connected components
        std::vector<std::vector<int>> CC_profiles = connected_components(edge_list);
        int S = CC_profiles.size();
        std::vector<std::vector<int>> nCC(S);
        Eigen::SparseMatrix<double, Eigen::RowMajor> nA(p, S);
        std::vector<int> neighbor_cc;

        for (int j = 0; j < S; ++j) {
            // Map back to original indices
            for (int idx : CC_profiles[j]) {
                nCC[j].push_back(cvoxels[idx]);
            }

            // Find neighbors
            std::unordered_set<int> neighbor_voxels_set;
            for (int idx : nCC[j]) {
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, idx); it; ++it) {
                    neighbor_voxels_set.insert(it.col());
                }
            }

            // Determine which connected components the neighbors belong to
            std::unordered_set<int> tcc_set;
            for (int neighbor : neighbor_voxels_set) {
                if (ck_cc[neighbor] != -1) {
                    tcc_set.insert(ck_cc[neighbor]);
                }
            }

            // Update nA and neighbor_cc
            for (int cc_idx : tcc_set) {
                nA.insert(cc_idx, j) = 1;
                neighbor_cc.push_back(cc_idx);
            }
        }

        // Remove duplicates from neighbor_cc
        std::sort(neighbor_cc.begin(), neighbor_cc.end());
        neighbor_cc.erase(std::unique(neighbor_cc.begin(), neighbor_cc.end()), neighbor_cc.end());

        if (neighbor_cc.empty()) {
            // Initialize new connected components
            for (int j = 0; j < S; ++j) {
                ncc += 1;
                // Update ck_cc
                for (int idx : nCC[j]) {
                    ck_cc[idx] = ncc;
                }
                CC[ncc] = nCC[j];

                // Update duration
                duration(ncc, 0) = threshold[i];

                // Update E
                E.insert(ncc, ncc) = threshold[i];

                // Update history
                history[ncc] = std::vector<int>();
            }
        } else {
            // Construct nA_tmp with self-loops
            int N = neighbor_cc.size() + S;
            std::vector<std::vector<int>> edge_list_tmp(N);

            // Add self-loops
            for (int x = 0; x < N; ++x) {
                edge_list_tmp[x].push_back(x);
            }

            // Add edges between neighbor_cc and new components
            for (size_t row = 0; row < neighbor_cc.size(); ++row) {
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(nA, neighbor_cc[row]); it; ++it) {
                    int col = it.col();
                    if (col >= S) continue; // Prevent out-of-bounds
                    int col_idx = neighbor_cc.size() + col;
                    edge_list_tmp[row].push_back(col_idx);
                    edge_list_tmp[col_idx].push_back(row);
                }
            }

            // Extract connected components from nA_tmp
            std::vector<std::vector<int>> CC_profiles_tmp = connected_components(edge_list_tmp);
            int S_tmp = CC_profiles_tmp.size();

            for (int j = 0; j < S_tmp; ++j) {
                std::vector<int> tind = CC_profiles_tmp[j];
                std::vector<int> tind1, tind2;
                for (int idx : tind) {
                    if (idx < neighbor_cc.size()) {
                        tind1.push_back(neighbor_cc[idx]);
                    } else {
                        tind2.push_back(idx - neighbor_cc.size());
                    }
                }

                // Sort tind1 and tind2 to ensure consistent order
                std::sort(tind1.begin(), tind1.end());
                std::sort(tind2.begin(), tind2.end());

                if (tind1.size() == 1) {
                    int cc_idx = tind1[0];
                    // Merge nCC[tind2] into CC[cc_idx]
                    for (int idx : tind2) {
                        CC[cc_idx].insert(CC[cc_idx].end(), nCC[idx].begin(), nCC[idx].end());
                    }
                    // Remove duplicates
                    std::sort(CC[cc_idx].begin(), CC[cc_idx].end());
                    CC[cc_idx].erase(std::unique(CC[cc_idx].begin(), CC[cc_idx].end()), CC[cc_idx].end());

                    // Update ck_cc
                    for (int idx : CC[cc_idx]) {
                        ck_cc[idx] = cc_idx;
                    }
                } else {
                    ncc += 1;
                    // Create a new connected component
                    std::vector<int> new_CC;
                    for (int idx : tind1) {
                        new_CC.insert(new_CC.end(), CC[idx].begin(), CC[idx].end());
                    }
                    for (int idx : tind2) {
                        new_CC.insert(new_CC.end(), nCC[idx].begin(), nCC[idx].end());
                    }

                    // Remove duplicates
                    std::sort(new_CC.begin(), new_CC.end());
                    new_CC.erase(std::unique(new_CC.begin(), new_CC.end()), new_CC.end());

                    // Update ck_cc
                    for (int idx : new_CC) {
                        ck_cc[idx] = ncc;
                    }

                    // Add new connected component
                    CC[ncc] = new_CC;

                    // Update duration
                    duration(ncc, 0) = threshold[i];
                    for (int idx : tind1) {
                        if (idx >= 0 && idx < p) { // Safety check
                            duration(idx, 1) = threshold[i];
                        }
                    }

                    for (size_t ind = 0; ind < tind1.size(); ++ind) {
                        int e = tind1[ind];
                        for (size_t j = 0; j < tind1.size(); ++j) {
                            int j_idx = tind1[j];
                            if (j_idx != e) {
                                // Set E[j, e] to threshold[i]
                                E.coeffRef(j_idx, e) = threshold[i];
                                // Ensure symmetry by setting E[e, j] as well
                                E.coeffRef(e, j_idx) = threshold[i];
                            }
                            // If j_idx == e, retain the original E[j_idx, e]
                            // No action needed since we don't modify E[e, e]
                        }
                    }

                    // Update E
                    // Set E[new_cc, new_cc] = threshold[i]
                    E.insert(ncc, ncc) = threshold[i];

                    // Set E[new_cc, tind1] and E[tind1, new_cc] to threshold[i]
                    for (size_t idx1 = 0; idx1 < tind1.size(); ++idx1) {
                        int tind1_idx = tind1[idx1];
                        E.insert(ncc, tind1_idx) = threshold[i];
                        E.insert(tind1_idx, ncc) = threshold[i];
                    }

                    // Update history
                    history[ncc] = tind1;
                }
            }
        }
    }

    // Remove empty entries from CC and history
    int valid_entries = ncc + 1;
    CC.resize(valid_entries);
    history.resize(valid_entries);
    duration.conservativeResize(valid_entries, 2);
    E.conservativeResize(valid_entries, valid_entries);

    return std::make_tuple(CC, E, duration, history);
}