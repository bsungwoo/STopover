#include "make_original_dendrogram.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// =================== Helper Functions ===================

// Helper function to extract connected nodes using BFS
std::set<int> extract_connected_nodes(const std::vector<std::vector<int>>& edge_list, int sel_node_idx) {
    std::unordered_set<int> cc_set_unordered;
    std::queue<int> to_visit;
    to_visit.push(sel_node_idx);
    cc_set_unordered.insert(sel_node_idx);

    while (!to_visit.empty()) {
        int vertex = to_visit.front();
        to_visit.pop();

        for (int neighbor : edge_list[vertex]) {
            if (cc_set_unordered.find(neighbor) == cc_set_unordered.end()) {
                cc_set_unordered.insert(neighbor);
                to_visit.push(neighbor);
            }
        }
    }

    // Convert unordered_set to set to maintain consistency
    std::set<int> cc_set(cc_set_unordered.begin(), cc_set_unordered.end());
    return cc_set;
}

// Function to find connected components in a graph represented by an edge list
std::vector<std::set<int>> connected_components(const std::vector<std::vector<int>>& edge_list) {
    size_t n = edge_list.size();
    std::unordered_set<int> all_cc_set;
    std::vector<std::set<int>> connected_components_list;

    for (size_t vertex = 0; vertex < n; ++vertex) {
        if (all_cc_set.find(vertex) == all_cc_set.end()) {
            std::set<int> cc_set = extract_connected_nodes(edge_list, vertex);
            all_cc_set.insert(cc_set.begin(), cc_set.end());
            connected_components_list.push_back(cc_set);
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
    std::vector<std::vector<int>> CC;  // Initialize CC as an empty vector
    Eigen::VectorXi ck_cc = Eigen::VectorXi::Constant(p, -1);
    Eigen::MatrixXd duration(0, 2);    // Initialize duration with 0 rows
    std::vector<std::vector<int>> history; // Initialize history as an empty vector

    // Initialize E as an empty sparse matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> E(0, 0);

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
        std::unordered_map<int, size_t> voxel_to_index;
        for (size_t idx = 0; idx < cvoxels.size(); ++idx) {
            voxel_to_index[cvoxels[idx]] = idx;
        }

        // Build sub adjacency matrix
        size_t n_cvoxels = cvoxels.size();
        std::vector<std::vector<int>> edge_list(n_cvoxels);

        for (size_t idx = 0; idx < cvoxels.size(); ++idx) {
            int original_idx = cvoxels[idx];
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, original_idx); it; ++it) {
                int col = it.col();
                if (voxel_to_index.find(col) != voxel_to_index.end() && col != original_idx) {
                    edge_list[idx].push_back(voxel_to_index[col]);
                }
            }
        }

        // Extract connected components
        std::vector<std::set<int>> CC_profiles = connected_components(edge_list);
        size_t S = CC_profiles.size();
        std::vector<std::vector<int>> nCC(S);
        Eigen::SparseMatrix<double, Eigen::RowMajor> nA(p, S);
        std::vector<int> neighbor_cc;

        for (size_t j = 0; j < S; ++j) {
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
            for (size_t j = 0; j < S; ++j) {
                // Update ck_cc
                int cc_idx = CC.size(); // Next CC index
                for (int idx : nCC[j]) {
                    ck_cc[idx] = cc_idx;
                }
                CC.push_back(nCC[j]);

                // Update duration
                duration.conservativeResize(CC.size(), 2);
                duration(CC.size() - 1, 0) = threshold[i];

                // Update E
                E.conservativeResize(CC.size(), CC.size());
                E.insert(CC.size() - 1, CC.size() - 1) = threshold[i];

                // Update history
                history.push_back(std::vector<int>());
            }
        } else {
            // Construct nA_tmp
            size_t N = neighbor_cc.size() + S;
            Eigen::SparseMatrix<double, Eigen::RowMajor> nA_tmp(N, N);

            // Set diagonal elements to 1
            for (size_t idx = 0; idx < N; ++idx) {
                nA_tmp.insert(idx, idx) = 1;
            }

            // Fill nA_tmp
            for (size_t row = 0; row < neighbor_cc.size(); ++row) {
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(nA, neighbor_cc[row]); it; ++it) {
                    nA_tmp.insert(row, neighbor_cc.size() + it.col()) = it.value();
                    nA_tmp.insert(neighbor_cc.size() + it.col(), row) = it.value();
                }
            }

            // Extract connected components from nA_tmp
            std::vector<std::vector<int>> edge_list_tmp(N);
            for (size_t k = 0; k < nA_tmp.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(nA_tmp, k); it; ++it) {
                    if (it.row() != it.col() && it.value() != 0) {
                        edge_list_tmp[it.row()].push_back(it.col());
                    }
                }
            }

            std::vector<std::set<int>> CC_profiles_tmp = connected_components(edge_list_tmp);
            size_t S_tmp = CC_profiles_tmp.size();

            for (size_t j = 0; j < S_tmp; ++j) {
                std::vector<int> tind(CC_profiles_tmp[j].begin(), CC_profiles_tmp[j].end());
                std::vector<int> tind1, tind2;
                for (size_t idx = 0; idx < tind.size(); ++idx) {
                    if (tind[idx] < neighbor_cc.size()) {
                        tind1.push_back(neighbor_cc[tind[idx]]);
                    } else {
                        tind2.push_back(tind[idx] - neighbor_cc.size());
                    }
                }

                if (tind1.size() == 1) {
                    int cc_idx = tind1[0];
                    // Merge nCC[tind2] into CC[cc_idx]
                    for (int idx : tind2) {
                        CC[cc_idx].insert(CC[cc_idx].end(), nCC[idx].begin(), nCC[idx].end());
                    }
                    // Update ck_cc
                    for (int idx : CC[cc_idx]) {
                        ck_cc[idx] = cc_idx;
                    }
                } else {
                    // Create a new connected component
                    int new_cc_idx = CC.size();
                    std::vector<int> new_CC;
                    for (int idx : tind1) {
                        new_CC.insert(new_CC.end(), CC[idx].begin(), CC[idx].end());
                    }
                    for (int idx : tind2) {
                        new_CC.insert(new_CC.end(), nCC[idx].begin(), nCC[idx].end());
                    }

                    // Update ck_cc
                    for (int idx : new_CC) {
                        ck_cc[idx] = new_cc_idx;
                    }

                    // Add new connected component
                    CC.push_back(new_CC);

                    // Update duration
                    duration.conservativeResize(CC.size(), 2);
                    duration(CC.size() - 1, 0) = threshold[i];
                    for (int idx : tind1) {
                        duration(idx, 1) = threshold[i];
                    }

                    // Update E
                    E.conservativeResize(CC.size(), CC.size());
                    for (size_t idx1 = 0; idx1 < tind1.size(); ++idx1) {
                        for (size_t idx2 = idx1; idx2 < tind1.size(); ++idx2) {
                            E.coeffRef(tind1[idx1], tind1[idx2]) = threshold[i];
                            E.coeffRef(tind1[idx2], tind1[idx1]) = threshold[i];
                        }
                        E.coeffRef(new_cc_idx, tind1[idx1]) = threshold[i];
                        E.coeffRef(tind1[idx1], new_cc_idx) = threshold[i];
                    }
                    E.coeffRef(new_cc_idx, new_cc_idx) = threshold[i];

                    // Update history
                    history.push_back(tind1);
                }
            }
        }
    }

    // No need to remove empty entries since CC is dynamically sized

    return std::make_tuple(CC, E, duration, history);
}