#include "make_original_dendrogram.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <set>
#include <queue>
#include <tuple>

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
std::vector<std::set<int>> connected_components_generator(const Eigen::SparseMatrix<int>& A) {
    std::vector<std::set<int>> components;
    std::vector<bool> visited(A.rows(), false);

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
                    for (Eigen::SparseMatrix<int>::InnerIterator it(A, curr); it; ++it) {
                        if (!visited[it.index()]) {
                            queue.push(it.index());
                        }
                    }
                }
            }
            components.push_back(cc_set);
        }
    }
    return components;
}

// Function to create the original dendrogram with connected components
std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<double>, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_original_dendrogram_cc(const Eigen::VectorXd& U, const Eigen::SparseMatrix<int>& A, const std::vector<double>& threshold) {
    int p = U.size();
    std::vector<std::vector<int>> CC(p);
    Eigen::SparseMatrix<double> E(p, p);
    Eigen::MatrixXd duration(p, 2);
    std::vector<std::vector<int>> history(p);

    // Initialize variables
    std::vector<int> ck_cc(p, -1);
    int ncc = -1;

    for (size_t i = 0; i < threshold.size(); ++i) {
        // Choose voxels that satisfy the threshold
        std::vector<int> cvoxels;
        for (int idx = 0; idx < p; ++idx) {
            if (U(idx) >= threshold[i] && (i == 0 || U(idx) < threshold[i - 1])) {
                cvoxels.push_back(idx);
            }
        }

        // Connected components for the current threshold
        std::vector<std::set<int>> CC_profiles = connected_components_generator(A);

        // Process each connected component
        for (const auto& component : CC_profiles) {
            std::vector<int> current_cc(component.begin(), component.end());
            ncc++;
            CC[ncc] = current_cc;
            ck_cc[current_cc[0]] = ncc;

            duration(ncc, 0) = threshold[i];
            E.coeffRef(ncc, ncc) = threshold[i];
            history[ncc] = {};  // Populate the history
        }
    }

    return std::make_tuple(CC, E, duration, history);
}