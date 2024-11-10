#include "make_smoothed_dendrogram.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <tuple>
#include <set>
#include <unordered_set>

using namespace std;

/**
 * @brief Constructs a smoothed dendrogram based on provided history and duration matrices.
 *
 * @param cCC Vector of connected components (each component is a vector of integers).
 * @param cE Sparse adjacency matrix representing connections (Eigen::SparseMatrix<double>).
 * @param cduration Matrix containing duration information (Eigen::MatrixXd).
 * @param chistory Vector of connected components history (each component is a vector of integers).
 * @param lim_size Vector of two elements specifying [min_size, max_size].
 * @return A tuple containing:
 *         - nCC: Updated vector of connected components.
 *         - nE: Updated sparse adjacency matrix.
 *         - nduration: Updated duration matrix.
 *         - nchildren: Updated vector of connected components history.
 */
std::tuple<std::vector<std::vector<int>>,
           Eigen::SparseMatrix<double>,
           Eigen::MatrixXd,
           std::vector<std::vector<int>>>
make_smoothed_dendrogram(const std::vector<std::vector<int>>& cCC,
                         const Eigen::SparseMatrix<double>& cE,
                         const Eigen::MatrixXd& cduration,
                         const std::vector<std::vector<int>>& chistory,
                         const Eigen::Vector2d& lim_size) {
    double min_size = lim_size[0];
    double max_size = lim_size[1];

    int ncc = static_cast<int>(cCC.size());

    // Compute length_duration
    Eigen::VectorXd length_duration = cduration.col(0) - cduration.col(1);

    // Compute length_cc
    std::vector<int> length_cc(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_cc[i] = static_cast<int>(cCC[i].size());
    }

    // Initialize layers
    std::vector<std::vector<int>> nlayer;

    // Find CCs with no parent
    std::vector<int> length_history(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_history[i] = static_cast<int>(chistory[i].size());
    }

    // Identify non-empty CCs
    std::vector<int> ind_notempty;
    for (int i = 0; i < ncc; ++i) {
        if (cduration.row(i).sum() != 0) {
            ind_notempty.push_back(i);
        }
    }

    // Identify empty CCs
    std::vector<int> all_indices(ncc);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::vector<int> ind_empty;
    std::set_difference(all_indices.begin(), all_indices.end(),
                        ind_notempty.begin(), ind_notempty.end(),
                        std::back_inserter(ind_empty));

    // Identify leaf CCs
    std::vector<int> ind_past;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0 && std::find(ind_empty.begin(), ind_empty.end(), i) == ind_empty.end()) {
            ind_past.push_back(i);
        }
    }
    nlayer.push_back(ind_past);

    // Build the dendrogram layers
    while (static_cast<int>(ind_past.size()) < static_cast<int>(ind_notempty.size())) {
        std::vector<int> tind;
        for (int i = 0; i < ncc; ++i) {
            if (!chistory[i].empty()) {
                bool is_subset = true;
                for (const auto& h_elem : chistory[i]) {
                    if (std::find(ind_past.begin(), ind_past.end(), h_elem) == ind_past.end()) {
                        is_subset = false;
                        break;
                    }
                }
                if (is_subset) {
                    tind.push_back(i);
                }
            }
        }

        // Remove already included indices and empty indices
        std::vector<int> ttind;
        for (const auto& idx : tind) {
            if (std::find(ind_past.begin(), ind_past.end(), idx) == ind_past.end() &&
                std::find(ind_empty.begin(), ind_empty.end(), idx) == ind_empty.end()) {
                ttind.push_back(idx);
            }
        }

        if (!ttind.empty()) {
            nlayer.emplace_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        } else {
            break; // Prevent infinite loop
        }
    }

    // Initialization
    auto nCC = cCC;  // Copy of cCC
    auto nduration = cduration;  // Copy of cduration
    auto nchildren = chistory;  // Copy of chistory
    Eigen::SparseMatrix<double> nE = cE;  // Copy of cE
    std::vector<int> nparent(ncc, -1);
    std::vector<int> ilayer(ncc, -1);

    for (int i = 0; i < ncc; ++i) {
        if (!nchildren[i].empty()) {
            for (int child : nchildren[i]) {
                nparent[child] = i;
            }
        }

        // Find the layer index for component i
        for (size_t l = 0; l < nlayer.size(); ++l) {
            if (std::find(nlayer[l].begin(), nlayer[l].end(), i) != nlayer[l].end()) {
                ilayer[i] = static_cast<int>(l);
                break; // Assuming each component appears in only one layer
            }
        }
    }

    // Delete CCs smaller than min_size
    std::vector<int> ck_delete(ncc, 0);
    for (size_t i = 0; i < nlayer.size(); ++i) {
        for (size_t j = 0; j < nlayer[i].size(); ++j) {
            int ii = nlayer[i][j];
            if (ii != 0 && length_cc[ii] < min_size && ck_delete[ii] == 0) {
                if (nparent[ii] != -1) {
                    // Find siblings
                    int parent_idx = nparent[ii];
                    std::vector<int>& siblings = nchildren[parent_idx];
                    std::vector<int> ck(siblings.size());
                    for (size_t k = 0; k < siblings.size(); ++k) {
                        ck[k] = (length_cc[siblings[k]] >= min_size) ? 1 : 0;
                    }

                    int ck_sum = std::accumulate(ck.begin(), ck.end(), 0);

                    if (ck_sum <= 1) {
                        // All children merge into the parent
                        ii = parent_idx;
                        if (ck_sum == 1) {
                            // Find the sibling with size >= min_size
                            int tind = -1;
                            for (size_t k = 0; k < ck.size(); ++k) {
                                if (ck[k] == 1) {
                                    tind = siblings[k];
                                    break;
                                }
                            }
                            if (tind != -1) {
                                nchildren[ii] = nchildren[tind];
                                for (int child : nchildren[ii]) {
                                    nparent[child] = ii;
                                }
                                nduration.row(ii) = nduration.row(ii).array().max(nduration.row(tind).array());
                            }
                        } else {
                            // No child has size >= min_size
                            double max_dur = nduration(ii, 0);
                            double min_dur = nduration(ii, 1);
                            for (int idx : siblings) {
                                max_dur = std::max(max_dur, nduration(idx, 0));
                                min_dur = std::min(min_dur, nduration(idx, 1));
                            }
                            nduration(ii, 0) = max_dur;
                            nduration(ii, 1) = min_dur;
                            nchildren[ii].clear();
                        }

                        // Update adjacency matrix nE
                        nE.row(ii).setZero();
                        nE.col(ii).setZero();
                        nE.coeffRef(ii, ii) = nduration(ii, 0);

                        // Delete all children of the parent
                        std::vector<int> delete_list = siblings;
                        for (size_t k = 0; k < siblings.size(); ++k) {
                            if (ck[k] == 0) {
                                delete_list.push_back(siblings[k]);
                            }
                        }

                        // Remove duplicates
                        std::sort(delete_list.begin(), delete_list.end());
                        delete_list.erase(std::unique(delete_list.begin(), delete_list.end()), delete_list.end());

                        for (int idx : delete_list) {
                            ck_delete[idx] = 1;
                            nCC[idx].clear();
                            nchildren[idx].clear();
                            nparent[idx] = -1;
                            nE.row(idx).setZero();
                            nE.col(idx).setZero();
                            nduration.row(idx).setZero();
                            length_cc[idx] = 0;
                            // Remove idx from layer
                            int layer_idx = ilayer[idx];
                            if (layer_idx != -1) {
                                auto& layer_components = nlayer[layer_idx];
                                layer_components.erase(std::remove(layer_components.begin(), layer_components.end(), idx), layer_components.end());
                            }
                        }
                    } else {
                        // Mark ii for deletion
                        ck_delete[ii] = 1;
                        // Remove ii from its parent's children
                        auto& siblings = nchildren[nparent[ii]];
                        siblings.erase(std::remove(siblings.begin(), siblings.end(), ii), siblings.end());
                        // Clear data for ii
                        nCC[ii].clear();
                        nchildren[ii].clear();
                        nparent[ii] = -1;
                        nE.row(ii).setZero();
                        nE.col(ii).setZero();
                        nduration.row(ii).setZero();
                        length_cc[ii] = 0;
                        // Remove ii from layer
                        int layer_idx = ilayer[ii];
                        if (layer_idx != -1) {
                            auto& layer_components = nlayer[layer_idx];
                            layer_components.erase(std::remove(layer_components.begin(), layer_components.end(), ii), layer_components.end());
                        }
                    }
                } else {
                    // No parent; mark for deletion
                    ck_delete[ii] = 1;
                    nCC[ii].clear();
                    nchildren[ii].clear();
                    nparent[ii] = -1;
                    nE.row(ii).setZero();
                    nE.col(ii).setZero();
                    nduration.row(ii).setZero();
                    length_cc[ii] = 0;
                    // Remove ii from layer
                    int layer_idx = ilayer[ii];
                    if (layer_idx != -1) {
                        auto& layer_components = nlayer[layer_idx];
                        layer_components.erase(std::remove(layer_components.begin(), layer_components.end(), ii), layer_components.end());
                    }
                }
            }
        }
    }

    // Update layers
    nlayer.clear();
    length_history.clear();
    for (int i = 0; i < ncc; ++i) {
        length_history.push_back(static_cast<int>(nchildren[i].size()));
    }

    // Identify non-empty CCs
    ind_notempty.clear();
    for (int i = 0; i < ncc; ++i) {
        if (nduration.row(i).sum() != 0) {
            ind_notempty.push_back(i);
        }
    }

    // Identify empty CCs
    ind_empty.clear();
    std::set_difference(all_indices.begin(), all_indices.end(),
                        ind_notempty.begin(), ind_notempty.end(),
                        std::back_inserter(ind_empty));

    // Identify leaf CCs
    ind_past.clear();
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0 && std::find(ind_empty.begin(), ind_empty.end(), i) == ind_empty.end()) {
            ind_past.push_back(i);
        }
    }
    nlayer.push_back(ind_past);

    // Build the dendrogram layers again
    while (static_cast<int>(ind_past.size()) < static_cast<int>(ind_notempty.size())) {
        std::vector<int> tind;
        for (int i = 0; i < ncc; ++i) {
            if (!nchildren[i].empty()) {
                bool is_subset = true;
                for (const auto& h_elem : nchildren[i]) {
                    if (std::find(ind_past.begin(), ind_past.end(), h_elem) == ind_past.end()) {
                        is_subset = false;
                        break;
                    }
                }
                if (is_subset) {
                    tind.push_back(i);
                }
            }
        }

        // Remove already included indices and empty indices
        std::vector<int> ttind;
        for (const auto& idx : tind) {
            if (std::find(ind_past.begin(), ind_past.end(), idx) == ind_past.end() &&
                std::find(ind_empty.begin(), ind_empty.end(), idx) == ind_empty.end()) {
                ttind.push_back(idx);
            }
        }

        if (!ttind.empty()) {
            nlayer.emplace_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        } else {
            break; // Prevent infinite loop
        }
    }

    // Recalculate length_duration and length_cc
    length_duration = nduration.col(0) - nduration.col(1);
    for (int i = 0; i < ncc; ++i) {
        length_cc[i] = static_cast<int>(nCC[i].size());
    }

    return std::make_tuple(nCC, nE, nduration, nchildren);
}