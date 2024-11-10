#include "make_smoothed_dendrogram.h"
#include "utils.h" // Include any shared utilities
#include <algorithm>
#include <iostream>
#include <set>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <numeric>      // For std::accumulate
#include <iterator>     // For std::set_difference

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
std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<double>, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_smoothed_dendrogram(const std::vector<std::vector<int>>& cCC,
                         const Eigen::SparseMatrix<double>& cE,
                         const Eigen::MatrixXd& cduration,
                         const std::vector<std::vector<int>>& chistory,
                         const Eigen::Vector2d& lim_size) {

    double min_size = lim_size[0];
    double max_size = lim_size[1];

    int ncc = static_cast<int>(cCC.size());

    // Compute length_duration = cduration[:,0] - cduration[:,1]
    Eigen::VectorXd length_duration = cduration.col(0) - cduration.col(1);

    // Compute length_cc = sizes of connected components
    std::vector<int> length_cc(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_cc[i] = static_cast<int>(cCC[i].size());
    }

    // Layer of dendrogram
    std::vector<std::vector<int>> layer;

    // Find CCs with no parent (history size == 0)
    std::vector<int> length_history(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_history[i] = static_cast<int>(chistory[i].size());
    }

    // Leaf CCs
    std::vector<int> ind_past;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0) {
            ind_past.push_back(i);
        }
    }
    layer.push_back(ind_past);

    // Build the dendrogram layers
    while (static_cast<int>(ind_past.size()) < ncc) {
        std::vector<int> tind(ncc, 0);
        for (int i = 0; i < ncc; ++i) {
            if (!chistory[i].empty()) {
                std::vector<int> intersection;
                std::set_intersection(chistory[i].begin(), chistory[i].end(),
                                      ind_past.begin(), ind_past.end(),
                                      std::back_inserter(intersection));
                if (static_cast<int>(intersection.size()) == static_cast<int>(chistory[i].size())) {
                    tind[i] = 1;
                }
            }
        }

        std::vector<int> ttind;
        for (int i = 0; i < ncc; ++i) {
            if (tind[i] == 1 && std::find(ind_past.begin(), ind_past.end(), i) == ind_past.end()) {
                ttind.push_back(i);
            }
        }

        if (!ttind.empty()) {
            std::sort(ttind.begin(), ttind.end());
            ttind.erase(std::unique(ttind.begin(), ttind.end()), ttind.end());
            layer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        } else {
            break; // Prevent infinite loop
        }
    }

    // Initialization
    auto nCC = cCC; // Copy of cCC
    auto nduration = cduration; // Copy of cduration
    auto nchildren = chistory; // Copy of chistory
    Eigen::SparseMatrix<double> nE = cE; // Copy of cE
    std::vector<int> nparent(ncc, -1);
    std::vector<int> ilayer(ncc, -1);

    for (int i = 0; i < ncc; ++i) {
        if (!nchildren[i].empty()) {
            for (int child : nchildren[i]) {
                nparent[child] = i;
            }
        }
        // Find the layer index for component i
        for (size_t l = 0; l < layer.size(); ++l) {
            if (std::find(layer[l].begin(), layer[l].end(), i) != layer[l].end()) {
                ilayer[i] = static_cast<int>(l);
                break; // Presumed to have only one nonzero element for each i
            }
        }
    }

    // Delete CCs of which size is smaller than min_size
    std::vector<int> ck_delete(ncc, 0);

    for (size_t i = 0; i < layer.size(); ++i) {
        for (size_t j = 0; j < layer[i].size(); ++j) {
            int ii = layer[i][j];
            if (ii != 0) {
                if ((length_cc[ii] < min_size) && (ck_delete[ii] == 0)) {
                    if (nparent[ii] != -1) {
                        // find sisters and brothers
                        std::vector<int> jj = nchildren[nparent[ii]];
                        // ck = np.array([1 if (e >= min_size) else 0 for e in length_cc[jj]])
                        std::vector<int> ck(jj.size());
                        for (size_t k = 0; k < jj.size(); ++k) {
                            ck[k] = (length_cc[jj[k]] >= min_size) ? 1 : 0;
                        }

                        int ck_sum = std::accumulate(ck.begin(), ck.end(), 0);

                        if (ck_sum <= 1) {
                            // All the children come back into the parent's belly
                            ii = nparent[ii];
                            if (ck_sum == 1) {
                                // tind = jj[np.where(ck == 1)[0]][0]
                                int tind = -1;
                                for (size_t k = 0; k < ck.size(); ++k) {
                                    if (ck[k] == 1) {
                                        tind = jj[k];
                                        break;
                                    }
                                }
                                if (tind == -1) {
                                    continue;
                                }
                                nchildren[ii] = nchildren[tind];
                                for (int child : nchildren[ii]) {
                                    nparent[child] = ii;
                                }
                                nduration(ii, 0) = std::max(nduration(ii, 0), nduration(tind, 0));
                                nduration(ii, 1) = std::min(nduration(ii, 1), nduration(tind, 1));
                            } else {
                                double max_val = nduration(ii, 0);
                                double min_val = nduration(ii, 1);
                                for (int idx : jj) {
                                    max_val = std::max(max_val, nduration(idx, 0));
                                    min_val = std::min(min_val, nduration(idx, 1));
                                }
                                nduration(ii, 0) = max_val;
                                nduration(ii, 1) = min_val;
                                nchildren[ii].clear();
                            }
                            // Update nE
                            for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
                                it.valueRef() = 0;
                            }
                            for (int row = 0; row < nE.rows(); ++row) {
                                nE.coeffRef(row, ii) = 0;
                            }
                            nE.coeffRef(ii, ii) = nduration(ii, 0);
                            length_duration[ii] = nduration(ii, 0) - nduration(ii, 1);

                            // delete all children of my parent
                            std::vector<int> delete_list = jj;
                            for (size_t k = 0; k < jj.size(); ++k) {
                                if (ck[k] == 0) {
                                    // ind_notdelete = np.where(ck_delete == 0)[0]
                                    std::vector<int> ind_notdelete;
                                    for (int idx = 0; idx < ncc; ++idx) {
                                        if (ck_delete[idx] == 0) {
                                            ind_notdelete.push_back(idx);
                                        }
                                    }
                                    // ind_children = ind_notdelete[np.where(list(map(lambda x: len(np.setdiff1d(nCC[x],nCC[jj[k]]))==0, ind_notdelete)))[0]]
                                    std::vector<int> ind_children;
                                    for (int idx : ind_notdelete) {
                                        std::vector<int> setdiff;
                                        std::set_difference(nCC[idx].begin(), nCC[idx].end(),
                                                            nCC[jj[k]].begin(), nCC[jj[k]].end(),
                                                            std::back_inserter(setdiff));
                                        if (setdiff.empty()) {
                                            ind_children.push_back(idx);
                                        }
                                    }
                                    delete_list.insert(delete_list.end(), ind_children.begin(), ind_children.end());
                                }
                            }
                            std::sort(delete_list.begin(), delete_list.end());
                            delete_list.erase(std::unique(delete_list.begin(), delete_list.end()), delete_list.end());
                            jj = delete_list;

                            for (int idx : jj) {
                                ck_delete[idx] = 1;
                                nCC[idx].clear();
                                nchildren[idx].clear();
                                nparent[idx] = 0;
                                for (Eigen::SparseMatrix<double>::InnerIterator it(nE, idx); it; ++it) {
                                    it.valueRef() = 0;
                                }
                                for (int row = 0; row < nE.rows(); ++row) {
                                    nE.coeffRef(row, idx) = 0;
                                }
                                nduration.row(idx).setZero();
                                length_cc[idx] = 0;
                                length_duration[idx] = 0;
                                int l_idx = ilayer[idx];
                                if (l_idx >= 0 && l_idx < static_cast<int>(layer.size())) {
                                    for (size_t m = 0; m < layer[l_idx].size(); ++m) {
                                        if (layer[l_idx][m] == idx) {
                                            layer[l_idx][m] = 0;
                                        }
                                    }
                                }
                            }
                        } else {
                            // Mark ii for deletion
                            ck_delete[ii] = 1;
                            if (ck_sum <= ncc) {
                                auto& siblings = nchildren[nparent[ii]];
                                siblings.erase(std::remove(siblings.begin(), siblings.end(), ii), siblings.end());
                            }
                            nCC[ii].clear();
                            nchildren[ii].clear();
                            nparent[ii] = 0;
                            for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
                                it.valueRef() = 0;
                            }
                            for (int row = 0; row < nE.rows(); ++row) {
                                nE.coeffRef(row, ii) = 0;
                            }
                            nduration.row(ii).setZero();
                            length_cc[ii] = 0;
                            length_duration[ii] = 0;
                            int l_idx = ilayer[ii];
                            if (l_idx >= 0 && l_idx < static_cast<int>(layer.size())) {
                                for (size_t m = 0; m < layer[l_idx].size(); ++m) {
                                    if (layer[l_idx][m] == ii) {
                                        layer[l_idx][m] = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Layer update
    // Estimate the depth of dendrogram
    layer.clear();
    length_history.clear();
    for (int i = 0; i < ncc; ++i) {
        length_history.push_back(static_cast<int>(nchildren[i].size()));
    }

    // ind_notempty = np.where(np.sum(nduration, axis=1) != 0)[0]
    std::vector<int> ind_notempty;
    for (int i = 0; i < ncc; ++i) {
        if (nduration.row(i).sum() != 0) {
            ind_notempty.push_back(i);
        }
    }

    // ind_empty = np.setdiff1d(range(len(nchildren)), ind_notempty)
    std::vector<int> all_indices(ncc);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::vector<int> ind_empty;
    std::set_difference(all_indices.begin(), all_indices.end(),
                        ind_notempty.begin(), ind_notempty.end(),
                        std::back_inserter(ind_empty));

    // ind_past = np.setdiff1d(np.where(length_history == 0)[0], ind_empty)
    std::vector<int> ind_past_temp;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0) {
            ind_past_temp.push_back(i);
        }
    }
    std::vector<int> ind_past_new;
    std::set_difference(ind_past_temp.begin(), ind_past_temp.end(),
                        ind_empty.begin(), ind_empty.end(),
                        std::back_inserter(ind_past_new));
    ind_past = ind_past_new;
    layer.push_back(ind_past);

    while (static_cast<int>(ind_past.size()) < static_cast<int>(ind_notempty.size())) {
        std::vector<int> tind(ncc, 0);
        for (int i = 0; i < ncc; ++i) {
            if (!nchildren[i].empty()) {
                std::vector<int> intersection;
                std::set_intersection(nchildren[i].begin(), nchildren[i].end(),
                                      ind_past.begin(), ind_past.end(),
                                      std::back_inserter(intersection));
                if (static_cast<int>(intersection.size()) == static_cast<int>(nchildren[i].size())) {
                    tind[i] = 1;
                }
            }
        }

        std::vector<int> temp;
        for (int i = 0; i < ncc; ++i) {
            if (tind[i] == 1 && std::find(ind_past.begin(), ind_past.end(), i) == ind_past.end()
                && std::find(ind_empty.begin(), ind_empty.end(), i) == ind_empty.end()) {
                temp.push_back(i);
            }
        }

        if (!temp.empty()) {
            layer.push_back(temp);
            ind_past.insert(ind_past.end(), temp.begin(), temp.end());
        } else {
            break; // Prevent infinite loop
        }
    }

    // Recalculate length_duration and length_cc
    length_duration = nduration.col(0) - nduration.col(1);
    for (int i = 0; i < ncc; ++i) {
        length_cc[i] = static_cast<int>(nCC[i].size());
    }

    // Sort CCs based on duration in descending order
    std::vector<std::pair<int, double>> sval_ind;
    for (int i = 0; i < ncc; ++i) {
        sval_ind.emplace_back(i, length_duration(i));
    }

    std::sort(sval_ind.begin(), sval_ind.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second > b.second;
              });

    // Extract indices and values where duration > 0
    std::vector<int> sind;
    for (const auto& pair : sval_ind) {
        if (pair.second > 0) {
            sind.push_back(pair.first);
        }
    }

    // Select CCs with the longest duration
    while (!sind.empty()) {
        int ii = sind[0];
        // Find all CCs in ind_notempty that are subsets of nCC[ii]
        std::vector<int> jj;
        for (const auto& e : ind_notempty) {
            if (e != ii && nCC[e] == nCC[ii]) {
                jj.push_back(e);
            }
        }

        // Find parent candidates
        std::vector<int> iparent;
        for (const auto& e : ind_notempty) {
            if (e != ii && std::includes(nCC[e].begin(), nCC[e].end(),
                                         nCC[ii].begin(), nCC[ii].end())) {
                iparent.push_back(e);
            }
        }
        iparent.erase(std::remove(iparent.begin(), iparent.end(), ii), iparent.end());
        for (const auto& val : jj) {
            iparent.erase(std::remove(iparent.begin(), iparent.end(), val), iparent.end());
        }

        // Update duration
        double max_dur = nduration(ii, 0);
        double min_dur = nduration(ii, 1);
        for (const auto& idx : jj) {
            max_dur = std::max(max_dur, nduration(idx, 0));
            min_dur = std::min(min_dur, nduration(idx, 1));
        }
        nduration(ii, 0) = max_dur;
        nduration(ii, 1) = min_dur;
        nchildren[ii].clear();

        // Update nE
        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
            it.valueRef() = 0;
        }
        for (int row = 0; row < nE.rows(); ++row) {
            nE.coeffRef(row, ii) = 0;
        }
        nE.coeffRef(ii, ii) = nduration(ii, 0);

        // Delete children
        for (const auto& idx : jj) {
            nCC[idx].clear();
            nchildren[idx].clear();
            nparent[idx] = 0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(nE, idx); it; ++it) {
                it.valueRef() = 0;
            }
            for (int row = 0; row < nE.rows(); ++row) {
                nE.coeffRef(row, idx) = 0;
            }
            nduration.row(idx).setZero();
        }

        // Remove processed indices from sind and ind_notempty
        sind.erase(std::remove_if(sind.begin(), sind.end(),
                                  [&](int x) { return x == ii || std::find(jj.begin(), jj.end(), x) != jj.end()
                                                 || std::find(iparent.begin(), iparent.end(), x) != iparent.end(); }),
                   sind.end());

        for (const auto& idx : jj) {
            ind_notempty.erase(std::remove(ind_notempty.begin(), ind_notempty.end(), idx), ind_notempty.end());
        }
    }

    return std::make_tuple(nCC, nE, nduration, nchildren);
}