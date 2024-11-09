#include "make_smoothed_dendrogram.h"
#include "utils.h" // Include the shared utilities
#include <algorithm>
#include <iostream>
#include <set>
#include <vector>
#include <tuple>
#include <limits>

namespace STopoverUtils {

/**
 * @brief Constructs a smoothed dendrogram based on provided history and duration matrices.
 *
 * @param cCC Vector of connected components (each component is a vector of integers).
 * @param cE Adjacency matrix or similar representation (Eigen::MatrixXd).
 * @param cduration Matrix containing duration information (Eigen::MatrixXd).
 * @param chistory Vector of connected components history (each component is a vector of integers).
 * @param lim_size Vector of two elements specifying [min_size, max_size].
 * @return A tuple containing:
 *         - nCC: Updated vector of connected components.
 *         - nE: Updated adjacency matrix.
 *         - nduration: Updated duration matrix.
 *         - nchildren: Updated vector of connected components history.
 */
std::tuple<std::vector<std::vector<int>>, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_smoothed_dendrogram(const std::vector<std::vector<int>>& cCC,
                         const Eigen::MatrixXd& cE,
                         const Eigen::MatrixXd& cduration,
                         const std::vector<std::vector<int>>& chistory,
                         const Eigen::Vector2d& lim_size) {

    // Extract min and max size
    double min_size = lim_size(0);
    // double max_size = lim_size(1); // Unused variable removed

    Eigen::Index ncc = static_cast<Eigen::Index>(cCC.size());

    // Compute length_duration = cduration[:,0] - cduration[:,1]
    Eigen::VectorXd length_duration = cduration.col(0) - cduration.col(1);

    // Compute length_cc = [len(x) for x in cCC]
    std::vector<int> length_cc(ncc, 0);
    for (Eigen::Index i = 0; i < ncc; ++i) {
        length_cc[i] = static_cast<int>(cCC[i].size());
    }

    // Layer of dendrogram
    std::vector<std::vector<int>> nlayer;

    // Find CCs with no parent (history size == 0)
    std::vector<int> length_history(ncc, 0);
    for (Eigen::Index i = 0; i < ncc; ++i) {
        length_history[i] = static_cast<int>(chistory[i].size());
    }

    // Identify non-empty CCs
    std::vector<int> ind_notempty;
    for (Eigen::Index i = 0; i < ncc; ++i) {
        if (cduration.row(i).sum() != 0) {
            ind_notempty.push_back(static_cast<int>(i));
        }
    }

    // Identify empty CCs
    std::vector<int> ind_empty;
    for (Eigen::Index i = 0; i < ncc; ++i) {
        if (std::find(ind_notempty.begin(), ind_notempty.end(), static_cast<int>(i)) == ind_notempty.end()) {
            ind_empty.push_back(static_cast<int>(i));
        }
    }

    // Identify leaf CCs (no history and not empty)
    std::vector<int> ind_past;
    for (Eigen::Index i = 0; i < ncc; ++i) {
        if (length_history[i] == 0 &&
            std::find(ind_empty.begin(), ind_empty.end(), static_cast<int>(i)) == ind_empty.end()) {
            ind_past.push_back(static_cast<int>(i));
        }
    }
    nlayer.emplace_back(ind_past);

    // Iteratively find other layers
    while (static_cast<int>(ind_past.size()) < static_cast<int>(ind_notempty.size())) {
        std::vector<int> tind;
        for (Eigen::Index i = 0; i < ncc; ++i) {
            if (length_history[i] > 0 && is_subset(chistory[i], ind_past)) {
                tind.push_back(static_cast<int>(i));
            }
        }

        // Remove already included indices
        std::vector<int> ttind;
        for (const auto& idx : tind) {
            if (std::find(ind_past.begin(), ind_past.end(), idx) == ind_past.end()) {
                ttind.push_back(idx);
            }
        }

        if (!ttind.empty()) {
            nlayer.emplace_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        } else {
            break; // Prevent infinite loop in case of inconsistencies
        }
    }

    // Initialization
    std::vector<std::vector<int>> nCC = cCC;
    Eigen::MatrixXd nE = cE;
    Eigen::MatrixXd nduration = cduration;
    std::vector<std::vector<int>> nchildren = chistory;
    Eigen::VectorXi nparent = Eigen::VectorXi::Constant(ncc, -1);

    std::vector<int> ilayer(ncc, -1);

    for (Eigen::Index i = 0; i < ncc; ++i) {
        if (!nchildren[i].empty()) {
            for (const auto& child : nchildren[i]) {
                if (child >= 0 && child < ncc) { // Ensure child index is valid
                    nparent[child] = static_cast<int>(i);
                }
            }
        }
        // Find which layer the current CC belongs to
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(nlayer.size()); ++j) {
            if (std::find(nlayer[j].begin(), nlayer[j].end(), static_cast<int>(i)) != nlayer[j].end()) {
                ilayer[i] = static_cast<int>(j);
                break;
            }
        }
    }

    // Delete CCs of which size is smaller than min_size
    Eigen::VectorXi ck_delete = Eigen::VectorXi::Zero(ncc);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(nlayer.size()); ++i) {
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(nlayer[i].size()); ++j) {
            int ii = nlayer[i][j];
            if (ii != 0 && static_cast<double>(length_cc[ii]) < min_size && ck_delete[ii] == 0) { // Assuming min_size is duration(ii,1)
                if (nparent[ii] != -1) {
                    std::vector<int> jj = nchildren[nparent[ii]];
                    Eigen::VectorXi ck(jj.size());
                    for (Eigen::Index k = 0; k < jj.size(); ++k) {
                        ck(k) = (length_cc[jj[k]] >= min_size) ? 1 : 0;
                    }
                    if (ck.sum() <= 1) {
                        ii = nparent[ii];
                        if (ck.sum() == 1) {
                            // Find the index with ck == 1
                            int tind = -1;
                            for (Eigen::Index k = 0; k < ck.size(); ++k) {
                                if (ck(k) == 1) {
                                    tind = jj[k];
                                    break;
                                }
                            }
                            if (tind != -1) {
                                nchildren[ii] = nchildren[tind];
                                for (const auto& child : nchildren[ii]) {
                                    nparent[child] = ii;
                                }
                                // Update duration
                                nduration(ii, 0) = std::max(nduration(ii, 0), nduration(tind, 0));
                                nduration(ii, 1) = std::min(nduration(ii, 1), nduration(tind, 1));
                            }
                        } else {
                            // Update duration
                            double max_dur = -std::numeric_limits<double>::infinity();
                            double min_dur = std::numeric_limits<double>::infinity();
                            for (const auto& j_val : jj) {
                                max_dur = std::max(max_dur, nduration(j_val, 0));
                                min_dur = std::min(min_dur, nduration(j_val, 1));
                            }
                            nduration(ii, 0) = max_dur;
                            nduration(ii, 1) = min_dur;
                            nchildren[ii].clear();
                        }
                        // Update nE
                        nE.row(ii).setZero();
                        nE.col(ii).setZero();
                        nE(ii, ii) = nduration(ii, 0);

                        // Delete all children of the parent
                        for (const auto& j_val : jj) {
                            ck_delete[j_val] = 1;
                            nCC[j_val].clear();
                            nchildren[j_val].clear();
                            nparent[j_val] = -1;
                            nE.row(j_val).setZero();
                            nE.col(j_val).setZero();
                            nduration.row(j_val).setZero();
                            length_cc[j_val] = 0;
                            // Remove from layer
                            for (auto& layer_vec : nlayer) {
                                std::replace(layer_vec.begin(), layer_vec.end(), j_val, 0);
                            }
                        }
                    } else {
                        // Mark for deletion
                        ck_delete[ii] = 1;
                        if (ck.sum() <= static_cast<int>(ncc)) {
                            // Remove ii from its parent's children
                            auto& siblings = nchildren[nparent[ii]];
                            siblings.erase(std::remove(siblings.begin(), siblings.end(), ii), siblings.end());
                        }
                        // Clear CC
                        nCC[ii].clear();
                        nchildren[ii].clear();
                        nparent[ii] = -1;
                        nE.row(ii).setZero();
                        nE.col(ii).setZero();
                        nduration.row(ii).setZero();
                        length_cc[ii] = 0;
                        // Remove from layer
                        for (auto& layer_vec : nlayer) {
                            std::replace(layer_vec.begin(), layer_vec.end(), ii, 0);
                        }
                    }
                }
            }

            // Layer update
            nlayer.clear();
            // Recompute layers after deletion
            // Find CCs with no parent (history size == 0)
            ind_past.clear();
            for (Eigen::Index i = 0; i < ncc; ++i) {
                if (chistory[i].empty() && nduration.row(i).sum() != 0) {
                    ind_past.push_back(static_cast<int>(i));
                }
            }
            nlayer.emplace_back(ind_past);

            // Iteratively find other layers
            while (static_cast<int>(ind_past.size()) < static_cast<int>(ind_notempty.size())) {
                std::vector<int> tind;
                for (Eigen::Index i = 0; i < ncc; ++i) {
                    if (!chistory[i].empty() && is_subset(chistory[i], ind_past)) {
                        tind.push_back(static_cast<int>(i));
                    }
                }
                // Remove already included indices
                std::vector<int> ttind;
                for (const auto& idx : tind) {
                    if (std::find(ind_past.begin(), ind_past.end(), idx) == ind_past.end()) {
                        ttind.push_back(idx);
                    }
                }

                if (!ttind.empty()) {
                    nlayer.emplace_back(ttind);
                    ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
                } else {
                    break; // Prevent infinite loop in case of inconsistencies
                }
            }

            // Compute length_duration and length_cc again
            length_duration = nduration.col(0) - nduration.col(1);
            length_cc.assign(ncc, 0);
            for (Eigen::Index i = 0; i < ncc; ++i) {
                length_cc[i] = static_cast<int>(nCC[i].size());
            }

            // Sort CCs based on duration in descending order
            std::vector<std::pair<int, double>> sval_ind;
            for (Eigen::Index i = 0; i < ncc; ++i) {
                sval_ind.emplace_back(static_cast<int>(i), length_duration(i));
            }
            std::sort(sval_ind.begin(), sval_ind.end(),
                      [](const std::pair<int, double>& a, const std::pair<int, double>& b) -> bool {
                          return a.second > b.second;
                      });

            // Extract sorted indices
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
                    bool is_subset_e = true;
                    for (const auto& node : nCC[e]) {
                        if (std::find(nCC[ii].begin(), nCC[ii].end(), node) == nCC[ii].end()) {
                            is_subset_e = false;
                            break;
                        }
                    }
                    if (is_subset_e && e != ii) {
                        jj.push_back(e);
                    }
                }

                // Find parent candidates
                std::vector<int> iparent;
                for (const auto& e : ind_notempty) {
                    bool condition = false;
                    for (const auto& node : nCC[e]) {
                        if (std::find(nCC[ii].begin(), nCC[ii].end(), node) == nCC[ii].end()) {
                            condition = true;
                            break;
                        }
                    }
                    if (condition && e != ii && std::find(jj.begin(), jj.end(), e) == jj.end()) {
                        iparent.push_back(e);
                    }
                }

                // Update duration
                double max_dur = -std::numeric_limits<double>::infinity();
                double min_dur = std::numeric_limits<double>::infinity();
                for (const auto& j_val : jj) {
                    max_dur = std::max(max_dur, nduration(j_val, 0));
                    min_dur = std::min(min_dur, nduration(j_val, 1));
                }
                nduration(ii, 0) = std::max(nduration(ii, 0), max_dur);
                nduration(ii, 1) = std::min(nduration(ii, 1), min_dur);
                nchildren[ii].clear();
                nE.row(ii).setZero();
                nE.col(ii).setZero();
                nE(ii, ii) = nduration(ii, 0);

                // Delete children
                for (const auto& j_val : jj) {
                    nCC[j_val].clear();
                    nchildren[j_val].clear();
                    nparent[j_val] = -1;
                    nE.row(j_val).setZero();
                    nE.col(j_val).setZero();
                    nduration.row(j_val).setZero();
                }

                // Remove ii from sind
                sind.erase(sind.begin());

                // Remove jj from sind if present
                sind.erase(std::remove_if(sind.begin(), sind.end(),
                                          [&](int x) { return std::find(jj.begin(), jj.end(), x) != jj.end(); }),
                           sind.end());
            }

            // Ensure that all layers do not contain zero entries
            for (auto& layer_vec : nlayer) {
                layer_vec.erase(std::remove(layer_vec.begin(), layer_vec.end(), 0), layer_vec.end());
            }

            // Exit the loop after processing
            break;
        }

        // Ensure that all code paths return a value
        return std::make_tuple(nCC, nE, nduration, nchildren);
    }

} // namespace STopoverUtils