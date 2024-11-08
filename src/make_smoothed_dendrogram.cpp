#include "make_smoothed_dendrogram.h"
#include <algorithm>
#include <iostream>
#include <set>
#include <unordered_set>
#include <queue>

// Helper function to compute maximum element in a list of lists
int compute_p(const std::vector<std::vector<int>>& cCC) {
    int p = 0;
    for (const auto& cc : cCC) {
        if (!cc.empty()) {
            p = std::max(p, *std::max_element(cc.begin(), cc.end()));
        }
    }
    return p;
}

// Helper function to check if all elements of subset are in superset
bool is_subset(const std::vector<int>& subset, const std::vector<int>& superset) {
    return std::all_of(subset.begin(), subset.end(), [&](int x) {
        return std::find(superset.begin(), superset.end(), x) != superset.end();
    });
}

std::tuple<
    std::vector<std::vector<int>>,   // nCC
    Eigen::MatrixXd,                 // nE
    Eigen::MatrixXd,                 // nduration
    std::vector<std::vector<int>>    // nchildren
>
make_smoothed_dendrogram(
    const std::vector<std::vector<int>>& cCC,
    const Eigen::MatrixXd& cE,
    const Eigen::MatrixXd& cduration,
    const std::vector<std::vector<int>>& chistory,
    const Eigen::Vector2d& lim_size
) {
    double min_size = lim_size(0);
    double max_size = lim_size(1);

    int p = compute_p(cCC);
    int ncc = cCC.size();

    // Compute length_duration and length_cc
    Eigen::VectorXd length_duration = cduration.col(0).transpose() - cduration.col(1).transpose();
    std::vector<int> length_cc(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_cc[i] = cCC[i].size();
    }

    // Layer of dendrogram
    std::vector<std::vector<int>> layer;

    // Find CCs with no parent (history size == 0)
    std::vector<int> ind_past;
    for (int i = 0; i < ncc; ++i) {
        if (chistory[i].size() == 0) {
            ind_past.push_back(i);
        }
    }
    layer.push_back(ind_past);

    // Iteratively find other layers
    while (ind_past.size() < ncc) {
        std::vector<int> tind;
        for (int i = 0; i < ncc; ++i) {
            if (chistory[i].size() > 0 && is_subset(chistory[i], ind_past)) {
                tind.push_back(i);
            }
        }
        // Find indices not already in ind_past
        std::vector<int> ttind;
        for (const auto& idx : tind) {
            if (std::find(ind_past.begin(), ind_past.end(), idx) == ind_past.end()) {
                ttind.push_back(idx);
            }
        }
        if (!ttind.empty()) {
            layer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        } else {
            break; // Prevent infinite loop in case of inconsistencies
        }
    }

    // Initialization
    std::vector<std::vector<int>> nCC = cCC;
    Eigen::MatrixXd nduration = cduration;
    std::vector<std::vector<int>> nchildren = chistory;
    Eigen::MatrixXd nE = cE;
    Eigen::VectorXi nparent = Eigen::VectorXi::Constant(ncc, -1);
    std::vector<int> ilayer(ncc, -1);

    for (int i = 0; i < ncc; ++i) {
        if (!nchildren[i].empty()) {
            for (const auto& child : nchildren[i]) {
                nparent[child] = i;
            }
        }
        // Find which layer the current CC belongs to
        for (size_t j = 0; j < layer.size(); ++j) {
            if (std::find(layer[j].begin(), layer[j].end(), i) != layer[j].end()) {
                ilayer[i] = j;
                break;
            }
        }
    }

    // Delete CCs of which size is smaller than min_size
    Eigen::VectorXi ck_delete = Eigen::VectorXi::Zero(ncc);
    for (size_t i = 0; i < layer.size(); ++i) {
        for (size_t j = 0; j < layer[i].size(); ++j) {
            int ii = layer[i][j];
            if (ii != 0 && length_cc[ii] < min_size && ck_delete[ii] == 0) {
                if (nparent[ii] != -1) {
                    std::vector<int> jj = nchildren[nparent[ii]];
                    Eigen::VectorXi ck(jj.size());
                    for (size_t k = 0; k < jj.size(); ++k) {
                        ck(k) = (length_cc[jj[k]] >= min_size) ? 1 : 0;
                    }
                    if (ck.sum() <= 1) {
                        ii = nparent[ii];
                        if (ck.sum() == 1) {
                            // Find the index with ck == 1
                            int tind = -1;
                            for (size_t k = 0; k < ck.size(); ++k) {
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
                                nduration.row(ii).maxCoeff();
                                nduration.row(ii).minCoeff();
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
                        length_duration[ii] = nduration(ii, 0) - nduration(ii, 1);

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
                            length_duration[j_val] = 0;
                            // Remove from layer
                            for (auto& layer_vec : layer) {
                                std::replace(layer_vec.begin(), layer_vec.end(), j_val, 0);
                            }
                        }
                    } else {
                        // Mark for deletion
                        ck_delete[ii] = 1;
                        if (ck.sum() <= ncc) {
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
                        length_duration[ii] = 0;
                        // Remove from layer
                        for (auto& layer_vec : layer) {
                            std::replace(layer_vec.begin(), layer_vec.end(), ii, 0);
                        }
                    }
                } else {
                    // If no parent, just mark for deletion
                    ck_delete[ii] = 1;
                    nCC[ii].clear();
                    nchildren[ii].clear();
                    nparent[ii] = -1;
                    nE.row(ii).setZero();
                    nE.col(ii).setZero();
                    nduration.row(ii).setZero();
                    length_cc[ii] = 0;
                    length_duration[ii] = 0;
                    // Remove from layer
                    for (auto& layer_vec : layer) {
                        std::replace(layer_vec.begin(), layer_vec.end(), ii, 0);
                    }
                }
            }
        }

        // Layer update
        layer.clear();
        // Recompute length_history based on updated nchildren
        std::vector<int> length_history;
        for (const auto& children : nchildren) {
            length_history.push_back(children.size());
        }

        // Identify non-empty CCs
        std::vector<int> ind_notempty;
        for (int i = 0; i < ncc; ++i) {
            if (nduration.row(i).sum() != 0) {
                ind_notempty.push_back(i);
            }
        }
        // Identify empty CCs
        std::vector<int> ind_empty;
        for (int i = 0; i < ncc; ++i) {
            if (std::find(ind_notempty.begin(), ind_notempty.end(), i) == ind_notempty.end()) {
                ind_empty.push_back(i);
            }
        }
        // Identify leaf CCs excluding empty
        ind_past.clear();
        for (int i = 0; i < ncc; ++i) {
            if (length_history[i] == 0 && std::find(ind_empty.begin(), ind_empty.end(), i) == ind_empty.end()) {
                ind_past.push_back(i);
            }
        }
        layer.push_back(ind_past);

        // Iteratively find other layers
        while (ind_past.size() < ind_notempty.size()) {
            std::vector<int> tind;
            for (int i = 0; i < ncc; ++i) {
                if (nchildren[i].size() > 0 && is_subset(nchildren[i], ind_past)) {
                    tind.push_back(i);
                }
            }
            // Find indices not already in ind_past or ind_empty
            std::vector<int> ttind;
            for (const auto& idx : tind) {
                if (std::find(ind_past.begin(), ind_past.end(), idx) == ind_past.end() &&
                    std::find(ind_empty.begin(), ind_empty.end(), idx) == ind_empty.end()) {
                    ttind.push_back(idx);
                }
            }
            if (!ttind.empty()) {
                layer.push_back(ttind);
                ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
            } else {
                break; // Prevent infinite loop
            }
        }

        // Compute length_duration and length_cc again
        length_duration = nduration.col(0).transpose() - nduration.col(1).transpose();
        length_cc.assign(ncc, 0);
        for (int i = 0; i < ncc; ++i) {
            length_cc[i] = nCC[i].size();
        }

        // Sort CCs based on duration
        std::vector<std::pair<int, double>> sval_ind;
        for (int i = 0; i < length_duration.size(); ++i) {
            sval_ind.emplace_back(i, length_duration(i));
        }
        // Sort in descending order of duration
        std::sort(sval_ind.begin(), sval_ind.end(),
                  [&](const std::pair<int, double>& a, const std::pair<int, double>& b) -> bool {
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
            nduration.row(ii).maxCoeff();
            nduration.row(ii).minCoeff();

            // Clear children and update nE
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

        return std::make_tuple(nCC, nE, nduration, nchildren);
    }