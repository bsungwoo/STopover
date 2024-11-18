#include "make_smoothed_dendrogram.h"
#include <algorithm>
#include <numeric>
#include <set>
#include <iostream>

// Helper function to replace elements in a vector
template <typename T>
void replace_in_vector(std::vector<T>& vec, const T& old_value, const T& new_value) {
    std::replace(vec.begin(), vec.end(), old_value, new_value);
}

std::tuple<
    std::vector<std::vector<int>>,
    Eigen::SparseMatrix<double>,
    Eigen::MatrixXd,
    std::vector<std::vector<int>>
>
make_smoothed_dendrogram(
    const std::vector<std::vector<int>>& cCC,
    const Eigen::SparseMatrix<double>& cE,
    const Eigen::MatrixXd& cduration,
    const std::vector<std::vector<int>>& chistory,
    const Eigen::Vector2d& lim_size
)
{
    double min_size = lim_size[0];
    double max_size = lim_size[1];

    int ncc = cCC.size();
    Eigen::VectorXd length_duration = cduration.col(0) - cduration.col(1);
    Eigen::VectorXi length_cc(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_cc[i] = cCC[i].size();
    }

    // Layer of dendrogram
    std::vector<std::vector<int>> layer;

    // Find CCs with no parent
    Eigen::VectorXi length_history(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_history[i] = chistory[i].size();
    }

    // Leaf CCs
    std::vector<int> ind_past;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0) {
            ind_past.push_back(i);
        }
    }
    layer.push_back(ind_past);

    // Build layers
    while (ind_past.size() < static_cast<size_t>(ncc)) {
        std::vector<int> ttind;
        for (int i = 0; i < ncc; ++i) {
            // Sort chistory[i] and ind_past for set operations
            std::vector<int> chistory_i_sorted = chistory[i];
            std::sort(chistory_i_sorted.begin(), chistory_i_sorted.end());

            std::vector<int> ind_past_sorted = ind_past;
            std::sort(ind_past_sorted.begin(), ind_past_sorted.end());

            std::vector<int> intersect;
            std::set_intersection(
                chistory_i_sorted.begin(), chistory_i_sorted.end(),
                ind_past_sorted.begin(), ind_past_sorted.end(),
                std::back_inserter(intersect)
            );

            if (!chistory_i_sorted.empty() && intersect.size() == chistory_i_sorted.size()) {
                if (std::find(ind_past.begin(), ind_past.end(), i) == ind_past.end()) {
                    ttind.push_back(i);
                }
            }
        }

        if (!ttind.empty()) {
            layer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        }
        else {
            break;
        }
    }

    // Initialization
    std::vector<std::vector<int>> nCC = cCC;
    Eigen::MatrixXd nduration = cduration;
    std::vector<std::vector<int>> nchildren = chistory;
    Eigen::SparseMatrix<double> nE = cE;
    Eigen::VectorXi nparent = Eigen::VectorXi::Constant(ncc, -1);
    std::vector<int> ilayer(ncc, -1);

    for (int i = 0; i < ncc; ++i) {
        // Update nparent
        for (int child : nchildren[i]) {
            nparent[child] = i;
        }
        // Find layer index
        for (size_t l = 0; l < layer.size(); ++l) {
            if (std::find(layer[l].begin(), layer[l].end(), i) != layer[l].end()) {
                ilayer[i] = l;
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
                    int parent_idx = nparent[ii];
                    // Find siblings (children of the parent)
                    std::vector<int> jj = nchildren[parent_idx];

                    // Compute ck
                    std::vector<int> ck(jj.size());
                    for (size_t k = 0; k < jj.size(); ++k) {
                        ck[k] = (length_cc[jj[k]] >= min_size) ? 1 : 0;
                    }

                    int sum_ck = std::accumulate(ck.begin(), ck.end(), 0);

                    if (sum_ck <= 1) {
                        // All the children come back into the parent's belly
                        ii = parent_idx;
                        if (sum_ck == 1) {
                            // Find sibling to keep
                            int tind = -1;
                            for (size_t idx = 0; idx < ck.size(); ++idx) {
                                if (ck[idx] == 1) {
                                    tind = jj[idx];
                                    break;
                                }
                            }
                            if (tind != -1) {
                                nchildren[ii] = nchildren[tind];
                                for (int child : nchildren[ii]) {
                                    nparent[child] = ii;
                                }
                                // Update nduration
                                nduration.row(ii) = nduration.row(ii).cwiseMax(nduration.row(tind));
                            }
                        }
                        else {
                            // No siblings to keep
                            // Update nduration
                            std::vector<int> indices = jj;
                            indices.push_back(ii);
                            double max_start = nduration(indices[0], 0);
                            double min_end = nduration(indices[0], 1);
                            for (int idx : indices) {
                                max_start = std::max(max_start, nduration(idx, 0));
                                min_end = std::min(min_end, nduration(idx, 1));
                            }
                            nduration(ii, 0) = max_start;
                            nduration(ii, 1) = min_end;

                            nchildren[ii].clear();
                        }

                        // Update nE
                        // Zero out the row and column corresponding to ii
                        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
                            nE.coeffRef(ii, it.col()) = 0.0;
                        }
                        for (int k = 0; k < nE.outerSize(); ++k) {
                            for (Eigen::SparseMatrix<double>::InnerIterator it(nE, k); it; ++it) {
                                if (it.col() == ii) {
                                    nE.coeffRef(k, ii) = 0.0;
                                }
                            }
                        }
                        nE.coeffRef(ii, ii) = nduration(ii, 0);

                        length_duration[ii] = nduration(ii, 0) - nduration(ii, 1);

                        // Delete all children of my parent
                        std::vector<int> delete_list = jj;
                        for (size_t k = 0; k < jj.size(); ++k) {
                            if (ck[k] == 0) {
                                // Add indices of CCs identical to nCC[jj[k]]
                                for (int idx = 0; idx < ncc; ++idx) {
                                    if (ck_delete[idx] == 0 && nCC[idx] == nCC[jj[k]]) {
                                        delete_list.push_back(idx);
                                    }
                                }
                            }
                        }
                        // Remove duplicates
                        std::sort(delete_list.begin(), delete_list.end());
                        delete_list.erase(std::unique(delete_list.begin(), delete_list.end()), delete_list.end());

                        for (int idx : delete_list) {
                            ck_delete[idx] = 1;
                            nCC[idx].clear();
                            nchildren[idx].clear();
                            nparent[idx] = 0;
                            // Zero out the row and column in nE
                            for (Eigen::SparseMatrix<double>::InnerIterator it(nE, idx); it; ++it) {
                                nE.coeffRef(idx, it.col()) = 0.0;
                            }
                            for (int k = 0; k < nE.outerSize(); ++k) {
                                for (Eigen::SparseMatrix<double>::InnerIterator it(nE, k); it; ++it) {
                                    if (it.col() == idx) {
                                        nE.coeffRef(k, idx) = 0.0;
                                    }
                                }
                            }
                            nduration.row(idx).setZero();
                            length_cc[idx] = 0;
                            length_duration[idx] = 0;
                            int l_idx = ilayer[idx];
                            if (l_idx >= 0 && l_idx < layer.size()) {
                                replace_in_vector(layer[l_idx], idx, 0);
                            }
                        }
                    }
                    else {
                        // Delete current component
                        ck_delete[ii] = 1;
                        // Remove ii from parent's children
                        auto& parent_children = nchildren[parent_idx];
                        parent_children.erase(std::remove(parent_children.begin(), parent_children.end(), ii), parent_children.end());

                        nCC[ii].clear();
                        nchildren[ii].clear();
                        nparent[ii] = 0;
                        // Zero out the row and column in nE
                        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
                            nE.coeffRef(ii, it.col()) = 0.0;
                        }
                        for (int k = 0; k < nE.outerSize(); ++k) {
                            for (Eigen::SparseMatrix<double>::InnerIterator it(nE, k); it; ++it) {
                                if (it.col() == ii) {
                                    nE.coeffRef(k, ii) = 0.0;
                                }
                            }
                        }
                        nduration.row(ii).setZero();
                        length_cc[ii] = 0;
                        length_duration[ii] = 0;
                        int l_idx = ilayer[ii];
                        if (l_idx >= 0 && l_idx < layer.size()) {
                            replace_in_vector(layer[l_idx], ii, 0);
                        }
                    }
                }
                else {
                    // No parent, delete component
                    ck_delete[ii] = 1;
                    nCC[ii].clear();
                    nchildren[ii].clear();
                    nparent[ii] = 0;
                    // Zero out the row and column in nE
                    for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
                        nE.coeffRef(ii, it.col()) = 0.0;
                    }
                    for (int k = 0; k < nE.outerSize(); ++k) {
                        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, k); it; ++it) {
                            if (it.col() == ii) {
                                nE.coeffRef(k, ii) = 0.0;
                            }
                        }
                    }
                    nduration.row(ii).setZero();
                    length_cc[ii] = 0;
                    int l_idx = ilayer[ii];
                    if (l_idx >= 0 && l_idx < layer.size()) {
                        replace_in_vector(layer[l_idx], ii, 0);
                    }
                }
            }
        }
    }

    // Layer update
    // Estimate the depth of dendrogram
    layer.clear();
    length_duration = nduration.col(0) - nduration.col(1);

    std::vector<int> ind_notempty;
    for (int idx = 0; idx < ncc; ++idx) {
        if (nduration.row(idx).sum() != 0) {
            ind_notempty.push_back(idx);
        }
    }

    std::vector<int> ind_empty;
    for (int idx = 0; idx < ncc; ++idx) {
        if (nduration.row(idx).sum() == 0) {
            ind_empty.push_back(idx);
        }
    }

    ind_past.clear();
    for (int idx = 0; idx < ncc; ++idx) {
        if (nchildren[idx].empty() && nduration.row(idx).sum() != 0) {
            ind_past.push_back(idx);
        }
    }
    layer.push_back(ind_past);

    while (ind_past.size() < ind_notempty.size()) {
        std::vector<int> ttind;
        for (int idx : ind_notempty) {
            if (!nchildren[idx].empty()) {
                std::vector<int> intersect;
                std::set_intersection(
                    nchildren[idx].begin(), nchildren[idx].end(),
                    ind_past.begin(), ind_past.end(),
                    std::back_inserter(intersect)
                );
                if (intersect.size() == nchildren[idx].size()) {
                    if (std::find(ind_past.begin(), ind_past.end(), idx) == ind_past.end()) {
                        ttind.push_back(idx);
                    }
                }
            }
        }
        if (!ttind.empty()) {
            layer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        }
        else {
            break;
        }
    }

    // Update length_duration and length_cc
    for (int i = 0; i < ncc; ++i) {
        length_duration[i] = nduration(i, 0) - nduration(i, 1);
        length_cc[i] = nCC[i].size();
    }

    // Sort CCs by length_duration
    std::vector<std::pair<int, double>> sval_ind;
    for (int i = 0; i < ncc; ++i) {
        sval_ind.push_back({ i, length_duration[i] });
    }
    // Sort in descending order of duration
    std::sort(sval_ind.begin(), sval_ind.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
        return a.second > b.second;
    });

    // Filter out CCs with positive duration
    std::vector<int> sind;
    for (const auto& pair : sval_ind) {
        if (pair.second > 0) {
            sind.push_back(pair.first);
        }
    }

    // Find the CC with the maximum size
    int tval = 0;
    int tind = -1;
    for (int i = 0; i < ncc; ++i) {
        if (length_cc[i] > tval) {
            tval = length_cc[i];
            tind = i;
        }
    }
    // Reorder sind
    sind.erase(std::remove(sind.begin(), sind.end(), tind), sind.end());
    sind.push_back(tind);

    // Select CCs with the longest duration
    while (!sind.empty()) {
        int ii = sind[0];
        // Find CCs identical to nCC[ii]
        std::vector<int> jj;
        for (int idx : ind_notempty) {
            if (idx != ii && nCC[idx] == nCC[ii]) {
                jj.push_back(idx);
            }
        }

        // Find parents that include nCC[ii]
        std::vector<int> iparent;
        for (int idx : ind_notempty) {
            if (idx != ii && nCC[idx].size() > nCC[ii].size()) {
                std::vector<int> diff;
                std::set_difference(
                    nCC[idx].begin(), nCC[idx].end(),
                    nCC[ii].begin(), nCC[ii].end(),
                    std::back_inserter(diff)
                );
                if (diff.size() < nCC[idx].size()) {
                    iparent.push_back(idx);
                }
            }
        }

        // Update nduration
        std::vector<int> indices = jj;
        indices.push_back(ii);
        double max_start = nduration(ii, 0);
        double min_end = nduration(ii, 1);
        for (int idx : indices) {
            max_start = std::max(max_start, nduration(idx, 0));
            min_end = std::min(min_end, nduration(idx, 1));
        }
        nduration.row(ii) << max_start, min_end;

        nchildren[ii].clear();
        // Update nE
        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
            nE.coeffRef(ii, it.col()) = 0.0;
        }
        for (int k = 0; k < nE.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(nE, k); it; ++it) {
                if (it.col() == ii) {
                    nE.coeffRef(k, ii) = 0.0;
                }
            }
        }
        nE.coeffRef(ii, ii) = nduration(ii, 0);

        // Delete all identical CCs
        for (int idx : jj) {
            nCC[idx].clear();
            nchildren[idx].clear();
            nparent[idx] = 0;
            // Zero out the row and column in nE
            for (Eigen::SparseMatrix<double>::InnerIterator it(nE, idx); it; ++it) {
                nE.coeffRef(idx, it.col()) = 0.0;
            }
            for (int k = 0; k < nE.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(nE, k); it; ++it) {
                    if (it.col() == idx) {
                        nE.coeffRef(k, idx) = 0.0;
                    }
                }
            }
            nduration.row(idx).setZero();
        }

        // Remove processed indices from sind
        std::vector<int> to_remove = iparent;
        to_remove.insert(to_remove.end(), jj.begin(), jj.end());
        to_remove.push_back(ii);
        for (int idx : to_remove) {
            sind.erase(std::remove(sind.begin(), sind.end(), idx), sind.end());
        }
        // Update ind_notempty
        for (int idx : jj) {
            ind_notempty.erase(std::remove(ind_notempty.begin(), ind_notempty.end(), idx), ind_notempty.end());
        }
    }

    return std::make_tuple(nCC, nE, nduration, nchildren);
}