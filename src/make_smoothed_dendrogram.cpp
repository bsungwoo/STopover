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
    const double min_size = lim_size[0];
    const double max_size = lim_size[1];
    const double EPSILON = 1e-9; // Tolerance for floating-point comparisons

    const int ncc = static_cast<int>(cCC.size());
    Eigen::VectorXd length_duration = cduration.col(0) - cduration.col(1);
    Eigen::VectorXi length_cc(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_cc[i] = static_cast<int>(cCC[i].size());
    }

    // Layers of dendrogram
    std::vector<std::vector<int>> layer;

    // Find CCs with no parent (leaf nodes)
    Eigen::VectorXi length_history(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_history[i] = static_cast<int>(chistory[i].size());
    }

    // Initialize ind_past with leaf CCs
    std::vector<int> ind_past;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0) {
            ind_past.push_back(i);
        }
    }
    layer.push_back(ind_past);

    // Build layers (First while loop)
    const int MAX_ITERATIONS = ncc * 10; // Safeguard to prevent infinite loops
    int iteration = 0;
    while (static_cast<int>(ind_past.size()) < ncc && iteration < MAX_ITERATIONS) {
        std::vector<int> ttind;
        for (int i = 0; i < ncc; ++i) {
            if (!chistory[i].empty()) {
                // Check if all children are in ind_past
                bool all_children_in_past = true;
                for (int child : chistory[i]) {
                    if (std::find(ind_past.begin(), ind_past.end(), child) == ind_past.end()) {
                        all_children_in_past = false;
                        break;
                    }
                }
                if (all_children_in_past && std::find(ind_past.begin(), ind_past.end(), i) == ind_past.end()) {
                    ttind.push_back(i);
                }
            }
        }

        if (!ttind.empty()) {
            layer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        } else {
            // No progress can be made; break to prevent infinite loop
            break;
        }

        iteration++;
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
                ilayer[i] = static_cast<int>(l);
                break;
            }
        }
    }

    // Delete CCs with size smaller than min_size
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
                                // Correct update to nduration
                                nduration(ii, 0) = std::max(nduration(ii, 0), nduration(tind, 0));
                                nduration(ii, 1) = std::min(nduration(ii, 1), nduration(tind, 1));
                            }
                        } else {
                            // No siblings to keep
                            // Update nduration
                            std::vector<int> indices = jj;
                            indices.push_back(ii);
                            double max_start = nduration(ii, 0);
                            double min_end = nduration(ii, 1);
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

                        // Delete all children of the parent
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
                            nparent[idx] = -1;
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
                            length_duration[idx] = 0.0;
                            int l_idx = ilayer[idx];
                            if (l_idx >= 0 && l_idx < static_cast<int>(layer.size())) {
                                replace_in_vector(layer[l_idx], idx, -1);
                            }
                        }
                    } else {
                        // Delete current component
                        ck_delete[ii] = 1;
                        // Remove ii from parent's children
                        auto& parent_children = nchildren[nparent[ii]];
                        parent_children.erase(std::remove(parent_children.begin(), parent_children.end(), ii), parent_children.end());

                        nCC[ii].clear();
                        nchildren[ii].clear();
                        nparent[ii] = -1;
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
                        length_duration[ii] = 0.0;
                        int l_idx = ilayer[ii];
                        if (l_idx >= 0 && l_idx < static_cast<int>(layer.size())) {
                            replace_in_vector(layer[l_idx], ii, -1);
                        }
                    }
                }
            }
        }
    }

    // Layer update (Second while loop)
    // Estimate the depth of dendrogram
    layer.clear();
    length_duration = nduration.col(0) - nduration.col(1);

    std::vector<int> ind_notempty;
    for (int idx = 0; idx < ncc; ++idx) {
        if (nduration.row(idx).norm() > EPSILON) {
            ind_notempty.push_back(idx);
        }
    }

    std::vector<int> ind_past_new;
    for (int idx = 0; idx < ncc; ++idx) {
        if (nchildren[idx].empty() && nduration.row(idx).norm() > EPSILON) {
            ind_past_new.push_back(idx);
        }
    }
    layer.push_back(ind_past_new);

    // Second while loop with safeguard
    const int MAX_ITERATIONS_SECOND_LOOP = ncc * 2; // Adjust as needed
    int current_iteration_second_loop = 0;

    while (static_cast<int>(ind_past_new.size()) < static_cast<int>(ind_notempty.size()) && current_iteration_second_loop < MAX_ITERATIONS_SECOND_LOOP) {
        std::vector<int> ttind;
        for (int idx : ind_notempty) {
            if (!nchildren[idx].empty()) {
                // Check if all children are in ind_past_new
                bool all_children_in_past = true;
                for (int child : nchildren[idx]) {
                    if (std::find(ind_past_new.begin(), ind_past_new.end(), child) == ind_past_new.end()) {
                        all_children_in_past = false;
                        break;
                    }
                }
                if (all_children_in_past && std::find(ind_past_new.begin(), ind_past_new.end(), idx) == ind_past_new.end()) {
                    ttind.push_back(idx);
                }
            }
        }
        if (!ttind.empty()) {
            layer.push_back(ttind);
            ind_past_new.insert(ind_past_new.end(), ttind.begin(), ttind.end());
        } else {
            // No progress can be made; break to prevent infinite loop
            break;
        }
        current_iteration_second_loop++;
    }

    // Update length_duration and length_cc
    for (int i = 0; i < ncc; ++i) {
        length_duration[i] = nduration(i, 0) - nduration(i, 1);
        length_cc[i] = static_cast<int>(nCC[i].size());
    }

    // Sort CCs by length_duration
    std::vector<std::pair<int, double>> sval_ind;
    for (int i = 0; i < ncc; ++i) {
        sval_ind.emplace_back(i, length_duration[i]);
    }
    // Sort in descending order of duration
    std::sort(sval_ind.begin(), sval_ind.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
        return a.second > b.second;
    });

    // Filter out CCs with positive duration
    std::vector<int> sind;
    for (const auto& pair : sval_ind) {
        if (pair.second > EPSILON) {
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
    // Reorder sind: remove 'tind' and append it at the end
    sind.erase(std::remove(sind.begin(), sind.end(), tind), sind.end());
    if (tind != -1) {
        sind.push_back(tind);
    }

    // Select CCs with the longest duration
    while (!sind.empty()) {
        int ii = sind[0];
        sind.erase(sind.begin());

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

        // Update nduration: max of first column, min of second column
        double max_start = nduration(ii, 0);
        double min_end = nduration(ii, 1);
        for (int idx : jj) {
            max_start = std::max(max_start, nduration(idx, 0));
            min_end = std::min(min_end, nduration(idx, 1));
        }
        nduration(ii, 0) = max_start;
        nduration(ii, 1) = min_end;

        nchildren[ii].clear();

        // Update nE: Zero out the row and column for ii, set self-loop
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

        // Delete all identical CCs (jj)
        for (int idx : jj) {
            nCC[idx].clear();
            nchildren[idx].clear();
            nparent[idx] = -1;
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
        for (int parent : iparent) {
            sind.erase(std::remove(sind.begin(), sind.end(), parent), sind.end());
        }
        for (int idx : jj) {
            sind.erase(std::remove(sind.begin(), sind.end(), idx), sind.end());
        }
        sind.erase(std::remove(sind.begin(), sind.end(), ii), sind.end());

        // Update ind_notempty by removing deleted indices
        for (int idx : jj) {
            ind_notempty.erase(std::remove(ind_notempty.begin(), ind_notempty.end(), idx), ind_notempty.end());
        }
    }

    return std::make_tuple(nCC, nE, nduration, nchildren);
}