#include "make_smoothed_dendrogram.h"
#include <algorithm>
#include <numeric>
#include <set>
#include <iostream>

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
    const Eigen::Vector2d lim_size
)
{
    double min_size = lim_size[0];
    double max_size = lim_size[1];

    int p = 0;
    for (const auto& cc : cCC) {
        if (!cc.empty()) {
            int local_max = *std::max_element(cc.begin(), cc.end());
            if (local_max > p) {
                p = local_max;
            }
        }
    }

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
        if (length_history[i] == 0 && length_cc[i] > 0) {
            ind_past.push_back(i);
        }
    }
    layer.push_back(ind_past);

    // Build layers
    while (ind_past.size() < static_cast<size_t>(ncc)) {
        Eigen::VectorXi tind(ncc);
        for (int i = 0; i < ncc; ++i) {
            std::vector<int> intersect;
            std::set_intersection(
                chistory[i].begin(), chistory[i].end(),
                ind_past.begin(), ind_past.end(),
                std::back_inserter(intersect)
            );
            tind[i] = (intersect.size() == chistory[i].size()) ? 1 : 0;
        }

        std::vector<int> ttind;
        for (int i = 0; i < ncc; ++i) {
            if (tind[i] == 1 && std::find(ind_past.begin(), ind_past.end(), i) == ind_past.end()) {
                ttind.push_back(i);
            }
        }

        if (!ttind.empty()) {
            layer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        } else {
            break;
        }
    }

    // Initialization
    std::vector<std::vector<int>> nCC = cCC;
    Eigen::MatrixXd nduration = cduration;
    std::vector<std::vector<int>> nchildren = chistory;
    Eigen::SparseMatrix<double> nE = cE;
    Eigen::VectorXi nparent = Eigen::VectorXi::Constant(ncc, -1);
    std::vector<int> ilayer(ncc);

    for (int i = 0; i < ncc; ++i) {
        if (!nchildren[i].empty()) {
            for (size_t j = 0; j < nchildren[i].size(); ++j) {
                nparent[nchildren[i][j]] = i;
            }
        }
        // Find the index in 'layer' which contains i
        int layer_index = -1;
        for (size_t l = 0; l < layer.size(); ++l) {
            if (std::find(layer[l].begin(), layer[l].end(), i) != layer[l].end()) {
                layer_index = static_cast<int>(l);
                break;
            }
        }
        ilayer[i] = layer_index;
    }

    // Delete CCs of which size is smaller than min_size
    Eigen::VectorXi ck_delete = Eigen::VectorXi::Zero(ncc);

    for (size_t i = 0; i < layer.size(); ++i) {
        for (size_t j = 0; j < layer[i].size(); ++j) {
            int ii = layer[i][j];
            if (ii != 0) {
                if ((length_cc[ii] < min_size) && (ck_delete[ii] == 0)) {
                    std::vector<int> jj;
                    std::vector<int> ck;
                    if (nparent[ii] != -1) {
                        // Find siblings
                        jj = nchildren[nparent[ii]];
                        ck.resize(jj.size());
                        for (size_t k = 0; k < jj.size(); ++k) {
                            ck[k] = (length_cc[jj[k]] >= min_size) ? 1 : 0;
                        }
                    } else {
                        ck.push_back(ncc + 1);
                    }

                    int sum_ck = std::accumulate(ck.begin(), ck.end(), 0);
                    if (sum_ck <= 1) {
                        // All the children come back into the parent's belly
                        ii = nparent[ii];
                        if (sum_ck == 1) {
                            std::vector<int> indices_where_ck_is_1;
                            for (size_t idx = 0; idx < ck.size(); ++idx) {
                                if (ck[idx] == 1) {
                                    indices_where_ck_is_1.push_back(idx);
                                }
                            }
                            int tind = jj[indices_where_ck_is_1[0]];

                            nchildren[ii] = nchildren[tind];
                            for (int child : nchildren[ii]) {
                                nparent[child] = ii;
                            }

                            nduration(ii, 0) = std::max(nduration(ii, 0), nduration(tind, 0));
                            nduration(ii, 1) = std::min(nduration(ii, 1), nduration(tind, 1));
                        } else {
                            std::vector<int> indices_to_consider = jj;
                            indices_to_consider.push_back(ii);

                            double max_d0 = nduration(ii, 0);
                            double min_d1 = nduration(ii, 1);
                            for (int idx : jj) {
                                max_d0 = std::max(max_d0, nduration(idx, 0));
                                min_d1 = std::min(min_d1, nduration(idx, 1));
                            }
                            nduration(ii, 0) = max_d0;
                            nduration(ii, 1) = min_d1;
                            nchildren[ii].clear();
                        }

                        // Update nE
                        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
                            nE.coeffRef(ii, it.col()) = 0.0;
                        }
                        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
                            nE.coeffRef(it.row(), ii) = 0.0;
                        }
                        nE.coeffRef(ii, ii) = nduration(ii, 0);
                        length_duration[ii] = nduration(ii, 0) - nduration(ii, 1);

                        std::vector<int> delete_list = jj;
                        for (size_t k = 0; k < jj.size(); ++k) {
                            if (ck[k] == 0) {
                                std::vector<int> ind_notdelete;
                                for (int idx = 0; idx < ncc; ++idx) {
                                    if (ck_delete[idx] == 0) {
                                        ind_notdelete.push_back(idx);
                                    }
                                }

                                std::vector<int> ind_children;
                                for (int idx : ind_notdelete) {
                                    std::vector<int> diff;
                                    std::set_difference(
                                        nCC[idx].begin(), nCC[idx].end(),
                                        nCC[jj[k]].begin(), nCC[jj[k]].end(),
                                        std::back_inserter(diff)
                                    );
                                    if (diff.empty()) {
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
                                nE.coeffRef(idx, it.col()) = 0.0;
                            }
                            for (Eigen::SparseMatrix<double>::InnerIterator it(nE, idx); it; ++it) {
                                nE.coeffRef(it.row(), idx) = 0.0;
                            }
                            nduration.row(idx).setZero();
                            length_cc[idx] = 0;
                            length_duration[idx] = 0;
                            int l_idx = ilayer[idx];
                            for (size_t l = 0; l < layer[l_idx].size(); ++l) {
                                if (layer[l_idx][l] == idx) {
                                    layer[l_idx][l] = 0;
                                }
                            }
                        }
                    } else {
                        ck_delete[ii] = 1;
                        if (sum_ck <= ncc) {
                            std::vector<int>& parent_children = nchildren[nparent[ii]];
                            parent_children.erase(std::remove(parent_children.begin(), parent_children.end(), ii), parent_children.end());
                        }
                        nCC[ii].clear();
                        nchildren[ii].clear();
                        nparent[ii] = 0;
                        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
                            nE.coeffRef(ii, it.col()) = 0.0;
                        }
                        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
                            nE.coeffRef(it.row(), ii) = 0.0;
                        }
                        nduration.row(ii).setZero();
                        length_cc[ii] = 0;
                        length_duration[ii] = 0;
                        int l_idx = ilayer[ii];
                        for (size_t l = 0; l < layer[l_idx].size(); ++l) {
                            if (layer[l_idx][l] == ii) {
                                layer[l_idx][l] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    // Layer update
    layer.clear();
    length_history.resize(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_history[i] = nchildren[i].size();
    }

    std::vector<int> ind_notempty;
    for (int i = 0; i < ncc; ++i) {
        if (nduration.row(i).sum() != 0) {
            ind_notempty.push_back(i);
        }
    }

    std::vector<int> ind_past;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0 && nduration.row(i).sum() != 0) {
            ind_past.push_back(i);
        }
    }
    layer.push_back(ind_past);

    while (ind_past.size() < ind_notempty.size()) {
        Eigen::VectorXi tind(ncc);
        for (int i = 0; i < ncc; ++i) {
            std::vector<int> intersect;
            std::set_intersection(
                nchildren[i].begin(), nchildren[i].end(),
                ind_past.begin(), ind_past.end(),
                std::back_inserter(intersect)
            );
            tind[i] = (intersect.size() == nchildren[i].size()) ? 1 : 0;
        }

        std::vector<int> ttind;
        for (int i = 0; i < ncc; ++i) {
            if (tind[i] == 1 && std::find(ind_past.begin(), ind_past.end(), i) == ind_past.end() && nduration.row(i).sum() != 0) {
                ttind.push_back(i);
            }
        }

        if (!ttind.empty()) {
            layer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        } else {
            break;
        }
    }

    length_duration = nduration.col(0) - nduration.col(1);
    for (int i = 0; i < ncc; ++i) {
        length_cc[i] = nCC[i].size();
    }

    // Sort connected components by duration length
    std::vector<std::pair<int, double>> sval_ind;
    for (int i = 0; i < length_duration.size(); ++i) {
        sval_ind.emplace_back(i, length_duration[i]);
    }
    std::sort(sval_ind.begin(), sval_ind.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
        return a.second > b.second;
    });

    std::vector<int> sind;
    for (const auto& pair : sval_ind) {
        if (pair.second > 0) {
            sind.push_back(pair.first);
        }
    }

    // Select CCs with the longest duration
    while (!sind.empty()) {
        int ii = sind.front();
        std::vector<int> jj;
        for (int idx : ind_notempty) {
            if (idx != ii) {
                std::vector<int> diff;
                std::set_difference(
                    nCC[idx].begin(), nCC[idx].end(),
                    nCC[ii].begin(), nCC[ii].end(),
                    std::back_inserter(diff)
                );
                if (diff.empty()) {
                    jj.push_back(idx);
                }
            }
        }

        std::vector<int> iparent;
        for (int idx : ind_notempty) {
            if (nCC[idx].size() > nCC[ii].size()) {
                std::vector<int> diff;
                std::set_difference(
                    nCC[ii].begin(), nCC[ii].end(),
                    nCC[idx].begin(), nCC[idx].end(),
                    std::back_inserter(diff)
                );
                if (diff.empty() && idx != ii && std::find(jj.begin(), jj.end(), idx) == jj.end()) {
                    iparent.push_back(idx);
                }
            }
        }

        // Update nduration and nchildren
        nduration(ii, 0) = nduration(ii, 0);
        nduration(ii, 1) = nduration(ii, 1);
        nchildren[ii].clear();
        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
            nE.coeffRef(ii, it.col()) = 0.0;
        }
        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, ii); it; ++it) {
            nE.coeffRef(it.row(), ii) = 0.0;
        }
        nE.coeffRef(ii, ii) = nduration(ii, 0);

        for (int idx : jj) {
            nCC[idx].clear();
            nchildren[idx].clear();
            nparent[idx] = 0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(nE, idx); it; ++it) {
                nE.coeffRef(idx, it.col()) = 0.0;
            }
            for (Eigen::SparseMatrix<double>::InnerIterator it(nE, idx); it; ++it) {
                nE.coeffRef(it.row(), idx) = 0.0;
            }
            nduration.row(idx).setZero();
        }

        std::vector<int> to_remove;
        to_remove.insert(to_remove.end(), iparent.begin(), iparent.end());
        to_remove.insert(to_remove.end(), jj.begin(), jj.end());
        to_remove.push_back(ii);
        std::sort(to_remove.begin(), to_remove.end());
        to_remove.erase(std::unique(to_remove.begin(), to_remove.end()), to_remove.end());

        for (int idx : to_remove) {
            sind.erase(std::remove(sind.begin(), sind.end(), idx), sind.end());
        }
        ind_notempty.erase(std::remove_if(ind_notempty.begin(), ind_notempty.end(), [&](int idx) {
            return std::find(jj.begin(), jj.end(), idx) != jj.end();
        }), ind_notempty.end());
    }

    return std::make_tuple(nCC, nE, nduration, nchildren);
}