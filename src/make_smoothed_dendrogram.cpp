#include "make_smoothed_dendrogram.h"
#include <algorithm>
#include <numeric>
#include <set>
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

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
            std::vector<int> intersect;
            std::set_intersection(
                chistory[i].begin(), chistory[i].end(),
                ind_past.begin(), ind_past.end(),
                std::back_inserter(intersect)
            );
            if (chistory[i].size() > 0 && intersect.size() == chistory[i].size()) {
                if (std::find(ind_past.begin(), ind_past.end(), i) == ind_past.end()) {
                    ttind.push_back(i);
                }
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
                    // Find siblings
                    std::vector<int> jj = nchildren[nparent[ii]];
                    std::vector<int> ck(jj.size());
                    for (size_t k = 0; k < jj.size(); ++k) {
                        ck[k] = (length_cc[jj[k]] >= min_size) ? 1 : 0;
                    }

                    int sum_ck = std::accumulate(ck.begin(), ck.end(), 0);
                    if (sum_ck <= 1) {
                        // Merge back into parent
                        int parent_idx = nparent[ii];
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
                                nchildren[parent_idx] = nchildren[tind];
                                for (int child : nchildren[parent_idx]) {
                                    nparent[child] = parent_idx;
                                }
                                nduration.row(parent_idx) = nduration.row(parent_idx).cwiseMax(nduration.row(tind));
                            }
                        } else {
                            // No siblings to keep
                            nduration.row(parent_idx) = nduration.row(parent_idx).cwiseMax(nduration.row(ii));
                            nchildren[parent_idx].clear();
                        }

                        // Update nE
                        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, parent_idx); it; ++it) {
                            nE.coeffRef(parent_idx, it.col()) = 0.0;
                        }
                        for (Eigen::SparseMatrix<double>::InnerIterator it(nE, parent_idx); it; ++it) {
                            nE.coeffRef(it.row(), parent_idx) = 0.0;
                        }
                        nE.coeffRef(parent_idx, parent_idx) = nduration(parent_idx, 0);

                        // Mark siblings for deletion
                        for (size_t k = 0; k < jj.size(); ++k) {
                            int idx = jj[k];
                            ck_delete[idx] = 1;
                            nCC[idx].clear();
                            nchildren[idx].clear();
                            nparent[idx] = 0;
                            nduration.row(idx).setZero();
                            length_cc[idx] = 0;
                            // Remove from layer
                            int l_idx = ilayer[idx];
                            std::replace(layer[l_idx].begin(), layer[l_idx].end(), idx, 0);
                        }
                    } else {
                        // Delete current component
                        ck_delete[ii] = 1;
                        nchildren[nparent[ii]].erase(std::remove(nchildren[nparent[ii]].begin(), nchildren[nparent[ii]].end(), ii), nchildren[nparent[ii]].end());
                        nCC[ii].clear();
                        nchildren[ii].clear();
                        nparent[ii] = 0;
                        nduration.row(ii).setZero();
                        length_cc[ii] = 0;
                        // Remove from layer
                        int l_idx = ilayer[ii];
                        std::replace(layer[l_idx].begin(), layer[l_idx].end(), ii, 0);
                    }
                } else {
                    // No parent, delete component
                    ck_delete[ii] = 1;
                    nCC[ii].clear();
                    nchildren[ii].clear();
                    nparent[ii] = 0;
                    nduration.row(ii).setZero();
                    length_cc[ii] = 0;
                    // Remove from layer
                    int l_idx = ilayer[ii];
                    std::replace(layer[l_idx].begin(), layer[l_idx].end(), ii, 0);
                }
            }
        }
    }

    // Rebuild layers
    layer.clear();
    ind_past.clear();
    for (int i = 0; i < ncc; ++i) {
        if (nchildren[i].empty() && !nCC[i].empty()) {
            ind_past.push_back(i);
        }
    }
    layer.push_back(ind_past);

    while (ind_past.size() < static_cast<size_t>(ncc)) {
        std::vector<int> ttind;
        for (int i = 0; i < ncc; ++i) {
            if (!nchildren[i].empty()) {
                std::vector<int> intersect;
                std::set_intersection(
                    nchildren[i].begin(), nchildren[i].end(),
                    ind_past.begin(), ind_past.end(),
                    std::back_inserter(intersect)
                );
                if (intersect.size() == nchildren[i].size()) {
                    if (std::find(ind_past.begin(), ind_past.end(), i) == ind_past.end()) {
                        ttind.push_back(i);
                    }
                }
            }
        }
        if (!ttind.empty()) {
            layer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        } else {
            break;
        }
    }

    // Return the results
    return std::make_tuple(nCC, nE, nduration, nchildren);
}

// Expose to Python via Pybind11
PYBIND11_MODULE(connected_components, m) {  // Module name within the STopover package
    m.def("make_smoothed_dendrogram", &make_smoothed_dendrogram, "make_smoothed_dendrogram",
          py::arg("cCC"), py::arg("cE"), py::arg("cduration"), py::arg("chistory"), 
          py::arg("lim_size");
}