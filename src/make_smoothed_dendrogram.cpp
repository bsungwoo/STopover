#include "make_smoothed_dendrogram.h"

std::tuple<std::vector<std::vector<int>>, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_smoothed_dendrogram(const std::vector<std::vector<int>>& cCC,
                         Eigen::MatrixXd cE,
                         Eigen::MatrixXd cduration,
                         const std::vector<std::vector<int>>& chistory,
                         Eigen::Vector2d lim_size) {
    
    double max_size = lim_size(1);
    double min_size = lim_size(0);

    int p = 0;
    for (const auto& cc : cCC) {
        if (!cc.empty()) {
            p = std::max(p, *std::max_element(cc.begin(), cc.end()));
        }
    }

    int ncc = cCC.size();
    Eigen::VectorXd length_duration = cduration.col(0).transpose() - cduration.col(1).transpose();
    std::vector<int> length_cc(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_cc[i] = cCC[i].size();
    }

    // Layer of dendrogram
    std::vector<std::vector<int>> layer;
    std::vector<int> length_history(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_history[i] = chistory[i].size();
    }

    // Find CCs with no parent
    std::vector<int> ind_past;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0) {
            ind_past.push_back(i);
        }
    }
    layer.push_back(ind_past);

    while (ind_past.size() < ncc) {
        std::vector<int> tind;
        for (int i = 0; i < ncc; ++i) {
            if (chistory[i].size() > 0 && std::includes(ind_past.begin(), ind_past.end(), chistory[i].begin(), chistory[i].end())) {
                tind.push_back(i);
            }
        }
        std::vector<int> ttind;
        std::set_difference(tind.begin(), tind.end(), ind_past.begin(), ind_past.end(), std::back_inserter(ttind));
        if (!ttind.empty()) {
            layer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        }
    }

    // Initialization
    std::vector<std::vector<int>> nCC = cCC;
    Eigen::MatrixXd nduration = cduration;
    std::vector<std::vector<int>> nchildren = chistory;
    Eigen::MatrixXd nE = cE;
    Eigen::VectorXi nparent = -Eigen::VectorXi::Ones(ncc);
    std::vector<int> ilayer(ncc, -1);

    for (int i = 0; i < ncc; ++i) {
        if (!nchildren[i].empty()) {
            for (const auto& child : nchildren[i]) {
                nparent[child] = i;
            }
        }
        for (size_t j = 0; j < layer.size(); ++j) {
            if (std::find(layer[j].begin(), layer[j].end(), i) != layer[j].end()) {
                ilayer[i] = j;
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
                        ck[k] = length_cc[jj[k]] >= min_size ? 1 : 0;
                    }
                    if (ck.sum() <= 1) {
                        ii = nparent[ii];
                        if (ck.sum() == 1) {
                            int tind = jj[std::distance(ck.data(), std::find(ck.data(), ck.data() + ck.size(), 1))];
                            nchildren[ii] = nchildren[tind];
                            for (const auto& child : nchildren[ii]) {
                                nparent[child] = ii;
                            }
                            nduration.row(ii) << nduration.row(ii).maxCoeff(), nduration.row(tind).minCoeff();
                        } else {
                            nduration.row(ii) << nduration(jj[0], 0), nduration(jj[0], 1);
                            nchildren[ii].clear();
                        }
                        nE.row(ii).setZero();
                        nE.col(ii).setZero();
                        nE(ii, ii) = nduration(ii, 0);
                        length_duration[ii] = nduration(ii, 0) - nduration(ii, 1);
                    }
                }
                ck_delete[ii] = 1;
            }
        }
    }

    return std::make_tuple(nCC, nE, nduration, nchildren);
}