#include "make_dendrogram_bar.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <numeric>
#include <set>

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_dendrogram_bar(const std::vector<std::vector<int>>& history,
                    const Eigen::MatrixXd& duration,
                    Eigen::MatrixXd cvertical_x = Eigen::MatrixXd(),
                    Eigen::MatrixXd cvertical_y = Eigen::MatrixXd(),
                    Eigen::MatrixXd chorizontal_x = Eigen::MatrixXd(),
                    Eigen::MatrixXd chorizontal_y = Eigen::MatrixXd(),
                    Eigen::MatrixXd cdots = Eigen::MatrixXd()) {

    bool is_new = (cvertical_x.size() == 0) && (cvertical_y.size() == 0) && (chorizontal_x.size() == 0)
                  && (chorizontal_y.size() == 0) && (cdots.size() == 0);

    int ncc = duration.rows();
    std::vector<std::vector<int>> nlayer;

    // Estimate the depth of dendrogram
    std::vector<int> length_history(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_history[i] = history[i].size();
    }

    Eigen::VectorXi ind_notempty = Eigen::VectorXi::LinSpaced(ncc, 0, ncc - 1);
    ind_notempty = ind_notempty.unaryExpr([&](int i) { return (duration.row(i).sum() != 0); }).eval();

    Eigen::VectorXi ind_empty = Eigen::VectorXi::LinSpaced(ncc, 0, ncc - 1);
    ind_empty = ind_empty.unaryExpr([&](int i) { return (duration.row(i).sum() == 0); }).eval();

    std::vector<int> ind_past;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0 && std::find(ind_empty.data(), ind_empty.data() + ind_empty.size(), i) == ind_empty.data() + ind_empty.size()) {
            ind_past.push_back(i);
        }
    }
    nlayer.push_back(ind_past);

    while (ind_past.size() < ind_notempty.size()) {
        std::vector<int> tind;
        for (int i = 0; i < ncc; ++i) {
            if (std::includes(ind_past.begin(), ind_past.end(), history[i].begin(), history[i].end())) {
                tind.push_back(i);
            }
        }
        std::vector<int> ttind;
        std::set_difference(tind.begin(), tind.end(), ind_past.begin(), ind_past.end(), std::back_inserter(ttind));
        if (!ttind.empty()) {
            nlayer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        }
    }

    if (is_new) {
        // Initialize matrices for dendrogram bars
        Eigen::MatrixXd nvertical_x = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd nvertical_y = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd nhorizontal_x = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd nhorizontal_y = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd ndots = Eigen::MatrixXd::Zero(ncc, 2);

        // Sort the first layer
        std::vector<int> sorted_layer;
        std::vector<std::pair<int, double>> sval_ind;
        for (int i = 0; i < nlayer[0].size(); ++i) {
            sval_ind.push_back(std::make_pair(i, duration(nlayer[0][i], 1)));
        }
        std::sort(sval_ind.begin(), sval_ind.end(), [](auto& a, auto& b) { return a.second > b.second; });
        for (auto& p : sval_ind) {
            sorted_layer.push_back(nlayer[0][p.first]);
        }

        for (int i = 0; i < sorted_layer.size(); ++i) {
            int ii = sorted_layer[i];
            nvertical_x(ii, 0) = i;
            nvertical_x(ii, 1) = i;
            nvertical_y(ii, 0) = duration(ii, 0);
            nvertical_y(ii, 1) = duration(ii, 1);
            ndots(ii, 0) = i;
            ndots(ii, 1) = duration(ii, 0);
        }

        for (int i = 1; i < nlayer.size(); ++i) {
            for (int j = 0; j < nlayer[i].size(); ++j) {
                int ii = nlayer[i][j];
                std::vector<double> tx;
                for (int h : history[ii]) {
                    tx.push_back(nvertical_x(h, 0));
                }
                if (!tx.empty()) {
                    double mean_tx = std::accumulate(tx.begin(), tx.end(), 0.0) / tx.size();
                    nvertical_x(ii, 0) = mean_tx;
                    nvertical_x(ii, 1) = mean_tx;
                    nhorizontal_x(ii, 0) = *std::min_element(tx.begin(), tx.end());
                    nhorizontal_x(ii, 1) = *std::max_element(tx.begin(), tx.end());
                    ndots(ii, 0) = mean_tx;
                }
                ndots(ii, 1) = duration(ii, 0);
                nvertical_y(ii, 0) = duration(ii, 0);
                nvertical_y(ii, 1) = duration(ii, 1);
                nhorizontal_y(ii, 0) = duration(ii, 0);
                nhorizontal_y(ii, 1) = duration(ii, 0);
            }
        }
        return std::make_tuple(nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, nlayer);
    } else {
        // If bars are already given, modify the existing matrices
        Eigen::MatrixXd& nvertical_x = cvertical_x;
        Eigen::MatrixXd& nvertical_y = cvertical_y;
        Eigen::MatrixXd& nhorizontal_x = chorizontal_x;
        Eigen::MatrixXd& nhorizontal_y = chorizontal_y;
        Eigen::MatrixXd& ndots = cdots;

        for (int j = 0; j < nlayer[0].size(); ++j) {
            int ii = nlayer[0][j];
            nvertical_y.row(ii) = duration.row(ii).transpose();
            nhorizontal_x.row(ii).setZero();
            nhorizontal_y.row(ii).setZero();
            ndots(ii, 0) = nvertical_x(ii, 0);
            ndots(ii, 1) = nvertical_y(ii, 1);
        }

        for (int i = 1; i < nlayer.size(); ++i) {
            for (int j = 0; j < nlayer[i].size(); ++j) {
                int ii = nlayer[i][j];
                std::vector<double> tx;
                for (int h : history[ii]) {
                    tx.push_back(nvertical_x(h, 0));
                }
                if (!tx.empty()) {
                    double mean_tx = std::accumulate(tx.begin(), tx.end(), 0.0) / tx.size();
                    nvertical_x(ii, 0) = mean_tx;
                    nvertical_x(ii, 1) = mean_tx;
                    nhorizontal_x(ii, 0) = *std::min_element(tx.begin(), tx.end());
                    nhorizontal_x(ii, 1) = *std::max_element(tx.begin(), tx.end());
                    ndots(ii, 0) = mean_tx;
                }
                ndots(ii, 1) = duration(ii, 0);
                nvertical_y.row(ii) = duration.row(ii).transpose();
                nhorizontal_y.row(ii).setConstant(duration(ii, 0));
            }
        }
        return std::make_tuple(nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, nlayer);
    }
}