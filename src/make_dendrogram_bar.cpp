#include "make_dendrogram_bar.h"
#include <algorithm>
#include <numeric>
#include <set>
#include <limits>

std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXd,
    Eigen::MatrixXd,
    Eigen::MatrixXd,
    Eigen::MatrixXd,
    std::vector<std::vector<int>>
>
make_dendrogram_bar(
    const std::vector<std::vector<int>>& history,
    const Eigen::MatrixXd& duration,
    const Eigen::MatrixXd& cvertical_x,
    const Eigen::MatrixXd& cvertical_y,
    const Eigen::MatrixXd& chorizontal_x,
    const Eigen::MatrixXd& chorizontal_y,
    const Eigen::MatrixXd& cdots
)
{
    bool is_new = (cvertical_x.size() == 0) && (cvertical_y.size() == 0) &&
                  (chorizontal_x.size() == 0) && (chorizontal_y.size() == 0) &&
                  (cdots.size() == 0);

    int ncc = duration.rows();

    // Estimate the depth of dendrogram
    std::vector<std::vector<int>> nlayer;

    // Find CCs with no parent
    std::vector<int> length_history(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_history[i] = history[i].size();
    }

    // Identify non-empty and empty indices based on the validity of duration values
    std::vector<int> ind_notempty;
    std::vector<int> ind_empty;
    for (int i = 0; i < ncc; ++i) {
        if (!std::isnan(duration(i, 0)) && !std::isnan(duration(i, 1))) {
            ind_notempty.push_back(i);
        } else {
            ind_empty.push_back(i);
        }
    }

    // Leaf CCs
    std::vector<int> ind_past;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0 &&
            std::find(ind_empty.begin(), ind_empty.end(), i) == ind_empty.end()) {
            ind_past.push_back(i);
        }
    }
    nlayer.push_back(ind_past);

    // Build layers
    while (ind_past.size() < ind_notempty.size()) {
        std::vector<bool> tind(ncc, false);
        for (int i = 0; i < ncc; ++i) {
            std::vector<int> intersect;
            std::set_intersection(
                history[i].begin(), history[i].end(),
                ind_past.begin(), ind_past.end(),
                std::back_inserter(intersect)
            );
            tind[i] = (!history[i].empty()) && (intersect.size() == history[i].size());
        }

        std::vector<int> ttind;
        for (int i = 0; i < ncc; ++i) {
            if (tind[i] &&
                std::find(ind_past.begin(), ind_past.end(), i) == ind_past.end() &&
                std::find(ind_empty.begin(), ind_empty.end(), i) == ind_empty.end()) {
                ttind.push_back(i);
            }
        }

        if (!ttind.empty()) {
            nlayer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        } else {
            break;
        }
    }

    Eigen::MatrixXd nvertical_x;
    Eigen::MatrixXd nvertical_y;
    Eigen::MatrixXd nhorizontal_x;
    Eigen::MatrixXd nhorizontal_y;
    Eigen::MatrixXd ndots;

    if (is_new) {
        nvertical_x = Eigen::MatrixXd::Zero(ncc, 2);
        nvertical_y = Eigen::MatrixXd::Zero(ncc, 2);
        nhorizontal_x = Eigen::MatrixXd::Zero(ncc, 2);
        nhorizontal_y = Eigen::MatrixXd::Zero(ncc, 2);
        ndots = Eigen::MatrixXd::Zero(ncc, 2);

        // Sort nlayer[0] by duration
        std::vector<std::pair<int, double>> sval_ind;
        for (size_t i = 0; i < nlayer[0].size(); ++i) {
            int idx = nlayer[0][i];
            sval_ind.emplace_back(idx, duration(idx, 1));
        }

        std::sort(sval_ind.begin(), sval_ind.end(),
                  [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                      return a.second > b.second;
                  });

        std::vector<int> sind;
        for (const auto& pair : sval_ind) {
            sind.push_back(pair.first);
        }

        for (size_t i = 0; i < sind.size(); ++i) {
            int ii = sind[i];
            nvertical_x.row(ii) = Eigen::Vector2d(i, i);
            nvertical_y.row(ii) = duration.row(ii);
            ndots.row(ii) = Eigen::Vector2d(i, duration(ii, 0));
        }

        for (size_t i = 1; i < nlayer.size(); ++i) {
            for (size_t j = 0; j < nlayer[i].size(); ++j) {
                int idx = nlayer[i][j];
                std::vector<double> tx;
                for (int h : history[idx]) {
                    tx.push_back(nvertical_x(h, 0));
                }
                if (!tx.empty()) {
                    double mean_tx = std::accumulate(tx.begin(), tx.end(), 0.0) / tx.size();
                    nvertical_x.row(idx) = Eigen::Vector2d(mean_tx, mean_tx);
                    double min_tx = *std::min_element(tx.begin(), tx.end());
                    double max_tx = *std::max_element(tx.begin(), tx.end());
                    nhorizontal_x.row(idx) = Eigen::Vector2d(min_tx, max_tx);
                    ndots(idx, 0) = mean_tx;
                }
                ndots(idx, 1) = duration(idx, 0);
                nvertical_y.row(idx) = duration.row(idx);
                nhorizontal_y.row(idx) = Eigen::Vector2d(duration(idx, 0), duration(idx, 0));
            }
        }
    } else {
        nvertical_x = cvertical_x;
        nvertical_y = cvertical_y;
        nhorizontal_x = chorizontal_x;
        nhorizontal_y = chorizontal_y;
        ndots = cdots;

        // Ensure matrices have the correct sizes
        if (nvertical_x.rows() != ncc) nvertical_x.conservativeResize(ncc, 2);
        if (nvertical_y.rows() != ncc) nvertical_y.conservativeResize(ncc, 2);
        if (nhorizontal_x.rows() != ncc) nhorizontal_x.conservativeResize(ncc, 2);
        if (nhorizontal_y.rows() != ncc) nhorizontal_y.conservativeResize(ncc, 2);
        if (ndots.rows() != ncc) ndots.conservativeResize(ncc, 2);

        for (int idx : ind_empty) {
            nvertical_x.row(idx).setConstant(std::numeric_limits<double>::quiet_NaN());
            nvertical_y.row(idx).setConstant(std::numeric_limits<double>::quiet_NaN());
            nhorizontal_x.row(idx).setConstant(std::numeric_limits<double>::quiet_NaN());
            nhorizontal_y.row(idx).setConstant(std::numeric_limits<double>::quiet_NaN());
            ndots.row(idx).setConstant(std::numeric_limits<double>::quiet_NaN());
        }

        for (size_t j = 0; j < nlayer[0].size(); ++j) {
            int ii = nlayer[0][j];
            nvertical_y.row(ii) = duration.row(ii).transpose();
            nhorizontal_x.row(ii).setConstant(std::numeric_limits<double>::quiet_NaN());
            nhorizontal_y.row(ii).setConstant(std::numeric_limits<double>::quiet_NaN());
            ndots.row(ii) = Eigen::Vector2d(nvertical_x(ii, 0), nvertical_y(ii, 1));
        }

        for (size_t i = 1; i < nlayer.size(); ++i) {
            for (size_t j = 0; j < nlayer[i].size(); ++j) {
                int idx = nlayer[i][j];
                std::vector<double> tx;
                for (int h : history[idx]) {
                    tx.push_back(nvertical_x(h, 0));
                }
                if (!tx.empty()) {
                    double mean_tx = std::accumulate(tx.begin(), tx.end(), 0.0) / tx.size();
                    nvertical_x.row(idx) = Eigen::Vector2d(mean_tx, mean_tx);
                    double min_tx = *std::min_element(tx.begin(), tx.end());
                    double max_tx = *std::max_element(tx.begin(), tx.end());
                    nhorizontal_x.row(idx) = Eigen::Vector2d(min_tx, max_tx);
                    ndots(idx, 0) = mean_tx;
                }
                ndots(idx, 1) = duration(idx, 0);
                nvertical_y.row(idx) = duration.row(idx);
                nhorizontal_y.row(idx) = Eigen::Vector2d(duration(idx, 0), duration(idx, 0));
            }
        }
    }

    return std::make_tuple(nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, nlayer);
}

// Expose to Python via Pybind11
PYBIND11_MODULE(connected_components, m) {  // Module name within the STopover package
    m.def("make_dendrogram_bar", &make_dendrogram_bar, "make_dendrogram_bar",
          py::arg("history"), py::arg("duration"), py::arg("cvertical_x"),
          py::arg("cvertical_y"), py::arg("chorizontal_x"), py::arg("chorizontal_y"),
          py::arg("cdots"));
}