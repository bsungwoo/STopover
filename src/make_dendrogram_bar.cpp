#include "make_dendrogram_bar.h"
#include <algorithm>
#include <numeric>
#include <set>
#include <limits>
#include <cmath>      // For std::isnan
#include <iostream>   // For debugging output (optional)

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

    // Identify non-empty and empty indices based on the sum of duration rows
    std::vector<int> ind_notempty;
    std::vector<int> ind_empty;
    for (int i = 0; i < ncc; ++i) {
        double row_sum = duration.row(i).sum();
        if (row_sum != 0.0) {
            ind_notempty.push_back(i);
        }
    }
    std::sort(ind_notempty.begin(), ind_notempty.end());

    // Compute ind_empty as the set difference between all indices and ind_notempty
    std::vector<int> indices(ncc);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<int> ind_empty;
    std::set_difference(
        indices.begin(), indices.end(),
        ind_notempty.begin(), ind_notempty.end(),
        std::back_inserter(ind_empty)
    );

    // Leaf CCs
    std::vector<int> ind_past;
    std::vector<int> indices_with_length_zero;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0) {
            indices_with_length_zero.push_back(i);
        }
    }
    std::sort(indices_with_length_zero.begin(), indices_with_length_zero.end());
    std::set_difference(
        indices_with_length_zero.begin(), indices_with_length_zero.end(),
        ind_empty.begin(), ind_empty.end(),
        std::back_inserter(ind_past)
    );
    nlayer.push_back(ind_past);

    // Build layers
    while (static_cast<int>(ind_past.size()) < static_cast<int>(ind_notempty.size())) {
        std::vector<bool> tind(ncc, false);
        for (int i = 0; i < ncc; ++i) {
            if (!history[i].empty()) {
                // Sort history[i]
                std::vector<int> sorted_history_i = history[i];
                std::sort(sorted_history_i.begin(), sorted_history_i.end());

                // Perform set_intersection with sorted inputs
                std::vector<int> intersect;
                std::set_intersection(
                    sorted_history_i.begin(), sorted_history_i.end(),
                    ind_past.begin(), ind_past.end(),
                    std::back_inserter(intersect)
                );
                tind[i] = (intersect.size() == sorted_history_i.size());
            }
        }

        // Compute ttind as the set difference between where tind is true and (ind_past + ind_empty)
        std::vector<int> where_tind_true;
        for (int i = 0; i < ncc; ++i) {
            if (tind[i]) {
                where_tind_true.push_back(i);
            }
        }
        std::sort(where_tind_true.begin(), where_tind_true.end());

        std::vector<int> ind_past_and_empty = ind_past;
        ind_past_and_empty.insert(ind_past_and_empty.end(), ind_empty.begin(), ind_empty.end());
        std::sort(ind_past_and_empty.begin(), ind_past_and_empty.end());

        std::vector<int> ttind;
        std::set_difference(
            where_tind_true.begin(), where_tind_true.end(),
            ind_past_and_empty.begin(), ind_past_and_empty.end(),
            std::back_inserter(ttind)
        );

        if (!ttind.empty()) {
            nlayer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
            // Sort ind_past to ensure set operations work correctly
            std::sort(ind_past.begin(), ind_past.end());
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
        for (int idx : nlayer[0]) {
            sval_ind.emplace_back(idx, duration(idx, 1));
        }

        // Stable sort to handle ties
        std::stable_sort(sval_ind.begin(), sval_ind.end(),
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

        for (size_t layer_idx = 1; layer_idx < nlayer.size(); ++layer_idx) {
            for (int idx : nlayer[layer_idx]) {
                std::vector<double> tx;
                for (int h : history[idx]) {
                    tx.push_back(nvertical_x(h, 0));
                }
                if (!tx.empty()) {
                    // Compute mean, min, max of tx
                    double sum_tx = std::accumulate(tx.begin(), tx.end(), 0.0);
                    double mean_tx = sum_tx / tx.size();
                    double min_tx = *std::min_element(tx.begin(), tx.end());
                    double max_tx = *std::max_element(tx.begin(), tx.end());

                    nvertical_x.row(idx) = Eigen::Vector2d(mean_tx, mean_tx);
                    nhorizontal_x.row(idx) = Eigen::Vector2d(min_tx, max_tx);
                    ndots(idx, 0) = mean_tx;
                }
                ndots(idx, 1) = duration(idx, 0);
                nvertical_y.row(idx) = duration.row(idx);
                nhorizontal_y.row(idx) = Eigen::Vector2d(duration(idx, 0), duration(idx, 0));
            }
        }
    } else {
        // Handling the case when is_new is false
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

        // Set NaN for ind_empty indices
        for (int idx : ind_empty) {
            nvertical_x.row(idx).setConstant(std::numeric_limits<double>::quiet_NaN());
            nvertical_y.row(idx).setConstant(std::numeric_limits<double>::quiet_NaN());
            nhorizontal_x.row(idx).setConstant(std::numeric_limits<double>::quiet_NaN());
            nhorizontal_y.row(idx).setConstant(std::numeric_limits<double>::quiet_NaN());
            ndots.row(idx).setConstant(std::numeric_limits<double>::quiet_NaN());
        }

        // Update leaf nodes
        for (int ii : nlayer[0]) {
            nvertical_y.row(ii) = duration.row(ii);
            nhorizontal_x.row(ii).setConstant(std::numeric_limits<double>::quiet_NaN());
            nhorizontal_y.row(ii).setConstant(std::numeric_limits<double>::quiet_NaN());
            ndots.row(ii) = Eigen::Vector2d(nvertical_x(ii, 0), nvertical_y(ii, 1));
        }

        for (size_t layer_idx = 1; layer_idx < nlayer.size(); ++layer_idx) {
            for (int idx : nlayer[layer_idx]) {
                std::vector<double> tx;
                for (int h : history[idx]) {
                    tx.push_back(nvertical_x(h, 0));
                }
                if (!tx.empty()) {
                    double sum_tx = std::accumulate(tx.begin(), tx.end(), 0.0);
                    double mean_tx = sum_tx / tx.size();
                    double min_tx = *std::min_element(tx.begin(), tx.end());
                    double max_tx = *std::max_element(tx.begin(), tx.end());

                    nvertical_x.row(idx) = Eigen::Vector2d(mean_tx, mean_tx);
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