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
    std::vector<std::vector<int>> nlayer;

    // Compute length_history
    std::vector<int> length_history(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_history[i] = history[i].size();
    }

    // Identify ind_notempty and ind_empty
    std::vector<int> ind_notempty;
    std::vector<int> ind_empty;

    for (int i = 0; i < ncc; ++i) {
        double row_sum = duration.row(i).sum();
        if (std::abs(row_sum) > 1e-12) {
            ind_notempty.push_back(i);
        } else {
            ind_empty.push_back(i);
        }
    }

    // ind_past: Nodes with no parents and not in ind_empty
    std::vector<int> indices(ncc);
    std::iota(indices.begin(), indices.end(), 0); // indices = [0, 1, ..., ncc-1]

    std::vector<int> indices_with_length_zero;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0) {
            indices_with_length_zero.push_back(i);
        }
    }

    // ind_past = set difference between indices_with_length_zero and ind_empty
    std::vector<int> ind_past;
    std::sort(indices_with_length_zero.begin(), indices_with_length_zero.end());
    std::sort(ind_empty.begin(), ind_empty.end());
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
            const auto& h = history[i];
            if (!h.empty()) {
                // Compute intersection between history[i] and ind_past
                std::vector<int> sorted_h = h;
                std::sort(sorted_h.begin(), sorted_h.end());
                std::vector<int> intersect;
                std::set_intersection(
                    sorted_h.begin(), sorted_h.end(),
                    ind_past.begin(), ind_past.end(),
                    std::back_inserter(intersect)
                );
                tind[i] = (intersect.size() == sorted_h.size());
            }
        }

        // Get indices where tind is true
        std::vector<int> where_tind_true;
        for (int i = 0; i < ncc; ++i) {
            if (tind[i]) {
                where_tind_true.push_back(i);
            }
        }

        // ttind = set difference between where_tind_true and (ind_past + ind_empty)
        std::vector<int> ind_past_and_empty = ind_past;
        ind_past_and_empty.insert(ind_past_and_empty.end(), ind_empty.begin(), ind_empty.end());
        std::sort(ind_past_and_empty.begin(), ind_past_and_empty.end());
        std::vector<int> ttind;
        std::sort(where_tind_true.begin(), where_tind_true.end());
        std::set_difference(
            where_tind_true.begin(), where_tind_true.end(),
            ind_past_and_empty.begin(), ind_past_and_empty.end(),
            std::back_inserter(ttind)
        );

        if (!ttind.empty()) {
            nlayer.push_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
            // Sort ind_past for set operations
            std::sort(ind_past.begin(), ind_past.end());
        } else {
            break;
        }
    }

    if (is_new) {
        // Initialize matrices
        Eigen::MatrixXd nvertical_x = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd nvertical_y = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd nhorizontal_x = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd nhorizontal_y = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd ndots = Eigen::MatrixXd::Zero(ncc, 2);

        // Sort nlayer[0] by duration descending
        std::vector<int> nlayer0 = nlayer[0];
        std::vector<std::pair<int, double>> duration_pairs;
        for (int idx : nlayer0) {
            duration_pairs.emplace_back(idx, duration(idx, 1));
        }

        std::sort(duration_pairs.begin(), duration_pairs.end(),
                  [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                      return a.second > b.second;
                  });

        std::vector<int> sind;
        for (const auto& p : duration_pairs) {
            sind.push_back(p.first);
        }

        // Assign positions to leaf nodes
        for (size_t i = 0; i < sind.size(); ++i) {
            int ii = sind[i];
            nvertical_x.row(ii) = Eigen::RowVector2d(i, i);
            nvertical_y.row(ii) = duration.row(ii);
            ndots.row(ii) = Eigen::RowVector2d(i, duration(ii, 0));
        }

        // Process subsequent layers
        for (size_t i = 1; i < nlayer.size(); ++i) {
            for (int idx : nlayer[i]) {
                const auto& history_idx = history[idx];
                std::vector<double> tx;
                for (int h : history_idx) {
                    tx.push_back(nvertical_x(h, 0));
                }
                if (!tx.empty()) {
                    double mean_tx = std::accumulate(tx.begin(), tx.end(), 0.0) / tx.size();
                    nvertical_x.row(idx) = Eigen::RowVector2d(mean_tx, mean_tx);
                    nhorizontal_x.row(idx) = Eigen::RowVector2d(*std::min_element(tx.begin(), tx.end()),
                                                                *std::max_element(tx.begin(), tx.end()));
                    ndots(idx, 0) = mean_tx;
                }
                ndots(idx, 1) = duration(idx, 0);
                nvertical_y.row(idx) = duration.row(idx);
                nhorizontal_y.row(idx) = Eigen::RowVector2d(duration(idx, 0), duration(idx, 0));
            }
        }

        // Return results
        return std::make_tuple(nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, nlayer);
    } else {
        // Use existing matrices
        Eigen::MatrixXd nvertical_x = cvertical_x;
        Eigen::MatrixXd nvertical_y = cvertical_y;
        Eigen::MatrixXd nhorizontal_x = chorizontal_x;
        Eigen::MatrixXd nhorizontal_y = chorizontal_y;
        Eigen::MatrixXd ndots = cdots;

        // Set ind_empty rows to zero
        for (int idx : ind_empty) {
            nvertical_x.row(idx).setZero();
            nvertical_y.row(idx).setZero();
            nhorizontal_x.row(idx).setZero();
            nhorizontal_y.row(idx).setZero();
            ndots.row(idx).setZero();
        }

        // Update leaf nodes
        for (int idx : nlayer[0]) {
            Eigen::RowVector2d sorted_duration = duration.row(idx);
            if (sorted_duration(0) > sorted_duration(1)) {
                std::swap(sorted_duration(0), sorted_duration(1));
            }
            nvertical_y.row(idx) = sorted_duration;
            nhorizontal_x.row(idx).setZero();
            nhorizontal_y.row(idx).setZero();
            ndots.row(idx) = Eigen::RowVector2d(nvertical_x(idx, 0), nvertical_y(idx, 1));
        }

        // Process subsequent layers
        for (size_t i = 1; i < nlayer.size(); ++i) {
            for (int idx : nlayer[i]) {
                const auto& history_idx = history[idx];
                std::vector<double> tx;
                for (int h : history_idx) {
                    tx.push_back(nvertical_x(h, 0));
                }
                if (!tx.empty()) {
                    double mean_tx = std::accumulate(tx.begin(), tx.end(), 0.0) / tx.size();
                    nvertical_x.row(idx) = Eigen::RowVector2d(mean_tx, mean_tx);
                    nhorizontal_x.row(idx) = Eigen::RowVector2d(*std::min_element(tx.begin(), tx.end()),
                                                                *std::max_element(tx.begin(), tx.end()));
                    ndots(idx, 0) = mean_tx;
                }
                ndots(idx, 1) = duration(idx, 0);
                nvertical_y.row(idx) = duration.row(idx);
                nhorizontal_y.row(idx) = Eigen::RowVector2d(duration(idx, 0), duration(idx, 0));
            }
        }

        // Return results
        return std::make_tuple(nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, nlayer);
    }
}