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

    // Identify non-empty and empty indices based on the validity of duration values
    std::vector<int> ind_notempty;
    std::vector<int> ind_empty;
    const double EPSILON = 0; // Tolerance for floating-point comparison
    for (int i = 0; i < ncc; ++i) {
        double row_sum = duration.row(i).sum();
        if (std::abs(row_sum) > EPSILON) {
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
    while (static_cast<int>(ind_past.size()) < static_cast<int>(ind_notempty.size())) {
        std::vector<bool> tind(ncc, false);
        for (int i = 0; i < ncc; ++i) {
            if (!history[i].empty()) {
                std::vector<int> intersect;
                std::set_intersection(
                    history[i].begin(), history[i].end(),
                    ind_past.begin(), ind_past.end(),
                    std::back_inserter(intersect)
                );
                tind[i] = (intersect.size() == history[i].size());
            }
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
            // Sort ind_past to ensure set_intersection works correctly
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
        // Use existing matrices
        nvertical_x = cvertical_x;
        nvertical_y = cvertical_y;
        nhorizontal_x = chorizontal_x;
        nhorizontal_y = chorizontal_y;
        ndots = cdots;

        // Remove resizing if input matrices are guaranteed to be (ncc, 2)
        // If not, ensure that input matrices are correctly sized before passing to this function

        // Set ind_empty rows to zero
        for (int idx : ind_empty) {
            nvertical_x.row(idx).setZero();
            nvertical_y.row(idx).setZero();
            nhorizontal_x.row(idx).setZero();
            nhorizontal_y.row(idx).setZero();
            ndots.row(idx).setZero();
        }

        // Update leaf nodes
        for (int ii : nlayer[0]) {
            // Sort duration.row(ii)
            Eigen::RowVector2d sorted_duration = duration.row(ii);
            if (sorted_duration(0) > sorted_duration(1)) {
                std::swap(sorted_duration(0), sorted_duration(1));
            }
            nvertical_y.row(ii) = sorted_duration;

            // Set nhorizontal_x and nhorizontal_y to zero
            nhorizontal_x.row(ii).setZero();
            nhorizontal_y.row(ii).setZero();

            // Assign ndots
            ndots.row(ii) = Eigen::RowVector2d(nvertical_x(ii, 0), nvertical_y(ii, 1));
        }

        // Process subsequent layers
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

                    nvertical_x.row(idx) = Eigen::RowVector2d(mean_tx, mean_tx);
                    nhorizontal_x.row(idx) = Eigen::RowVector2d(min_tx, max_tx);
                    ndots(idx, 0) = mean_tx;
                }
                ndots(idx, 1) = duration(idx, 0);
                nvertical_y.row(idx) = duration.row(idx);
                nhorizontal_y.row(idx) = Eigen::RowVector2d(duration(idx, 0), duration(idx, 0));
            }
        }
    }

    return std::make_tuple(nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, nlayer);
}