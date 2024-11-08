// make_dendrogram_bar.cpp
#include "make_dendrogram_bar.h"
#include "utils.h" // Include the shared utilities
#include <algorithm> // For std::find, std::min, std::max, etc.
#include <numeric>   // For std::accumulate
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <iostream>

namespace STopoverUtils {

/**
 * @brief Constructs a dendrogram bar based on provided history and duration matrices.
 *
 * This function translates the original Python implementation to C++.
 *
 * @param history Vector of connected components history.
 * @param duration Matrix containing duration information.
 * @param cvertical_x Vertical X coordinates (optional).
 * @param cvertical_y Vertical Y coordinates (optional).
 * @param chorizontal_x Horizontal X coordinates (optional).
 * @param chorizontal_y Horizontal Y coordinates (optional).
 * @param cdots Dots matrix (optional).
 * @return A tuple containing nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, and nlayer.
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_dendrogram_bar(const std::vector<std::vector<int>>& history,
                    const Eigen::MatrixXd& duration,
                    const Eigen::MatrixXd& cvertical_x,
                    const Eigen::MatrixXd& cvertical_y,
                    const Eigen::MatrixXd& chorizontal_x,
                    const Eigen::MatrixXd& chorizontal_y,
                    const Eigen::MatrixXd& cdots) {

    // Determine if this is a new dendrogram
    bool is_new = (cvertical_x.size() == 0) && (cvertical_y.size() == 0) &&
                  (chorizontal_x.size() == 0) && (chorizontal_y.size() == 0) &&
                  (cdots.size() == 0);

    size_t ncc = duration.rows();

    // Estimate the depth of dendrogram
    std::vector<std::vector<int>> nlayer;

    // Find CCs with no parent
    std::vector<int> length_history;
    length_history.reserve(history.size());
    for (const auto& cc : history) {
        length_history.push_back(static_cast<int>(cc.size()));
    }

    // Identify non-empty CCs
    std::vector<int> ind_notempty;
    for (int i = 0; i < static_cast<int>(ncc); ++i) {
        if (duration.row(i).sum() != 0) {
            ind_notempty.push_back(i);
        }
    }

    // Identify empty CCs
    std::vector<int> all_indices(ncc);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::vector<int> ind_empty;
    // Sort both vectors before set_difference
    std::vector<int> sorted_all_indices = all_indices;
    std::sort(sorted_all_indices.begin(), sorted_all_indices.end());
    std::vector<int> sorted_ind_notempty = ind_notempty;
    std::sort(sorted_ind_notempty.begin(), sorted_ind_notempty.end());
    std::set_difference(sorted_all_indices.begin(), sorted_all_indices.end(),
                        sorted_ind_notempty.begin(), sorted_ind_notempty.end(),
                        std::back_inserter(ind_empty));

    // Identify leaf CCs (no history and not empty)
    std::vector<int> ind_past;
    for (int i = 0; i < static_cast<int>(ncc); ++i) {
        if (length_history[i] == 0 &&
            std::find(ind_empty.begin(), ind_empty.end(), i) == ind_empty.end()) {
            ind_past.push_back(i);
        }
    }
    nlayer.emplace_back(ind_past);

    // Iteratively find other layers
    while (ind_past.size() < ind_notempty.size()) {
        std::vector<int> tind;
        for (int i = 0; i < static_cast<int>(ncc); ++i) {
            if (length_history[i] > 0) {
                bool subset = true;
                for (const auto& elem : history[i]) {
                    if (std::find(ind_past.begin(), ind_past.end(), elem) == ind_past.end()) {
                        subset = false;
                        break;
                    }
                }
                if (subset) {
                    tind.push_back(i);
                }
            }
        }

        // Remove already included indices and ind_empty
        std::vector<int> ttind;
        for (const auto& idx : tind) {
            if (std::find(ind_past.begin(), ind_past.end(), idx) == ind_past.end() &&
                std::find(ind_empty.begin(), ind_empty.end(), idx) == ind_empty.end()) {
                ttind.push_back(idx);
            }
        }

        if (!ttind.empty()) {
            nlayer.emplace_back(ttind);
            ind_past.insert(ind_past.end(), ttind.begin(), ttind.end());
        } else {
            break; // Prevent infinite loop in case of inconsistencies
        }
    }

    // Initialize output matrices and vectors
    Eigen::MatrixXd nvertical_x(ncc, 2);
    Eigen::MatrixXd nvertical_y(ncc, 2);
    Eigen::MatrixXd nhorizontal_x(ncc, 2);
    Eigen::MatrixXd nhorizontal_y(ncc, 2);
    Eigen::MatrixXd ndots(ncc, 2);
    std::vector<std::vector<int>> nlayer_out = nlayer; // To store layers

    if (is_new) {
        // Initialize matrices with zeros
        nvertical_x.setZero();
        nvertical_y.setZero();
        nhorizontal_x.setZero();
        nhorizontal_y.setZero();
        ndots.setZero();

        // Create a sorted list based on duration[nlayer[0],1] in descending order
        std::vector<std::pair<int, double>> sval_ind;
        for (auto idx : nlayer[0]) {
            sval_ind.emplace_back(idx, duration(idx, 1));
        }

        // Sort sval_ind in descending order based on the second element (duration value)
        std::sort(sval_ind.begin(), sval_ind.end(),
                  [&](const std::pair<int, double>& a, const std::pair<int, double>& b) -> bool {
                      return a.second > b.second;
                  });

        // Extract sorted indices
        std::vector<int> sind;
        for (const auto& pair : sval_ind) {
            sind.push_back(pair.first);
        }

        // Assign to nvertical_x, nvertical_y, ndots based on sorted indices
        for (size_t i = 0; i < sind.size(); ++i) {
            int ii = sind[i];
            nvertical_x(ii, 0) = static_cast<double>(i);
            nvertical_x(ii, 1) = static_cast<double>(i);
            nvertical_y(ii, 0) = duration(ii, 0);
            nvertical_y(ii, 1) = duration(ii, 1);
            ndots(ii, 0) = static_cast<double>(i);
            ndots(ii, 1) = duration(ii, 0);
        }

        // Process subsequent layers
        for (size_t i = 1; i < nlayer.size(); ++i) {
            for (size_t j = 0; j < nlayer[i].size(); ++j) {
                int ii = nlayer[i][j];
                const std::vector<int>& current_history = history[ii];

                // Extract nvertical_x values for the current history
                std::vector<double> tx;
                for (const auto& elem : current_history) {
                    tx.push_back(nvertical_x(elem, 0));
                }

                if (!tx.empty()) {
                    double sum_tx = std::accumulate(tx.begin(), tx.end(), 0.0);
                    double mean_tx = sum_tx / tx.size();
                    double min_tx = *std::min_element(tx.begin(), tx.end());
                    double max_tx = *std::max_element(tx.begin(), tx.end());

                    nvertical_x(ii, 0) = mean_tx;
                    nvertical_x(ii, 1) = mean_tx;
                    nhorizontal_x(ii, 0) = min_tx;
                    nhorizontal_x(ii, 1) = max_tx;
                    ndots(ii, 0) = mean_tx;
                }

                ndots(ii, 1) = duration(ii, 0);
                nvertical_y(ii, 0) = duration(ii, 0);
                nvertical_y(ii, 1) = duration(ii, 1);
                nhorizontal_y(ii, 0) = duration(ii, 0);
                nhorizontal_y(ii, 1) = duration(ii, 0);
            }
        }

    } else {
        // Use the provided matrices
        Eigen::MatrixXd nvertical_x = cvertical_x;
        Eigen::MatrixXd nvertical_y = cvertical_y;
        Eigen::MatrixXd nhorizontal_x = chorizontal_x;
        Eigen::MatrixXd nhorizontal_y = chorizontal_y;
        Eigen::MatrixXd ndots = cdots;

        // Set rows corresponding to ind_empty to zero
        for (const auto& idx : ind_empty) {
            nvertical_x.row(idx).setZero();
            nvertical_y.row(idx).setZero();
            nhorizontal_x.row(idx).setZero();
            nhorizontal_y.row(idx).setZero();
            ndots.row(idx).setZero();
        }

        // Process the first layer
        for (const auto& ii : nlayer[0]) {
            // Sort duration[ii, :] in ascending order
            // Since there are only two elements, use min and max
            double first = duration(ii, 0);
            double second = duration(ii, 1);
            double min_dur = std::min(first, second);
            double max_dur = std::max(first, second);
            nvertical_y(ii, 0) = min_dur;
            nvertical_y(ii, 1) = max_dur;

            nhorizontal_x.row(ii).setZero();
            nhorizontal_y.row(ii).setZero();
            ndots(ii, 0) = nvertical_x(ii, 0);
            ndots(ii, 1) = nvertical_y(ii, 1);
        }

        // Process subsequent layers
        for (size_t i = 1; i < nlayer.size(); ++i) {
            for (const auto& ii : nlayer[i]) {
                const std::vector<int>& current_history = history[ii];

                // Extract nvertical_x values for the current history
                std::vector<double> tx;
                for (const auto& elem : current_history) {
                    tx.push_back(nvertical_x(elem, 0));
                }

                if (!tx.empty()) {
                    double sum_tx = std::accumulate(tx.begin(), tx.end(), 0.0);
                    double mean_tx = sum_tx / tx.size();
                    double min_tx = *std::min_element(tx.begin(), tx.end());
                    double max_tx = *std::max_element(tx.begin(), tx.end());

                    nvertical_x(ii, 0) = mean_tx;
                    nvertical_x(ii, 1) = mean_tx;
                    nhorizontal_x(ii, 0) = min_tx;
                    nhorizontal_x(ii, 1) = max_tx;
                    ndots(ii, 0) = mean_tx;
                }

                ndots(ii, 1) = duration(ii, 0);
                nvertical_y(ii, 0) = duration(ii, 0);
                nvertical_y(ii, 1) = duration(ii, 1);
                nhorizontal_y(ii, 0) = duration(ii, 0);
                nhorizontal_y(ii, 1) = duration(ii, 0);
            }
        }
    }

    // Return the constructed matrices and layers
    return std::make_tuple(nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, nlayer_out);
}

} // namespace STopoverUtils