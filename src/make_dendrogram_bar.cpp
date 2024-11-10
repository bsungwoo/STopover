#include "make_dendrogram_bar.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <tuple>
#include <set>

using namespace std;

/**
 * @brief Constructs a dendrogram bar based on provided history and duration matrices.
 *
 * @param history Vector of connected components history.
 * @param duration Matrix containing duration information.
 * @param cvertical_x Optional precomputed vertical X coordinates.
 * @param cvertical_y Optional precomputed vertical Y coordinates.
 * @param chorizontal_x Optional precomputed horizontal X coordinates.
 * @param chorizontal_y Optional precomputed horizontal Y coordinates.
 * @param cdots Optional precomputed dots coordinates.
 * @return A tuple containing nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, and nlayer.
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_dendrogram_bar(const std::vector<std::vector<int>>& history,
                    const Eigen::MatrixXd& duration,
                    const Eigen::MatrixXd& cvertical_x = Eigen::MatrixXd(),
                    const Eigen::MatrixXd& cvertical_y = Eigen::MatrixXd(),
                    const Eigen::MatrixXd& chorizontal_x = Eigen::MatrixXd(),
                    const Eigen::MatrixXd& chorizontal_y = Eigen::MatrixXd(),
                    const Eigen::MatrixXd& cdots = Eigen::MatrixXd()) {
    bool is_new = (cvertical_x.size() == 0) && (cvertical_y.size() == 0) &&
                  (chorizontal_x.size() == 0) && (chorizontal_y.size() == 0) &&
                  (cdots.size() == 0);

    int ncc = duration.rows();

    // Estimate the depth of dendrogram
    std::vector<std::vector<int>> nlayer;

    // Find CCs with no parent
    std::vector<int> length_history(ncc);
    for (int i = 0; i < ncc; ++i) {
        length_history[i] = static_cast<int>(history[i].size());
    }

    // Identify non-empty CCs
    std::vector<int> ind_notempty;
    for (int i = 0; i < ncc; ++i) {
        if (duration.row(i).sum() != 0) {
            ind_notempty.push_back(i);
        }
    }

    // Identify empty CCs
    std::vector<int> all_indices(ncc);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::vector<int> ind_empty;
    std::set_difference(all_indices.begin(), all_indices.end(),
                        ind_notempty.begin(), ind_notempty.end(),
                        std::back_inserter(ind_empty));

    // Identify leaf CCs
    std::vector<int> ind_past;
    for (int i = 0; i < ncc; ++i) {
        if (length_history[i] == 0 && std::find(ind_empty.begin(), ind_empty.end(), i) == ind_empty.end()) {
            ind_past.push_back(i);
        }
    }
    nlayer.push_back(ind_past);

    // Build the dendrogram layers
    while (static_cast<int>(ind_past.size()) < static_cast<int>(ind_notempty.size())) {
        std::vector<int> tind;
        for (int i = 0; i < ncc; ++i) {
            if (!history[i].empty()) {
                bool is_subset = true;
                for (const auto& h_elem : history[i]) {
                    if (std::find(ind_past.begin(), ind_past.end(), h_elem) == ind_past.end()) {
                        is_subset = false;
                        break;
                    }
                }
                if (is_subset) {
                    tind.push_back(i);
                }
            }
        }

        // Remove already included indices and empty indices
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
            break; // Prevent infinite loop
        }
    }

    // Initialize matrices
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

        // Sort the first layer based on duration
        std::vector<std::pair<int, double>> sval_ind;
        for (int idx : nlayer[0]) {
            sval_ind.emplace_back(idx, duration(idx, 1));
        }

        // Sort in descending order
        std::sort(sval_ind.begin(), sval_ind.end(),
                  [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                      return a.second > b.second;
                  });

        // Assign positions in the dendrogram
        for (size_t i = 0; i < sval_ind.size(); ++i) {
            int ii = sval_ind[i].first;
            nvertical_x(ii, 0) = static_cast<double>(i);
            nvertical_x(ii, 1) = static_cast<double>(i);
            nvertical_y(ii, 0) = duration(ii, 0);
            nvertical_y(ii, 1) = duration(ii, 1);
            ndots(ii, 0) = static_cast<double>(i);
            ndots(ii, 1) = duration(ii, 0);
        }

        // Process subsequent layers
        for (size_t i = 1; i < nlayer.size(); ++i) {
            for (int idx : nlayer[i]) {
                std::vector<double> tx;
                for (int h_idx : history[idx]) {
                    tx.push_back(nvertical_x(h_idx, 0));
                }
                if (!tx.empty()) {
                    double mean_tx = std::accumulate(tx.begin(), tx.end(), 0.0) / tx.size();
                    double min_tx = *std::min_element(tx.begin(), tx.end());
                    double max_tx = *std::max_element(tx.begin(), tx.end());

                    nvertical_x(idx, 0) = mean_tx;
                    nvertical_x(idx, 1) = mean_tx;
                    nhorizontal_x(idx, 0) = min_tx;
                    nhorizontal_x(idx, 1) = max_tx;
                    ndots(idx, 0) = mean_tx;
                }
                ndots(idx, 1) = duration(idx, 0);
                nvertical_y(idx, 0) = duration(idx, 0);
                nvertical_y(idx, 1) = duration(idx, 1);
                nhorizontal_y(idx, 0) = duration(idx, 0);
                nhorizontal_y(idx, 1) = duration(idx, 0);
            }
        }
    } else {
        // Use provided matrices
        nvertical_x = cvertical_x;
        nvertical_y = cvertical_y;
        nhorizontal_x = chorizontal_x;
        nhorizontal_y = chorizontal_y;
        ndots = cdots;

        // Set rows corresponding to ind_empty to zero
        for (int idx : ind_empty) {
            nvertical_x.row(idx).setZero();
            nvertical_y.row(idx).setZero();
            nhorizontal_x.row(idx).setZero();
            nhorizontal_y.row(idx).setZero();
            ndots.row(idx).setZero();
        }

        // Process the first layer
        for (int idx : nlayer[0]) {
            double min_dur = std::min(duration(idx, 0), duration(idx, 1));
            double max_dur = std::max(duration(idx, 0), duration(idx, 1));
            nvertical_y(idx, 0) = min_dur;
            nvertical_y(idx, 1) = max_dur;
            nhorizontal_x.row(idx).setZero();
            nhorizontal_y.row(idx).setZero();
            ndots(idx, 0) = nvertical_x(idx, 0);
            ndots(idx, 1) = nvertical_y(idx, 1);
        }

        // Process subsequent layers
        for (size_t i = 1; i < nlayer.size(); ++i) {
            for (int idx : nlayer[i]) {
                std::vector<double> tx;
                for (int h_idx : history[idx]) {
                    tx.push_back(nvertical_x(h_idx, 0));
                }
                if (!tx.empty()) {
                    double mean_tx = std::accumulate(tx.begin(), tx.end(), 0.0) / tx.size();
                    double min_tx = *std::min_element(tx.begin(), tx.end());
                    double max_tx = *std::max_element(tx.begin(), tx.end());

                    nvertical_x(idx, 0) = mean_tx;
                    nvertical_x(idx, 1) = mean_tx;
                    nhorizontal_x(idx, 0) = min_tx;
                    nhorizontal_x(idx, 1) = max_tx;
                    ndots(idx, 0) = mean_tx;
                }
                ndots(idx, 1) = duration(idx, 0);
                nvertical_y(idx, 0) = duration(idx, 0);
                nvertical_y(idx, 1) = duration(idx, 1);
                nhorizontal_y(idx, 0) = duration(idx, 0);
                nhorizontal_y(idx, 1) = duration(idx, 0);
            }
        }
    }

    return std::make_tuple(nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, nlayer);
}