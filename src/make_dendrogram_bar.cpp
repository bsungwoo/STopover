#include "make_dendrogram_bar.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <set>

// Helper function to check if all elements of subset are in superset
bool is_subset(const std::vector<int>& subset, const std::vector<int>& superset) {
    for (const auto& elem : subset) {
        if (std::find(superset.begin(), superset.end(), elem) == superset.end()) {
            return false;
        }
    }
    return true;
}

// Helper function to compute intersection size
size_t intersection_size(const std::vector<int>& a, const std::vector<int>& b) {
    std::set<int> set_a(a.begin(), a.end());
    std::set<int> set_b(b.begin(), b.end());
    std::vector<int> intersection;
    std::set_intersection(set_a.begin(), set_a.end(),
                          set_b.begin(), set_b.end(),
                          std::back_inserter(intersection));
    return intersection.size();
}

std::tuple<
    Eigen::MatrixXd,    // nvertical_x
    Eigen::MatrixXd,    // nvertical_y
    Eigen::MatrixXd,    // nhorizontal_x
    Eigen::MatrixXd,    // nhorizontal_y
    Eigen::MatrixXd,    // ndots
    std::vector<std::vector<int>> // nlayer
>
make_dendrogram_bar(
    const std::vector<std::vector<int>>& history,
    const Eigen::MatrixXd& duration,
    const Eigen::MatrixXd& cvertical_x,
    const Eigen::MatrixXd& cvertical_y,
    const Eigen::MatrixXd& chorizontal_x,
    const Eigen::MatrixXd& chorizontal_y,
    const Eigen::MatrixXd& cdots
) {
    // Determine if this is a new dendrogram or updating existing one
    bool is_new = (cvertical_x.size() == 0) && (cvertical_y.size() == 0) &&
                  (chorizontal_x.size() == 0) && (chorizontal_y.size() == 0) &&
                  (cdots.size() == 0);

    int ncc = duration.rows();

    // Estimate the depth of dendrogram
    std::vector<std::vector<int>> nlayer;

    // Find CCs with no parent (history size == 0)
    std::vector<int> ind_past;
    for (int i = 0; i < ncc; ++i) {
        if (history[i].empty()) {
            ind_past.push_back(i);
        }
    }
    nlayer.emplace_back(ind_past);

    // Iteratively find other layers
    while (ind_past.size() < ncc) {
        std::vector<int> tind;
        for (int i = 0; i < ncc; ++i) {
            if (!history[i].empty() && is_subset(history[i], ind_past)) {
                tind.push_back(i);
            }
        }

        // Remove already included indices
        std::vector<int> ttind;
        for (const auto& idx : tind) {
            if (std::find(ind_past.begin(), ind_past.end(), idx) == ind_past.end()) {
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

    if (is_new) {
        // Initialize matrices for dendrogram bars
        Eigen::MatrixXd nvertical_x = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd nvertical_y = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd nhorizontal_x = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd nhorizontal_y = Eigen::MatrixXd::Zero(ncc, 2);
        Eigen::MatrixXd ndots = Eigen::MatrixXd::Zero(ncc, 2);

        // Sort the first layer based on duration[:,1] in descending order
        std::vector<std::pair<int, double>> sval_ind;
        for (size_t i = 0; i < nlayer[0].size(); ++i) {
            int idx = nlayer[0][i];
            sval_ind.emplace_back(idx, duration(idx, 1));
        }

        // Sort descending based on duration's death time
        std::sort(sval_ind.begin(), sval_ind.end(),
                  [](const std::pair<int, double>& a, const std::pair<int, double>& b) -> bool {
                      return a.second > b.second;
                  });

        // Extract sorted indices
        std::vector<int> sorted_layer;
        for (const auto& pair : sval_ind) {
            sorted_layer.push_back(pair.first);
        }

        // Assign positions for the first layer
        for (size_t i = 0; i < sorted_layer.size(); ++i) {
            int ii = sorted_layer[i];
            nvertical_x(ii, 0) = static_cast<double>(i);
            nvertical_x(ii, 1) = static_cast<double>(i);
            nvertical_y(ii, 0) = duration(ii, 0);
            nvertical_y(ii, 1) = duration(ii, 1);
            ndots(ii, 0) = static_cast<double>(i);
            ndots(ii, 1) = duration(ii, 0);
        }

        // Iterate over the remaining layers
        for (size_t i = 1; i < nlayer.size(); ++i) {
            for (size_t j = 0; j < nlayer[i].size(); ++j) {
                int ii = nlayer[i][j];
                std::vector<double> tx;

                // Collect the x-coordinates of the children
                for (const auto& child : history[ii]) {
                    tx.push_back(nvertical_x(child, 0));
                }

                if (!tx.empty()) {
                    // Compute mean, min, and max
                    double mean_tx = std::accumulate(tx.begin(), tx.end(), 0.0) / tx.size();
                    double min_tx = *std::min_element(tx.begin(), tx.end());
                    double max_tx = *std::max_element(tx.begin(), tx.end());

                    // Assign vertical x-coordinates
                    nvertical_x(ii, 0) = mean_tx;
                    nvertical_x(ii, 1) = mean_tx;

                    // Assign horizontal x-coordinates
                    nhorizontal_x(ii, 0) = min_tx;
                    nhorizontal_x(ii, 1) = max_tx;

                    // Assign dot x-coordinate
                    ndots(ii, 0) = mean_tx;
                }

                // Assign dot y-coordinate and vertical y-coordinates
                ndots(ii, 1) = duration(ii, 0);
                nvertical_y(ii, 0) = duration(ii, 0);
                nvertical_y(ii, 1) = duration(ii, 1);

                // Assign horizontal y-coordinates (constant at duration[ii,0])
                nhorizontal_y(ii, 0) = duration(ii, 0);
                nhorizontal_y(ii, 1) = duration(ii, 0);
            }
        }

        return std::make_tuple(nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, nlayer);
    } else {
        // Update existing matrices
        // Initialize copies of existing matrices
        Eigen::MatrixXd nvertical_x = cvertical_x;
        Eigen::MatrixXd nvertical_y = cvertical_y;
        Eigen::MatrixXd nhorizontal_x = chorizontal_x;
        Eigen::MatrixXd nhorizontal_y = chorizontal_y;
        Eigen::MatrixXd ndots = cdots;
        std::vector<std::vector<int>> nlayer = nlayer; // Updated layers

        // Iterate over the layers to update positions
        for (size_t i = 0; i < nlayer.size(); ++i) {
            for (size_t j = 0; j < nlayer[i].size(); ++j) {
                int ii = nlayer[i][j];
                if (i == 0) {
                    // First layer: sort based on duration
                    nvertical_y(ii, 0) = std::min(duration(ii, 0), duration(ii, 1));
                    nvertical_y(ii, 1) = std::max(duration(ii, 0), duration(ii, 1));
                    nhorizontal_x(ii, 0) = 0.0;
                    nhorizontal_x(ii, 1) = 0.0;
                    nhorizontal_y(ii, 0) = 0.0;
                    nhorizontal_y(ii, 1) = 0.0;
                    ndots(ii, 0) = nvertical_x(ii, 0);
                    ndots(ii, 1) = nvertical_y(ii, 1);
                } else {
                    // Subsequent layers
                    std::vector<double> tx;
                    for (const auto& child : history[ii]) {
                        tx.push_back(nvertical_x(child, 0));
                    }

                    if (!tx.empty()) {
                        double mean_tx = std::accumulate(tx.begin(), tx.end(), 0.0) / tx.size();
                        double min_tx = *std::min_element(tx.begin(), tx.end());
                        double max_tx = *std::max_element(tx.begin(), tx.end());

                        // Assign vertical x-coordinates
                        nvertical_x(ii, 0) = mean_tx;
                        nvertical_x(ii, 1) = mean_tx;

                        // Assign horizontal x-coordinates
                        nhorizontal_x(ii, 0) = min_tx;
                        nhorizontal_x(ii, 1) = max_tx;

                        // Assign dot x-coordinate
                        ndots(ii, 0) = mean_tx;
                    }

                    // Assign dot y-coordinate and vertical y-coordinates
                    ndots(ii, 1) = duration(ii, 0);
                    nvertical_y(ii, 0) = duration(ii, 0);
                    nvertical_y(ii, 1) = duration(ii, 1);

                    // Assign horizontal y-coordinates (constant at duration[ii,0])
                    nhorizontal_y(ii, 0) = duration(ii, 0);
                    nhorizontal_y(ii, 1) = duration(ii, 0);
                }
            }
        }

        return std::make_tuple(nvertical_x, nvertical_y, nhorizontal_x, nhorizontal_y, ndots, nlayer);
    }
}