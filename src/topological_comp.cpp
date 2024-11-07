#include "topological_comp.h"
#include <limits>    // for std::numeric_limits

// Function to compute adjacency matrix and Gaussian smoothing mask based on spatial locations
std::tuple<Eigen::SparseMatrix<int>, Eigen::MatrixXd> extract_adjacency_spatial(const Eigen::MatrixXd& loc, const std::string& spatial_type, double fwhm) {
    int p = loc.rows();
    Eigen::MatrixXd A(p, p);
    Eigen::MatrixXd arr_mod(p, p);
    double sigma = fwhm / 2.355;

    if (spatial_type == "visium") {
        // Calculate pairwise distances using Euclidean norm
        for (int i = 0; i < p; ++i) {
            for (int j = i + 1; j < p; ++j) {
                double dist = (loc.row(i) - loc.row(j)).norm();
                A(i, j) = A(j, i) = dist;
            }
        }

        // Replace distances exceeding fwhm with infinity
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < p; ++j) {
                if (A(i, j) > fwhm) {
                    A(i, j) = std::numeric_limits<double>::infinity();
                }
            }
        }

        // Gaussian smoothing
        arr_mod = (1 / (2 * M_PI * sigma * sigma)) * (-A.array().square() / (2 * sigma * sigma)).exp();

        // Calculate minimum distance where A > 0
        double min_distance = std::numeric_limits<double>::infinity();
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < p; ++j) {
                if (A(i, j) > 0 && A(i, j) < min_distance) {
                    min_distance = A(i, j);
                }
            }
        }

        // Create adjacency matrix based on minimum distances
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < p; ++j) {
                A(i, j) = (A(i, j) > 0 && A(i, j) <= min_distance) ? 1 : 0;
            }
        }

        Eigen::SparseMatrix<int> A_sparse = A.cast<int>().sparseView();
        return std::make_tuple(A_sparse, arr_mod);
    } else if (spatial_type == "imageST") {
        int rows = static_cast<int>(loc.col(1).maxCoeff()) + 1;
        int cols = static_cast<int>(loc.col(0).maxCoeff()) + 1;
        Eigen::SparseMatrix<int> adjacency(rows * cols, rows * cols);

        // Logic for constructing adjacency matrix for imageST
        for (int i = 0; i < loc.rows(); ++i) {
            int x = static_cast<int>(loc(i, 0));
            int y = static_cast<int>(loc(i, 1));

            if (x - 1 >= 0) {
                int neighbor1 = (x - 1) * cols + y;
                int current = x * cols + y;
                adjacency.insert(current, neighbor1) = 1;
                adjacency.insert(neighbor1, current) = 1;
            }
            if (y - 1 >= 0) {
                int neighbor2 = x * cols + (y - 1);
                int current = x * cols + y;
                adjacency.insert(current, neighbor2) = 1;
                adjacency.insert(neighbor2, current) = 1;
            }
        }

        // Subset the adjacency matrix to include only valid rows/cols
        std::vector<int> valid_indices;
        for (int i = 0; i < loc.rows(); ++i) {
            valid_indices.push_back(static_cast<int>(loc(i, 1)) * cols + static_cast<int>(loc(i, 0)));
        }

        Eigen::SparseMatrix<int> adjacency_subset(valid_indices.size(), valid_indices.size());
        for (size_t i = 0; i < valid_indices.size(); ++i) {
            for (size_t j = 0; j < valid_indices.size(); ++j) {
                adjacency_subset.insert(i, j) = adjacency.coeff(valid_indices[i], valid_indices[j]);
            }
        }

        return std::make_tuple(adjacency_subset, Eigen::MatrixXd());
    } else {
        throw std::invalid_argument("'spatial_type' not among ['visium', 'imageST']");
    }
}

// Placeholder functions - must be implemented or linked separately
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> make_original_dendrogram_cc(const Eigen::VectorXd&, const Eigen::SparseMatrix<int>&, const std::vector<double>&);
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> make_smoothed_dendrogram(const std::vector<std::vector<int>>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const Eigen::ArrayXd&);
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> make_dendrogram_bar(const std::vector<int>&, const std::vector<int>&);

// Function to extract connected components
std::vector<std::vector<int>> extract_connected_comp(const Eigen::VectorXd& tx, const Eigen::SparseMatrix<int>& A_sparse, const std::vector<double>& threshold_x, int num_spots, int min_size) {
    auto [cCC_x, cE_x, cduration_x, chistory_x] = make_original_dendrogram_cc(tx, A_sparse, threshold_x);
    auto [nCC_x, nduration_x, nhistory_x] = make_smoothed_dendrogram(cCC_x, cE_x, cduration_x, chistory_x, Eigen::ArrayXd::LinSpaced(2, min_size, num_spots));
    auto [cvertical_x_x, cvertical_y_x, chorizontal_x_x, chorizontal_y_x, cdots_x, clayer_x] = make_dendrogram_bar(chistory_x, cduration_x);

    // Assuming nlayer_x contains data as expected
    std::vector<int> sind = clayer_x;
    std::vector<std::vector<int>> CCx;
    for (int i : sind) {
        CCx.push_back(nCC_x[i]);
    }
    return CCx;
}

// Function to extract the connected location matrix
Eigen::SparseMatrix<int> extract_connected_loc_mat(const std::vector<std::vector<int>>& CC, int num_spots, const std::string& format) {
    Eigen::MatrixXi CC_loc_arr = Eigen::MatrixXi::Zero(num_spots, CC.size());

    for (size_t num = 0; num < CC.size(); ++num) {
        const auto& element = CC[num];
        for (int idx : element) {
            CC_loc_arr(idx, num) = num + 1;
        }
    }

    if (format == "sparse") {
        return CC_loc_arr.sparseView();
    } else {
        throw std::invalid_argument("Sparse format required for compatibility.");
    }
}

// Function to filter connected component locations based on expression values
Eigen::SparseMatrix<int> filter_connected_loc_exp(const Eigen::SparseMatrix<int>& CC_loc_mat, const Eigen::MatrixXd& feat_data, int thres_per) {
    Eigen::VectorXd CC_mat_sum = CC_loc_mat * Eigen::VectorXd::Ones(CC_loc_mat.cols());

    std::vector<std::pair<int, double>> CC_mean;
    for (int i = 0; i < CC_loc_mat.rows(); ++i) {
        double sum = CC_mat_sum(i);
        if (sum != 0) {
            CC_mean.emplace_back(i, feat_data(i));
        }
    }

    std::sort(CC_mean.begin(), CC_mean.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
    });

    int cutoff = static_cast<int>(CC_mean.size() * (1 - thres_per / 100.0));
    CC_mean.resize(cutoff);

    Eigen::SparseMatrix<int> CC_loc_mat_fin(CC_loc_mat.rows(), CC_mean.size());
    for (size_t idx = 0; idx < CC_mean.size(); ++idx) {
        CC_loc_mat_fin.col(idx) = CC_loc_mat.col(CC_mean[idx].first);
    }

    return CC_loc_mat_fin;
}

// Function for topological connected component analysis
std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>> topological_comp_res(const Eigen::VectorXd& feat, const Eigen::SparseMatrix<int>& A, const Eigen::MatrixXd& mask,
                                                                                         const std::string& spatial_type, int min_size, int thres_per, const std::string& return_mode) {
    if (return_mode != "all" && return_mode != "cc_loc" && return_mode != "jaccard_cc_list") {
        throw std::invalid_argument("'return_mode' should be among 'all', 'cc_loc', or 'jaccard_cc_list'");
    }

    int p = feat.size();
    Eigen::VectorXd smooth = (spatial_type == "visium") ? (mask * feat).array() / feat.sum() : feat;

    Eigen::VectorXd t = smooth.cwiseMax(0);
    std::vector<double> threshold(t.data(), t.data() + t.size());
    std::sort(threshold.begin(), threshold.end(), std::greater<double>());
    threshold.erase(std::unique(threshold.begin(), threshold.end()), threshold.end());

    auto CC_list = extract_connected_comp(t, A, threshold, p, min_size);

    Eigen::SparseMatrix<int> CC_loc_mat = extract_connected_loc_mat(CC_list, p, "sparse");
    CC_loc_mat = filter_connected_loc_exp(CC_loc_mat, feat, thres_per, false);

    return std::make_tuple(CC_list, CC_loc_mat);
}