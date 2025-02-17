#ifndef SMOOTHED_DENDROGRAM_H
#define SMOOTHED_DENDROGRAM_H

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <numeric>
#include <set>
#include <cmath>
#include <tuple>

std::tuple<std::vector<std::vector<int>>, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<std::vector<int>>>
make_smoothed_dendrogram(const std::vector<std::vector<int>>& cCC,
                         Eigen::MatrixXd cE,
                         Eigen::MatrixXd cduration,
                         const std::vector<std::vector<int>>& chistory,
                         Eigen::Vector2d lim_size = Eigen::Vector2d(0, std::numeric_limits<double>::infinity()));

#endif // SMOOTHED_DENDROGRAM_H