#ifndef DENDROGRAM_BAR_H
#define DENDROGRAM_BAR_H

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
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
                    Eigen::MatrixXd cdots = Eigen::MatrixXd())

#endif // DENDROGRAM_BAR_H