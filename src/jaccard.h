#ifndef JACCARD_H
#define JACCARD_H

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <set>

// Function to calculate the Jaccard composite index from connected component locations
double jaccard_composite(const Eigen::MatrixXd& CCx_loc_sum, const Eigen::MatrixXd& CCy_loc_sum,
                         const Eigen::MatrixXd& feat_x = Eigen::MatrixXd(), const Eigen::MatrixXd& feat_y = Eigen::MatrixXd());

#endif // JACCARD_H