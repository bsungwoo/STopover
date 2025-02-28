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
#include <string>

// Function to calculate the Jaccard composite index from connected component locations
double jaccard_composite(const std::vector<std::vector<int>>& cc_1, 
                         const std::vector<std::vector<int>>& cc_2,
                         const std::string& jaccard_type);

#endif // JACCARD_H