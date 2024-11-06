#ifndef TYPE_CONVERSION_H
#define TYPE_CONVERSION_H

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;
using namespace Eigen;

// Function to convert Eigen::SparseMatrix to scipy.sparse.csr_matrix
SparseMatrix<double> scipy_sparse_to_eigen_sparse(const py::object& scipy_sparse) 

// Function to convert Eigen::SparseMatrix to scipy.sparse.csr_matrix
py::object eigen_sparse_to_scipy_sparse(const SparseMatrix<int>& eigen_matrix)

#endif // TYPE_CONVERSION_H