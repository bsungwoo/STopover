#ifndef TYPE_CONVERSION_H
#define TYPE_CONVERSION_H

#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include "logging.h"  // Include logging header
#include <string>
#include <stdexcept>

namespace py = pybind11;

// Function to convert a SciPy sparse matrix (COO format) to Eigen::SparseMatrix<double>
Eigen::SparseMatrix<double> scipy_sparse_to_eigen_sparse(const py::object& scipy_sparse_matrix);

// Function to convert Eigen::SparseMatrix<double> to SciPy CSR format
py::dict eigen_to_scipy_csr(const Eigen::SparseMatrix<double>& eigen_csr);

#endif // TYPE_CONVERSION_H