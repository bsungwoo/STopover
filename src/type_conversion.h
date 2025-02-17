#ifndef TYPE_CONVERSION_H
#define TYPE_CONVERSION_H

#include <Eigen/Sparse>
#include <pybind11/pybind11.h>

// Function to convert a SciPy sparse matrix (COO format) to Eigen::SparseMatrix<double>
Eigen::SparseMatrix<double> scipy_sparse_to_eigen_sparse(const pybind11::object& scipy_sparse_matrix);

#endif // TYPE_CONVERSION_H