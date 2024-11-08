#include "type_conversion.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

Eigen::SparseMatrix<double> scipy_sparse_to_eigen_sparse(const py::object& scipy_sparse_matrix) {
    try {
        // Ensure the matrix is in COO format
        // Extract row, col, data arrays
        py::array_t<int> row = scipy_sparse_matrix.attr("row").cast<py::array_t<int>>();
        py::array_t<int> col = scipy_sparse_matrix.attr("col").cast<py::array_t<int>>();
        py::array_t<double> data = scipy_sparse_matrix.attr("data").cast<py::array_t<double>>();

        // Get shape of the matrix
        py::tuple shape = scipy_sparse_matrix.attr("shape").cast<py::tuple>();
        if (shape.size() != 2) {
            throw std::invalid_argument("Sparse matrix must be two-dimensional.");
        }
        int nrows = shape[0].cast<int>();
        int ncols = shape[1].cast<int>();
        int nnz = data.size();

        // Validate that row, col, and data have the same length
        if (row.size() != nnz || col.size() != nnz) {
            throw std::invalid_argument("Row, column, and data arrays must have the same length.");
        }

        // Convert to Eigen::SparseMatrix<double>
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(nnz);

        auto row_ptr = row.unchecked<1>();
        auto col_ptr = col.unchecked<1>();
        auto data_ptr = data.unchecked<1>();

        for (int i = 0; i < nnz; ++i) {
            tripletList.emplace_back(row_ptr(i), col_ptr(i), data_ptr(i));
        }

        Eigen::SparseMatrix<double> mat(nrows, ncols);
        mat.setFromTriplets(tripletList.begin(), tripletList.end());
        mat.makeCompressed(); // Optimize the storage

        return mat;
    }
    catch (const py::cast_error& e) {
        throw std::runtime_error("Failed to cast attributes from SciPy sparse matrix: " + std::string(e.what()));
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error converting SciPy sparse matrix to Eigen::SparseMatrix<double>: " + std::string(e.what()));
    }
}