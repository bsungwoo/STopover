#include "type_conversion.h"
#include "logging.h"  // For log_message
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <fstream>
#include <chrono>
#include <iomanip>

namespace py = pybind11;

Eigen::SparseMatrix<double> scipy_sparse_to_eigen_sparse(const py::object& scipy_sparse_matrix) {
    try {
        // Check if the matrix is in COO format, if not, convert it
        py::object matrix_format = scipy_sparse_matrix.attr("format")();
        if (py::cast<std::string>(matrix_format) != "coo") {
            log_message("Converting sparse matrix from " + py::cast<std::string>(matrix_format) + " to COO format");
            scipy_sparse_matrix = scipy_sparse_matrix.attr("tocoo")();
        }
        
        // Get the shape, row indices, column indices, and data from the COO matrix
        py::tuple shape = scipy_sparse_matrix.attr("shape");
        py::array_t<int> row_indices = scipy_sparse_matrix.attr("row");
        py::array_t<int> col_indices = scipy_sparse_matrix.attr("col");
        py::array_t<double> data = scipy_sparse_matrix.attr("data");
        
        // Get the data pointers
        auto row_ptr = row_indices.data();
        auto col_ptr = col_indices.data();
        auto data_ptr = data.data();
        
        // Get dimensions
        int rows = py::cast<int>(shape[0]);
        int cols = py::cast<int>(shape[1]);
        int nnz = data.size(); // Number of non-zero elements
        
        // Create the Eigen sparse matrix
        Eigen::SparseMatrix<double> eigen_sparse(rows, cols);
        eigen_sparse.reserve(nnz);
        
        // Fill the triplet list
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(nnz);
        
        for (int i = 0; i < nnz; ++i) {
            tripletList.push_back(Eigen::Triplet<double>(row_ptr[i], col_ptr[i], data_ptr[i]));
        }
        
        // Set from triplets and compress
        eigen_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
        eigen_sparse.makeCompressed();
        
        return eigen_sparse;
    }
    catch (const py::error_already_set& e) {
        log_message("Python error in scipy_sparse_to_eigen_sparse: " + std::string(e.what()));
        throw std::runtime_error("Python error in scipy_sparse_to_eigen_sparse: " + std::string(e.what()));
    }
    catch (const py::cast_error& e) {
        log_message("Cast error in scipy_sparse_to_eigen_sparse: " + std::string(e.what()));
        throw std::runtime_error("Failed to cast attributes from SciPy sparse matrix: " + std::string(e.what()));
    }
    catch (const std::exception& e) {
        log_message("Error in scipy_sparse_to_eigen_sparse: " + std::string(e.what()));
        throw std::runtime_error("Error converting SciPy sparse matrix to Eigen::SparseMatrix<double>: " + std::string(e.what()));
    }
}