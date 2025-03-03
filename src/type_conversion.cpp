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
        // Create a local copy to work with, since the input is const
        py::object matrix = scipy_sparse_matrix;
        
        // Check if the matrix is in COO format, if not, convert it
        py::str format = matrix.attr("format")();
        std::string format_str = py::cast<std::string>(format);
        
        if (format_str != "coo") {
            log_message("Converting sparse matrix from " + format_str + " to COO format");
            matrix = matrix.attr("tocoo")();
        }
        
        // Get dimensions - properly access the shape tuple
        py::tuple shape = matrix.attr("shape");
        int n_rows = py::cast<int>(shape[0]);
        int n_cols = py::cast<int>(shape[1]);
        
        // Get arrays
        py::array_t<int> row_indices = py::cast<py::array_t<int>>(matrix.attr("row"));
        py::array_t<int> col_indices = py::cast<py::array_t<int>>(matrix.attr("col"));
        py::array_t<double> data = py::cast<py::array_t<double>>(matrix.attr("data"));
        
        // Get buffer info
        py::buffer_info row_buffer = row_indices.request();
        py::buffer_info col_buffer = col_indices.request();
        py::buffer_info data_buffer = data.request();
        
        // Check dimensions
        if (row_buffer.ndim != 1 || col_buffer.ndim != 1 || data_buffer.ndim != 1) {
            throw std::runtime_error("Sparse matrix row/col/data arrays must be 1D");
        }
        
        int nnz = data_buffer.shape[0];
        
        if (row_buffer.shape[0] != nnz || col_buffer.shape[0] != nnz) {
            throw std::runtime_error("Sparse matrix row/col/data arrays must have the same length");
        }
        
        // Get pointers
        int* row_ptr = static_cast<int*>(row_buffer.ptr);
        int* col_ptr = static_cast<int*>(col_buffer.ptr);
        double* data_ptr = static_cast<double*>(data_buffer.ptr);
        
        // Build Eigen sparse matrix from triplets
        typedef Eigen::Triplet<double> T;
        std::vector<T> triplets;
        triplets.reserve(nnz);
        
        for (int i = 0; i < nnz; ++i) {
            triplets.push_back(T(row_ptr[i], col_ptr[i], data_ptr[i]));
        }
        
        Eigen::SparseMatrix<double> eigen_sparse(n_rows, n_cols);
        eigen_sparse.setFromTriplets(triplets.begin(), triplets.end());
        eigen_sparse.makeCompressed();
        
        log_message("Converted SciPy sparse matrix to Eigen::SparseMatrix<double> with shape (" + 
                   std::to_string(n_rows) + ", " + std::to_string(n_cols) + ") and " + 
                   std::to_string(eigen_sparse.nonZeros()) + " non-zero entries");
        
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