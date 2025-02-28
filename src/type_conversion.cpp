#include "type_conversion.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

Eigen::SparseMatrix<double> scipy_sparse_to_eigen_sparse(const py::object& scipy_sparse_matrix) {
    try {
        // Check if the matrix is in COO format, if not convert it
        py::object coo_matrix;
        if (py::hasattr(scipy_sparse_matrix, "format") && 
            scipy_sparse_matrix.attr("format").cast<std::string>() != "coo") {
            
            log_message("Converting sparse matrix from " + 
                       scipy_sparse_matrix.attr("format").cast<std::string>() + 
                       " to COO format");
            
            // Convert to COO format
            coo_matrix = scipy_sparse_matrix.attr("tocoo")();
        } else {
            coo_matrix = scipy_sparse_matrix;
        }
        
        // Extract row, col, data arrays from COO format
        py::array_t<int> row = coo_matrix.attr("row").cast<py::array_t<int>>();
        py::array_t<int> col = coo_matrix.attr("col").cast<py::array_t<int>>();
        py::array_t<double> data = coo_matrix.attr("data").cast<py::array_t<double>>();
        
        // Get matrix dimensions
        int nrows = coo_matrix.attr("shape").attr("__getitem__")(0).cast<int>();
        int ncols = coo_matrix.attr("shape").attr("__getitem__")(1).cast<int>();
        
        // Create triplet list for Eigen sparse matrix
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(data.size());
        
        auto row_unchecked = row.unchecked<1>();
        auto col_unchecked = col.unchecked<1>();
        auto data_unchecked = data.unchecked<1>();
        
        for (py::ssize_t i = 0; i < data.size(); ++i) {
            tripletList.push_back(Eigen::Triplet<double>(
                row_unchecked[i], col_unchecked[i], data_unchecked[i]));
        }
        
        Eigen::SparseMatrix<double> mat(nrows, ncols);
        mat.setFromTriplets(tripletList.begin(), tripletList.end());
        mat.makeCompressed(); // Optimize the storage
        
        return mat;
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