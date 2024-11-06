#include "type_conversion.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Sparse>

namespace py = pybind11;
using namespace Eigen;

// Function to convert scipy.sparse.csr_matrix to Eigen::SparseMatrix
SparseMatrix<double> scipy_sparse_to_eigen_sparse(const py::object& scipy_sparse) {
    // Assuming scipy_sparse is a CSR matrix
    py::object data = scipy_sparse.attr("data");
    py::object indices = scipy_sparse.attr("indices");
    py::object indptr = scipy_sparse.attr("indptr");
    py::object shape = scipy_sparse.attr("shape");

    auto data_array = data.cast<py::array_t<double>>();
    auto indices_array = indices.cast<py::array_t<int>>();
    auto indptr_array = indptr.cast<py::array_t<int>>();
    auto shape_tuple = shape.cast<std::tuple<int, int>>();

    int rows = std::get<0>(shape_tuple);
    int cols = std::get<1>(shape_tuple);

    SparseMatrix<double> eigen_sparse(rows, cols);

    std::vector<Triplet<double>> tripletList;
    tripletList.reserve(data_array.size());

    // Convert data from CSR to Eigen's SparseMatrix
    const double* data_ptr = static_cast<const double*>(data_array.data());
    const int* indices_ptr = static_cast<const int*>(indices_array.data());
    const int* indptr_ptr = static_cast<const int*>(indptr_array.data());

    for (int i = 0; i < rows; ++i) {
        for (int j = indptr_ptr[i]; j < indptr_ptr[i + 1]; ++j) {
            tripletList.emplace_back(i, indices_ptr[j], data_ptr[j]);
        }
    }

    eigen_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
    return eigen_sparse;
}

// Function to convert Eigen::SparseMatrix to scipy.sparse.csr_matrix
py::object eigen_sparse_to_scipy_sparse(const SparseMatrix<int>& eigen_matrix) {
    std::vector<int> data, indices, indptr(eigen_matrix.rows() + 1, 0);

    for (int k = 0; k < eigen_matrix.outerSize(); ++k) {
        for (SparseMatrix<int>::InnerIterator it(eigen_matrix, k); it; ++it) {
            data.push_back(it.value());
            indices.push_back(it.col());
            indptr[it.row() + 1]++;  // Track the end of each row
        }
    }

    // Accumulate indptr to get correct CSR format
    for (int i = 1; i < indptr.size(); ++i) {
        indptr[i] += indptr[i - 1];
    }

    // Convert C++ arrays to Python arrays for scipy.sparse.csr_matrix construction
    py::array_t<int> data_py(data.size(), data.data());
    py::array_t<int> indices_py(indices.size(), indices.data());
    py::array_t<int> indptr_py(indptr.size(), indptr.data());

    // Create a scipy.sparse.csr_matrix using Python tuple
    py::object scipy_sparse = py::module::import("scipy.sparse").attr("csr_matrix")(
        py::make_tuple(data_py, indices_py, indptr_py), py::make_tuple(eigen_matrix.rows(), eigen_matrix.cols())
    );
    return scipy_sparse;
}