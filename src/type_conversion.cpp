#include "type_conversion.h"
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
    std::vector<Triplet<int>> triplet_list;
    for (int k = 0; k < eigen_matrix.outerSize(); ++k) {
        for (SparseMatrix<int>::InnerIterator it(eigen_matrix, k); it; ++it) {
            triplet_list.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // Create the data, indices, and indptr arrays for scipy.sparse.csr_matrix
    py::list data, indices, indptr;
    int row = 0;
    for (const auto& triplet : triplet_list) {
        data.append(triplet.value());
        indices.append(triplet.col());
        if (row != triplet.row()) {
            indptr.append(triplet.row());
            row = triplet.row();
        }
    }
    indptr.append(eigen_matrix.outerSize());

    py::object scipy_sparse = py::module::import("scipy.sparse").attr("csr_matrix")(
        py::make_tuple(data, indices, indptr), py::make_tuple(eigen_matrix.rows(), eigen_matrix.cols())
    );
    return scipy_sparse;
}