// Include necessary headers
#include "make_original_dendrogram.h"
#include "make_smoothed_dendrogram.h"
#include "make_dendrogram_bar.h"
#include "topological_comp.h"
#include "parallelize.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // For automatic conversion of STL containers
#include <cstring> // For std::memcpy
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// ------------------------- Helper Functions -------------------------
// Function to convert NumPy array to Eigen::MatrixXd
Eigen::MatrixXd array_to_matrix(const py::array_t<double>& array) {
    if (array.ndim() != 2) {
        throw std::invalid_argument("Input array must be two-dimensional.");
    }

    py::buffer_info buf = array.request();
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];

    double* data_ptr = static_cast<double*>(buf.ptr);
    ssize_t stride_row = buf.strides[0] / sizeof(double);
    ssize_t stride_col = buf.strides[1] / sizeof(double);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        mat(data_ptr, rows, cols, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(stride_row, stride_col));

    return Eigen::MatrixXd(mat); // Make a deep copy if necessary
}

Eigen::VectorXd array_to_vector(const py::array_t<double>& array) {
    // Ensure the array is one-dimensional
    if (array.ndim() != 1) {
        throw std::invalid_argument("Input array must be one-dimensional.");
    }

    // Request a buffer descriptor from the NumPy array
    py::buffer_info buf = array.request();

    // Check if the array data type is double
    if (buf.format != py::format_descriptor<double>::format()) {
        throw std::invalid_argument("Input array must be of type float64.");
    }

    size_t size = buf.shape[0];
    double* data_ptr = static_cast<double*>(buf.ptr);
    ssize_t stride = buf.strides[0] / sizeof(double);

    // Create an Eigen::Map with custom stride
    typedef Eigen::Stride<Eigen::Dynamic, 1> StrideType;
    Eigen::Map<Eigen::VectorXd, 0, StrideType> vec(
        data_ptr, size, StrideType(stride, 1));

    // Make a copy of the vector to return
    return Eigen::VectorXd(vec);
}


// Function to convert Eigen::MatrixXd to NumPy array (with deep copy)
py::array_t<double> eigen_to_numpy(const Eigen::MatrixXd& mat) {
    return py::cast(mat);
}

// Function to convert Eigen::VectorXd to NumPy array
py::array_t<double> eigen_to_numpy_vector(const Eigen::VectorXd& vec) {
    // With Pybind11's Eigen support, this function is simplified
    return py::cast(vec);
}

// Function to convert Eigen::SparseMatrix<double> to SciPy's CSR components
py::dict eigen_to_scipy_csr(const Eigen::SparseMatrix<double>& eigen_csr) {
    std::vector<double> data;
    std::vector<py::ssize_t> indices;
    std::vector<py::ssize_t> indptr;
    indptr.reserve(static_cast<py::ssize_t>(eigen_csr.rows()) + 1);
    indptr.push_back(0);

    for (int k = 0; k < eigen_csr.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(eigen_csr, k); it; ++it) {
            data.push_back(it.value());
            indices.push_back(static_cast<py::ssize_t>(it.col()));
        }
        indptr.push_back(static_cast<py::ssize_t>(data.size()));
    }

    py::dict csr_dict;

    // Define shapes explicitly as separate vectors to avoid ambiguity
    std::vector<py::ssize_t> data_shape = { static_cast<py::ssize_t>(data.size()) };
    std::vector<py::ssize_t> indices_shape = { static_cast<py::ssize_t>(indices.size()) };
    std::vector<py::ssize_t> indptr_shape = { static_cast<py::ssize_t>(indptr.size()) };

    // Create NumPy arrays without using .copy(), ensuring proper ownership
    csr_dict["data"] = py::array_t<double>(data_shape, data.data());
    csr_dict["indices"] = py::array_t<py::ssize_t>(indices_shape, indices.data());
    csr_dict["indptr"] = py::array_t<py::ssize_t>(indptr_shape, indptr.data());
    csr_dict["shape"] = py::make_tuple(eigen_csr.rows(), eigen_csr.cols());

    return csr_dict;
}

// Function to convert SciPy's CSR components to Eigen::SparseMatrix<double>
Eigen::SparseMatrix<double> scipy_csr_to_eigen(const py::dict& csr_dict) {
    // Extract CSR components
    std::vector<double> data = csr_dict["data"].cast<std::vector<double>>();
    std::vector<py::ssize_t> indices = csr_dict["indices"].cast<std::vector<py::ssize_t>>();
    std::vector<py::ssize_t> indptr = csr_dict["indptr"].cast<std::vector<py::ssize_t>>();
    std::tuple<py::ssize_t, py::ssize_t> shape = csr_dict["shape"].cast<std::tuple<py::ssize_t, py::ssize_t>>();

    py::ssize_t rows = std::get<0>(shape);
    py::ssize_t cols = std::get<1>(shape);

    // Create Eigen::SparseMatrix
    Eigen::SparseMatrix<double> eigen_csr(rows, cols);
    eigen_csr.reserve(data.size());

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(data.size());

    for (py::ssize_t row = 0; row < rows; ++row) {
        for (py::ssize_t idx = indptr[row]; idx < indptr[row + 1]; ++idx) {
            py::ssize_t col = indices[idx];
            double value = data[idx];
            tripletList.emplace_back(static_cast<int>(row), static_cast<int>(col), value);
        }
    }

    eigen_csr.setFromTriplets(tripletList.begin(), tripletList.end());
    eigen_csr.makeCompressed(); // Optional: compress the matrix

    return eigen_csr;
}

// ------------------------- Wrapper Functions -------------------------

// Wrapper function for extract_adjacency_spatial
py::list extract_adjacency_spatial_py(
    const std::vector<py::array_t<double>>& locs,
    const std::string& spatial_type,
    double fwhm
) {
    py::list results;

    for (size_t i = 0; i < locs.size(); ++i) {
        // Convert loc array to Eigen::MatrixXd
        Eigen::MatrixXd loc_eigen = array_to_matrix(locs[i]);

        // Call the original C++ function
        std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd> adjacency_result =
            extract_adjacency_spatial(loc_eigen, spatial_type, fwhm);

        // Extract adjacency matrix and arr_mod
        Eigen::SparseMatrix<double> A_sparse = std::get<0>(adjacency_result);
        Eigen::MatrixXd arr_mod = std::get<1>(adjacency_result);

        // Convert Eigen::SparseMatrix to SciPy CSR format components
        py::dict csr_dict = eigen_to_scipy_csr(A_sparse);

        // Convert arr_mod to NumPy array using py::cast
        py::array_t<double> arr_mod_np = py::cast(arr_mod);

        // Create a Python tuple (csr_matrix_dict, arr_mod_np)
        py::tuple result_tuple = py::make_tuple(csr_dict, arr_mod_np);

        // Append to the results list
        results.append(result_tuple);
    }

    return results;
}

// Wrapper function for make_original_dendrogram_cc
py::tuple make_original_dendrogram_cc_py(
    const py::array_t<double>& U,
    const py::dict& A_csr,
    const std::vector<double>& threshold
) {
    try {
        // Convert U from NumPy array to Eigen::VectorXd
        Eigen::VectorXd U_eigen = array_to_vector(U);

        // Convert A_csr from SciPy CSR dict to Eigen::SparseMatrix<double>
        Eigen::SparseMatrix<double> A_eigen = scipy_csr_to_eigen(A_csr);

        // Call the original C++ function
        auto result = make_original_dendrogram_cc(U_eigen, A_eigen, threshold);

        // Unpack the results
        auto& nCC = std::get<0>(result);
        auto& nE_eigen = std::get<1>(result);
        auto& nduration = std::get<2>(result);
        auto& nchildren = std::get<3>(result);

        // Convert Eigen::SparseMatrix to SciPy CSR dict
        py::dict nE_csr = eigen_to_scipy_csr(nE_eigen);

        // Convert nduration to NumPy array using py::cast
        py::array_t<double> nduration_np = py::cast(nduration);

        // Return as a Python tuple
        return py::make_tuple(nCC, nE_csr, nduration_np, nchildren);
    }
    catch (const std::exception& e) {
        throw py::value_error(e.what());
    }
}

// Wrapper function for make_smoothed_dendrogram
py::tuple make_smoothed_dendrogram_py(
    const std::vector<std::vector<int>>& cCC,
    const py::dict& cE,
    const py::array_t<double>& cduration,
    const std::vector<std::vector<int>>& chistory,
    const py::array_t<double>& lim_size
) {
    try {
        // Convert cE from SciPy CSR to Eigen::SparseMatrix<double>
        Eigen::SparseMatrix<double> cE_eigen = scipy_csr_to_eigen(cE);

        // Convert cduration to Eigen::MatrixXd
        Eigen::MatrixXd cduration_eigen = array_to_matrix(cduration);

        // Convert lim_size to Eigen::Vector2d
        Eigen::VectorXd lim_size_vec = array_to_vector(lim_size);
        if (lim_size_vec.size() != 2) {
            throw std::invalid_argument("lim_size must be a 1D array with exactly 2 elements.");
        }
        Eigen::Vector2d lim_size_eigen = lim_size_vec.head<2>();

        // Call the original C++ function
        auto result = make_smoothed_dendrogram(cCC, cE_eigen, cduration_eigen, chistory, lim_size_eigen);

        // Unpack the results
        auto& nCC = std::get<0>(result);
        auto& nE_eigen = std::get<1>(result);
        auto& nduration = std::get<2>(result);
        auto& nchildren = std::get<3>(result);

        // Convert Eigen::SparseMatrix to SciPy CSR dict
        py::dict nE_csr = eigen_to_scipy_csr(nE_eigen);

        // Convert nduration to NumPy array using py::cast
        py::array_t<double> nduration_np = py::cast(nduration);

        // Return as a Python tuple
        return py::make_tuple(nCC, nE_csr, nduration_np, nchildren);
    }
    catch (const std::exception& e) {
        throw py::value_error(e.what());
    }
}


// Wrapper function for make_dendrogram_bar
py::tuple make_dendrogram_bar_py(
    const std::vector<std::vector<int>>& history,
    const py::array_t<double>& duration_np,
    py::object cvertical_x_obj = py::none(),
    py::object cvertical_y_obj = py::none(),
    py::object chorizontal_x_obj = py::none(),
    py::object chorizontal_y_obj = py::none(),
    py::object cdots_obj = py::none()
) {
    try {
        // Convert duration_np to Eigen::MatrixXd
        Eigen::MatrixXd duration = array_to_matrix(duration_np);

        // Initialize optional matrices as empty
        Eigen::MatrixXd cvertical_x = Eigen::MatrixXd();
        Eigen::MatrixXd cvertical_y = Eigen::MatrixXd();
        Eigen::MatrixXd chorizontal_x = Eigen::MatrixXd();
        Eigen::MatrixXd chorizontal_y = Eigen::MatrixXd();
        Eigen::MatrixXd cdots = Eigen::MatrixXd();

        // Check and cast each optional argument if not None
        if (!cvertical_x_obj.is_none()) {
            py::array_t<double> cvertical_x_np = cvertical_x_obj.cast<py::array_t<double>>();
            cvertical_x = array_to_matrix(cvertical_x_np);
        }

        if (!cvertical_y_obj.is_none()) {
            py::array_t<double> cvertical_y_np = cvertical_y_obj.cast<py::array_t<double>>();
            cvertical_y = array_to_matrix(cvertical_y_np);
        }

        if (!chorizontal_x_obj.is_none()) {
            py::array_t<double> chorizontal_x_np = chorizontal_x_obj.cast<py::array_t<double>>();
            chorizontal_x = array_to_matrix(chorizontal_x_np);
        }

        if (!chorizontal_y_obj.is_none()) {
            py::array_t<double> chorizontal_y_np = chorizontal_y_obj.cast<py::array_t<double>>();
            chorizontal_y = array_to_matrix(chorizontal_y_np);
        }

        if (!cdots_obj.is_none()) {
            py::array_t<double> cdots_np = cdots_obj.cast<py::array_t<double>>();
            cdots = array_to_matrix(cdots_np);
        }

        // Call the original C++ function
        auto result = make_dendrogram_bar(history, duration, cvertical_x, cvertical_y, chorizontal_x, chorizontal_y, cdots);

        // Unpack the results
        Eigen::MatrixXd nvertical_x = std::get<0>(result);
        Eigen::MatrixXd nvertical_y = std::get<1>(result);
        Eigen::MatrixXd nhorizontal_x = std::get<2>(result);
        Eigen::MatrixXd nhorizontal_y = std::get<3>(result);
        Eigen::MatrixXd ndots = std::get<4>(result);
        std::vector<std::vector<int>> nlayer = std::get<5>(result);

        // Convert Eigen::MatrixXd to NumPy arrays using py::cast
        py::array_t<double> nvertical_x_np_out = py::cast(nvertical_x);
        py::array_t<double> nvertical_y_np_out = py::cast(nvertical_y);
        py::array_t<double> nhorizontal_x_np_out = py::cast(nhorizontal_x);
        py::array_t<double> nhorizontal_y_np_out = py::cast(nhorizontal_y);
        py::array_t<double> ndots_np_out = py::cast(ndots);

        // Return as a Python tuple
        return py::make_tuple(
            nvertical_x_np_out,
            nvertical_y_np_out,
            nhorizontal_x_np_out,
            nhorizontal_y_np_out,
            ndots_np_out,
            nlayer
        );
    }
    catch (const std::exception& e) {
        throw py::value_error(e.what());
    }
}

// ------------------------- Pybind11 Module Binding -------------------------

// Pybind11 Module Binding
PYBIND11_MODULE(connected_components, m) {
    m.doc() = "Pybind11 wrapper for connected_components module functions";

    // Bind the wrapper functions without _py suffix
    m.def("extract_adjacency_spatial", &extract_adjacency_spatial_py, "Extract adjacency spatial information",
          py::arg("locs"),
          py::arg("spatial_type") = "visium",
          py::arg("fwhm") = 2.5);

    m.def("make_original_dendrogram_cc", &make_original_dendrogram_cc_py, 
          "Wrapper for make_original_dendrogram_cc that accepts Python data types",
          py::arg("U"),
          py::arg("A_csr"),
          py::arg("threshold"));

    m.def("make_smoothed_dendrogram", &make_smoothed_dendrogram_py, 
          "Wrapper for make_smoothed_dendrogram that accepts Python data types",
          py::arg("cCC"),
          py::arg("cE"),
          py::arg("cduration"),
          py::arg("chistory"),
          py::arg("lim_size"));

    m.def("make_dendrogram_bar", &make_dendrogram_bar_py,
          py::arg("history"),
          py::arg("duration"),
          py::arg("cvertical_x") = py::none(),
          py::arg("cvertical_y") = py::none(),
          py::arg("chorizontal_x") = py::none(),
          py::arg("chorizontal_y") = py::none(),
          py::arg("cdots") = py::none(),
          "Create dendrogram bars with optional parameters");
}