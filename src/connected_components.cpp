// Include necessary headers
#include "make_original_dendrogram.h"
#include "make_smoothed_dendrogram.h"
#include "make_dendrogram_bar.h"
#include "topological_comp.h"
#include "parallelize.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // For automatic conversion of STL containers

namespace py = pybind11;

// ------------------------- Helper Functions -------------------------

// Function to convert NumPy array to Eigen::MatrixXd
Eigen::MatrixXd array_to_matrix(const py::array& array) {
    // Ensure the array is two-dimensional
    if (array.ndim() != 2) {
        throw std::invalid_argument("Input array must be two-dimensional.");
    }

    // Check data type and handle accordingly
    if (py::isinstance<py::array_t<double>>(array)) {
        // Float64
        auto buf = array.request();
        size_t rows = buf.shape[0];
        size_t cols = buf.shape[1];
        const double* data_ptr = static_cast<const double*>(buf.ptr);

        // Ensure the array is contiguous
        if (!(array.flags() & py::array::c_style)) {
            throw std::invalid_argument("Input array must be C-contiguous.");
        }

        // Map data as row-major and then convert to column-major
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_map(data_ptr, rows, cols);
        Eigen::MatrixXd mat = mat_map;
        return mat;
    }
    else if (py::isinstance<py::array_t<float>>(array)) {
        // Float32
        auto buf = array.request();
        size_t rows = buf.shape[0];
        size_t cols = buf.shape[1];
        const float* data_ptr = static_cast<const float*>(buf.ptr);

        // Ensure the array is contiguous
        if (!(array.flags() & py::array::c_style)) {
            throw std::invalid_argument("Input array must be C-contiguous.");
        }

        // Convert float to double
        Eigen::MatrixXd mat(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                mat(i, j) = static_cast<double>(data_ptr[i * cols + j]);
            }
        }
        return mat;
    }
    else {
        throw std::invalid_argument("Unsupported data type. Only float32 and float64 are supported.");
    }
}

// Function to convert NumPy array to Eigen::VectorXd
Eigen::VectorXd array_to_vector(const py::array& array) {
    // Ensure the array is one-dimensional
    if (array.ndim() != 1) {
        throw std::invalid_argument("Input array must be one-dimensional.");
    }

    // Check data type and handle accordingly
    if (py::isinstance<py::array_t<double>>(array)) {
        // Float64
        auto buf = array.request();
        size_t size = buf.shape[0];
        const double* data_ptr = static_cast<const double*>(buf.ptr);

        // Ensure the array is contiguous
        if (!(array.flags() & py::array::c_style)) {
            throw std::invalid_argument("Input array must be C-contiguous.");
        }

        Eigen::Map<const Eigen::VectorXd> vec_map(data_ptr, size);
        Eigen::VectorXd vec = vec_map;
        return vec;
    }
    else if (py::isinstance<py::array_t<float>>(array)) {
        // Float32
        auto buf = array.request();
        size_t size = buf.shape[0];
        const float* data_ptr = static_cast<const float*>(buf.ptr);

        // Ensure the array is contiguous
        if (!(array.flags() & py::array::c_style)) {
            throw std::invalid_argument("Input array must be C-contiguous.");
        }

        // Convert float to double
        Eigen::VectorXd vec(size);
        for (size_t i = 0; i < size; ++i) {
            vec(i) = static_cast<double>(data_ptr[i]);
        }
        return vec;
    }
    else {
        throw std::invalid_argument("Unsupported data type. Only float32 and float64 are supported.");
    }
}

// Function to convert Eigen::SparseMatrix<double> to SciPy's CSR components
py::dict eigen_to_scipy_csr(const Eigen::SparseMatrix<double>& eigen_csr) {
    std::vector<double> data;
    std::vector<py::ssize_t> indices;
    std::vector<py::ssize_t> indptr;
    indptr.reserve(eigen_csr.rows() + 1);
    indptr.push_back(0);

    for (int k = 0; k < eigen_csr.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(eigen_csr, k); it; ++it) {
            data.push_back(it.value());
            indices.push_back(static_cast<py::ssize_t>(it.col()));
        }
        indptr.push_back(static_cast<py::ssize_t>(data.size()));
    }

    py::dict csr_dict;
    csr_dict["data"] = py::array_t<double>(data.size(), data.data());
    csr_dict["indices"] = py::array_t<py::ssize_t>(indices.size(), indices.data());
    csr_dict["indptr"] = py::array_t<py::ssize_t>(indptr.size(), indptr.data());
    csr_dict["shape"] = py::make_tuple(eigen_csr.rows(), eigen_csr.cols());

    return csr_dict;
}

// Function to convert Eigen::MatrixXd to NumPy array
py::array eigen_to_numpy(const Eigen::MatrixXd& mat) {
    // Create a NumPy array that shares memory with the Eigen matrix
    return py::array(
        { mat.rows(), mat.cols() }, // Shape
        { sizeof(double) * mat.cols(), sizeof(double) }, // Strides
        mat.data(), // Pointer to data
        py::cast(mat) // Reference to the Eigen matrix to keep it alive
    );
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
    return eigen_csr;
}

// ------------------------- Wrapper Functions -------------------------

// Wrapper function for extract_adjacency_spatial
py::list extract_adjacency_spatial_py(
    const std::vector<py::array_t<double>>& locs,
    const std::string& spatial_type = "visium",
    double fwhm = 2.5
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

        // Convert arr_mod to NumPy array
        py::array arr_mod_np = eigen_to_numpy(arr_mod);

        // Create a Python tuple (csr_matrix_dict, arr_mod_np)
        py::tuple result_tuple = py::make_tuple(csr_dict, arr_mod_np);

        // Append to the results list
        results.append(result_tuple);
    }

    return results;
}

// Wrapper function for make_original_dendrogram_cc
py::tuple make_original_dendrogram_cc_py(
    const py::array& U,
    const py::dict& A_csr,
    const std::vector<double>& threshold
) {
    // Convert U from NumPy array to Eigen::VectorXd
    Eigen::VectorXd U_eigen = array_to_vector(U);

    // Convert A_csr from SciPy CSR dict to Eigen::SparseMatrix<double>
    Eigen::SparseMatrix<double> A_eigen = scipy_csr_to_eigen(A_csr);

    // Call the original C++ function
    std::tuple<
        std::vector<std::vector<int>>,
        Eigen::SparseMatrix<double>,
        Eigen::MatrixXd,
        std::vector<std::vector<int>>
    > result = make_original_dendrogram_cc(U_eigen, A_eigen, threshold);

    // Unpack the results
    std::vector<std::vector<int>> nCC = std::get<0>(result);
    Eigen::SparseMatrix<double> nE_eigen = std::get<1>(result);
    Eigen::MatrixXd nduration = std::get<2>(result);
    std::vector<std::vector<int>> nchildren = std::get<3>(result);

    // Convert Eigen::SparseMatrix to SciPy CSR dict
    py::dict nE_csr = eigen_to_scipy_csr(nE_eigen);

    // Convert nduration to NumPy array
    py::array nduration_np = eigen_to_numpy(nduration);

    // Return as a Python tuple
    return py::make_tuple(nCC, nE_csr, nduration_np, nchildren);
}

// Wrapper function for make_smoothed_dendrogram
py::tuple make_smoothed_dendrogram_py(
    const std::vector<std::vector<int>>& cCC,
    const py::dict& cE,           // SciPy CSR matrix as dict
    const py::array& cduration,   // 2D NumPy array
    const std::vector<std::vector<int>>& chistory,
    const py::array& lim_size     // 1D NumPy array with 2 elements
) {
    // Convert cE from SciPy CSR to Eigen::SparseMatrix<double>
    Eigen::SparseMatrix<double> cE_eigen = scipy_csr_to_eigen(cE);

    // Convert cduration to Eigen::MatrixXd
    Eigen::MatrixXd cduration_eigen = array_to_matrix(cduration);

    // Convert lim_size to Eigen::Vector2d
    Eigen::VectorXd lim_size_vec = array_to_vector(lim_size);
    if (lim_size_vec.size() != 2) {
        throw std::invalid_argument("lim_size must be a 1D array with exactly 2 elements.");
    }
    Eigen::Vector2d lim_size_eigen;
    lim_size_eigen << lim_size_vec(0), lim_size_vec(1);

    // Call the original C++ function
    std::tuple<
        std::vector<std::vector<int>>,
        Eigen::SparseMatrix<double>,
        Eigen::MatrixXd,
        std::vector<std::vector<int>>
    > result = make_smoothed_dendrogram(cCC, cE_eigen, cduration_eigen, chistory, lim_size_eigen);

    // Unpack the results
    std::vector<std::vector<int>> nCC = std::get<0>(result);
    Eigen::SparseMatrix<double> nE_eigen = std::get<1>(result);
    Eigen::MatrixXd nduration = std::get<2>(result);
    std::vector<std::vector<int>> nchildren = std::get<3>(result);

    // Convert Eigen::SparseMatrix to SciPy CSR dict
    py::dict nE_csr = eigen_to_scipy_csr(nE_eigen);

    // Convert nduration to NumPy array
    py::array nduration_np = eigen_to_numpy(nduration);

    // Return as a Python tuple
    return py::make_tuple(nCC, nE_csr, nduration_np, nchildren);
}

// Wrapper function for make_dendrogram_bar
py::tuple make_dendrogram_bar_py(
    const std::vector<std::vector<int>>& history,
    const py::array& duration_np,
    const py::array& cvertical_x_np,
    const py::array& cvertical_y_np,
    const py::array& chorizontal_x_np,
    const py::array& chorizontal_y_np,
    const py::array& cdots_np
) {
    // Convert NumPy arrays to Eigen::MatrixXd
    Eigen::MatrixXd duration = array_to_matrix(duration_np);
    Eigen::MatrixXd cvertical_x = array_to_matrix(cvertical_x_np);
    Eigen::MatrixXd cvertical_y = array_to_matrix(cvertical_y_np);
    Eigen::MatrixXd chorizontal_x = array_to_matrix(chorizontal_x_np);
    Eigen::MatrixXd chorizontal_y = array_to_matrix(chorizontal_y_np);
    Eigen::MatrixXd cdots = array_to_matrix(cdots_np);

    // Call the original C++ function
    std::tuple<
        Eigen::MatrixXd,
        Eigen::MatrixXd,
        Eigen::MatrixXd,
        Eigen::MatrixXd,
        Eigen::MatrixXd,
        std::vector<std::vector<int>>
    > result = make_dendrogram_bar(history, duration, cvertical_x, cvertical_y, chorizontal_x, chorizontal_y, cdots);

    // Unpack the results
    Eigen::MatrixXd nvertical_x = std::get<0>(result);
    Eigen::MatrixXd nvertical_y = std::get<1>(result);
    Eigen::MatrixXd nhorizontal_x = std::get<2>(result);
    Eigen::MatrixXd nhorizontal_y = std::get<3>(result);
    Eigen::MatrixXd ndots = std::get<4>(result);
    std::vector<std::vector<int>> nlayer = std::get<5>(result);

    // Convert Eigen::MatrixXd to NumPy arrays
    py::array nvertical_x_np_out = eigen_to_numpy(nvertical_x);
    py::array nvertical_y_np_out = eigen_to_numpy(nvertical_y);
    py::array nhorizontal_x_np_out = eigen_to_numpy(nhorizontal_x);
    py::array nhorizontal_y_np_out = eigen_to_numpy(nhorizontal_y);
    py::array ndots_np_out = eigen_to_numpy(ndots);

    // Return as a Python tuple
    return py::make_tuple(nvertical_x_np_out, nvertical_y_np_out, nhorizontal_x_np_out, nhorizontal_y_np_out, ndots_np_out, nlayer);
}

// ------------------------- Pybind11 Module Binding -------------------------

PYBIND11_MODULE(connected_components, m) {  // Module name within the STopover package
    m.doc() = "Pybind11 wrapper for connected_components module functions";

    // Bind the helper conversion functions
    m.def("array_to_matrix", &array_to_matrix, "Convert NumPy array to Eigen::MatrixXd");
    m.def("array_to_vector", &array_to_vector, "Convert NumPy array to Eigen::VectorXd");
    m.def("eigen_to_numpy", &eigen_to_numpy, "Convert Eigen::MatrixXd to NumPy array");
    m.def("scipy_csr_to_eigen", &scipy_csr_to_eigen, "Convert SciPy CSR dict to Eigen::SparseMatrix<double>");
    m.def("eigen_to_scipy_csr", &eigen_to_scipy_csr, "Convert Eigen::SparseMatrix<double> to SciPy CSR dict");

    // Bind the wrapper functions
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
          "Wrapper for make_dendrogram_bar that accepts Python data types",
          py::arg("history"),
          py::arg("duration"),
          py::arg("cvertical_x"),
          py::arg("cvertical_y"),
          py::arg("chorizontal_x"),
          py::arg("chorizontal_y"),
          py::arg("cdots"));
}