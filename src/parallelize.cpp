#include "parallelize.h"
#include "type_conversion.h"  // Include the conversion header
#include "topological_comp.h"
#include "jaccard.h"

#include <future>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// ThreadPool constructor and destructor are already defined in the header (inline)

// Parallel function for extract_adjacency_spatial with type conversion and progress callback
std::vector<std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd>> parallel_extract_adjacency(
    const std::vector<py::object>& locs, 
    const std::string& spatial_type, double fwhm, int num_workers,
    py::function progress_callback) {

    ThreadPool pool(num_workers);
    std::vector<std::future<std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd>>> results;

    // Dispatch parallel tasks
    for (const auto& loc_py : locs) {
        // Convert the input Python object (NumPy array) to Eigen::MatrixXd
        Eigen::MatrixXd loc = loc_py.cast<Eigen::MatrixXd>();
        results.emplace_back(pool.enqueue(extract_adjacency_spatial, loc, spatial_type, fwhm));
        
        // Call the progress callback
        if (progress_callback) {
            progress_callback();
        }
    }

    // Collect the results
    std::vector<std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd>> output;
    output.reserve(results.size());
    for (auto& result : results) {
        output.push_back(result.get());
    }

    return output;
}

// Parallel function for topological_comp_res with type conversion and progress callback
std::vector<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>> parallel_topological_comp(
    const std::vector<py::array_t<double>>& feats,  
    const std::vector<py::object>& A_matrices,      
    const std::vector<py::array_t<double>>& masks,
    const std::string& spatial_type, int min_size, int thres_per, const std::string& return_mode, int num_workers,
    py::function progress_callback) {

    ThreadPool pool(num_workers);
    std::vector<std::future<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>>> results;

    // Dispatch parallel tasks
    for (size_t i = 0; i < feats.size(); ++i) {
        // Convert inputs using appropriate conversion functions
        Eigen::VectorXd feat = feats[i].cast<Eigen::VectorXd>();
        
        // Convert SciPy sparse matrix to Eigen::SparseMatrix<double>
        Eigen::SparseMatrix<double> A_matrix_double = scipy_sparse_to_eigen_sparse(A_matrices[i]);
        
        Eigen::MatrixXd mask = masks[i].cast<Eigen::MatrixXd>();

        // Enqueue the task
        results.emplace_back(pool.enqueue(topological_comp_res, feat, A_matrix_double, mask, spatial_type, min_size, thres_per, return_mode));
        
        // Call the progress callback
        if (progress_callback) {
            progress_callback();
        }
    }

    // Collect the results
    std::vector<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>> output;
    output.reserve(results.size());
    for (auto& result : results) {
        output.push_back(result.get());
    }

    return output;
}

// Parallel function for jaccard_composite with type conversion and progress callback
std::vector<double> parallel_jaccard_composite(
    const std::vector<py::array_t<double>>& CCx_loc_sums, 
    const std::vector<py::array_t<double>>& CCy_loc_sums,
    const std::vector<py::array_t<double>>& feat_xs, 
    const std::vector<py::array_t<double>>& feat_ys, 
    int num_workers,
    py::function progress_callback) {

    ThreadPool pool(num_workers);
    std::vector<std::future<double>> results;

    // Dispatch parallel tasks
    for (size_t i = 0; i < CCx_loc_sums.size(); ++i) {
        // Convert inputs from NumPy to Eigen
        Eigen::MatrixXd CCx_loc_sum = CCx_loc_sums[i].cast<Eigen::MatrixXd>();
        Eigen::MatrixXd CCy_loc_sum = CCy_loc_sums[i].cast<Eigen::MatrixXd>();
        Eigen::MatrixXd feat_x = feat_xs[i].cast<Eigen::MatrixXd>();
        Eigen::MatrixXd feat_y = feat_ys[i].cast<Eigen::MatrixXd>();

        // Enqueue the task
        results.emplace_back(pool.enqueue(jaccard_composite, CCx_loc_sum, CCy_loc_sum, feat_x, feat_y));
        
        // Call the progress callback
        if (progress_callback) {
            progress_callback();
        }
    }

    // Collect the results
    std::vector<double> output;
    output.reserve(results.size());
    for (auto& result : results) {
        output.push_back(result.get());
    }

    return output;
}

// Expose to Python via Pybind11
PYBIND11_MODULE(spatial_analysis, m) {
    m.def("parallel_extract_adjacency", &parallel_extract_adjacency, "Parallelized extract_adjacency_spatial function",
          py::arg("locs"), py::arg("spatial_type") = "visium", py::arg("fwhm") = 2.5, 
          py::arg("num_workers") = 4, py::arg("progress_callback"));
    
    m.def("parallel_topological_comp", &parallel_topological_comp, "Parallelized topological_comp_res function",
          py::arg("feats"), py::arg("A_matrices"), py::arg("masks"), py::arg("spatial_type") = "visium", 
          py::arg("min_size") = 5, py::arg("thres_per") = 30, py::arg("return_mode") = "all", 
          py::arg("num_workers") = 4, py::arg("progress_callback"));
    
    m.def("parallel_jaccard_composite", &parallel_jaccard_composite, "Parallelized jaccard_composite function",
          py::arg("CCx_loc_sums"), py::arg("CCy_loc_sums"), py::arg("feat_xs"), py::arg("feat_ys"), 
          py::arg("num_workers") = 4, py::arg("progress_callback"));
}