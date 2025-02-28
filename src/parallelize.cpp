#include "parallelize.h"
#include "type_conversion.h"       // For scipy_sparse_to_eigen_sparse
#include "topological_comp.h"      // For extract_adjacency_spatial and topological_comp_res
#include "jaccard.h"               // For jaccard_composite

#include <stdexcept>
#include <atomic>
#include <future>
#include <vector>
#include <tuple>
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// ---------------------------------------------------------------------
// ThreadPool Implementation
// ---------------------------------------------------------------------
ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this]{
                        return this->stop.load() || !this->tasks.empty();
                    });
                    if (this->stop.load() && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                try {
                    task();
                } catch (const std::exception &e) {
                    // Optionally log error: e.what()
                }
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        stop.store(true);
    }
    condition.notify_all();
    for (std::thread &worker : workers)
        if(worker.joinable())
            worker.join();
}

// ---------------------------------------------------------------------
// Parallel Function: extract_adjacency_spatial
// ---------------------------------------------------------------------
std::vector<std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd>> parallel_extract_adjacency(
    const std::vector<py::object>& locs,
    const std::string& spatial_type,
    double fwhm,
    int num_workers,
    py::function progress_callback)
{
    ThreadPool pool(num_workers);
    std::vector<std::future<std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd>>> results;
    std::atomic<int> count{0};

    for (const auto& loc_py : locs) {
        {
            // Release the GIL during heavy computation.
            py::gil_scoped_release release;
            Eigen::MatrixXd loc = loc_py.cast<Eigen::MatrixXd>();
            results.emplace_back(pool.enqueue(extract_adjacency_spatial, loc, spatial_type, fwhm));
        }
        if (progress_callback && (++count % 10 == 0)) {
            py::gil_scoped_acquire acquire;
            progress_callback();
        }
    }
    std::vector<std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd>> output;
    output.reserve(results.size());
    for (auto& result : results) {
        try {
            output.emplace_back(result.get());
        } catch (...) {
            // Optionally log error or skip this task.
        }
    }
    return output;
}

// ---------------------------------------------------------------------
// Parallel Function: topological_comp_res
// ---------------------------------------------------------------------
std::vector<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>> parallel_topological_comp(
    const std::vector<py::array_t<double>>& feats,
    const std::vector<py::object>& A_matrices,
    const std::vector<py::array_t<double>>& masks,
    const std::string& spatial_type,
    int min_size,
    int thres_per,
    const std::string& return_mode,
    int num_workers,
    py::function progress_callback)
{
    ThreadPool pool(num_workers);
    std::vector<std::future<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>>> results;
    std::atomic<int> count{0};

    for (size_t i = 0; i < feats.size(); ++i) {
        {
            py::gil_scoped_release release;
            Eigen::VectorXd feat;
            try {
                feat = feats[i].cast<Eigen::VectorXd>();
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to cast feature at index " + std::to_string(i) + ": " + e.what());
            }
            
            // Handle null masks properly
            Eigen::MatrixXd mask;
            if (masks[i].size() > 0) {
                try {
                    mask = masks[i].cast<Eigen::MatrixXd>();
                } catch (const std::exception& e) {
                    throw std::runtime_error("Failed to cast mask at index " + std::to_string(i) + ": " + e.what());
                }
            }
            
            // Convert SciPy sparse matrix (py::object) to Eigen::SparseMatrix<double> with error handling
            Eigen::SparseMatrix<double> A_matrix_double;
            try {
                A_matrix_double = scipy_sparse_to_eigen_sparse(A_matrices[i]);
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to convert sparse matrix at index " + std::to_string(i) + ": " + e.what());
            }
            
            results.emplace_back(pool.enqueue(topological_comp_res, feat, A_matrix_double, mask,
                                               spatial_type, min_size, thres_per, return_mode));
        }
        if (progress_callback && (++count % 10 == 0)) {
            py::gil_scoped_acquire acquire;
            try {
                progress_callback();
            } catch (const std::exception& e) {
                // Log error but continue processing
            }
        }
    }
    std::vector<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>> output;
    output.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        try {
            output.push_back(results[i].get());
        } catch (const std::exception& e) {
            // Log the error and push an empty result instead of crashing
            std::cerr << "Error processing result at index " << i << ": " << e.what() << std::endl;
            output.push_back(std::make_tuple(std::vector<std::vector<int>>(), Eigen::SparseMatrix<int>()));
        }
    }
    return output;
}

// ---------------------------------------------------------------------
// Parallel Function: jaccard_composite
// ---------------------------------------------------------------------
std::vector<double> parallel_jaccard_composite(
    const std::vector<py::array_t<double>>& CCx_loc_sums,
    const std::vector<py::array_t<double>>& CCy_loc_sums,
    const std::vector<py::array_t<double>>& feat_xs,
    const std::vector<py::array_t<double>>& feat_ys,
    int num_workers,
    py::function progress_callback)
{
    ThreadPool pool(num_workers);
    std::vector<std::future<double>> results;
    std::atomic<int> count{0};

    for (size_t i = 0; i < CCx_loc_sums.size(); ++i) {
        {
            py::gil_scoped_release release;
            Eigen::MatrixXd CCx_loc_sum = CCx_loc_sums[i].cast<Eigen::MatrixXd>();
            Eigen::MatrixXd CCy_loc_sum = CCy_loc_sums[i].cast<Eigen::MatrixXd>();
            Eigen::MatrixXd feat_x = feat_xs[i].cast<Eigen::MatrixXd>();
            Eigen::MatrixXd feat_y = feat_ys[i].cast<Eigen::MatrixXd>();
            results.emplace_back(pool.enqueue(jaccard_composite, CCx_loc_sum, CCy_loc_sum, feat_x, feat_y));
        }
        if (progress_callback && (++count % 10 == 0)) {
            py::gil_scoped_acquire acquire;
            progress_callback();
        }
    }
    std::vector<double> output;
    output.reserve(results.size());
    for (auto& result : results) {
        try {
            output.push_back(result.get());
        } catch (...) {
            // Optionally log error and skip.
        }
    }
    return output;
}

// ---------------------------------------------------------------------
// Pybind11 Module Definition
// ---------------------------------------------------------------------
PYBIND11_MODULE(parallelize, m) {
    m.doc() = "Parallelized functions for topological and Jaccard computations";

    m.def("parallel_extract_adjacency", &parallel_extract_adjacency,
          "Parallelized extract_adjacency_spatial function",
          py::arg("locs"),
          py::arg("spatial_type") = "visium",
          py::arg("fwhm") = 2.5,
          py::arg("num_workers") = 4,
          py::arg("progress_callback") = py::none());

    m.def("parallel_topological_comp", &parallel_topological_comp,
          "Parallelized topological_comp_res function",
          py::arg("feats"),
          py::arg("A_matrices"),
          py::arg("masks"),
          py::arg("spatial_type") = "visium",
          py::arg("min_size") = 5,
          py::arg("thres_per") = 30,
          py::arg("return_mode") = "all",
          py::arg("num_workers") = 4,
          py::arg("progress_callback") = py::none());

    m.def("parallel_jaccard_composite", &parallel_jaccard_composite,
          "Parallelized jaccard_composite function",
          py::arg("CCx_loc_sums"),
          py::arg("CCy_loc_sums"),
          py::arg("feat_xs"),
          py::arg("feat_ys"),
          py::arg("num_workers") = 4,
          py::arg("progress_callback") = py::none());
}