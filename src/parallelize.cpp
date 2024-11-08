#include "parallelize.h"
#include "type_conversion.h"  // Include the conversion header
#include "topological_comp.h"
#include "jaccard.h"

#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// ThreadPool constructor
ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    // Corrected lambda to return bool
                    this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
    }
}

// ThreadPool destructor
ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers)
        worker.join();
}

// Parallel function implementations...

// Example: Implement parallel_extract_adjacency
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
        output.emplace_back(result.get());
    }

    return output;
}

// Similarly, implement parallel_topological_comp and parallel_jaccard_composite

// Expose to Python via Pybind11
PYBIND11_MODULE(parallelize, m) {  // Module name within the STopover package
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