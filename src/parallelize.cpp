#include "parallelize.h"
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

// Parallel function for topological_comp_res with type conversion and progress callback
std::vector<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>> parallel_topological_comp(
    const std::vector<py::object>& locs, 
    const std::string& spatial_type, double fwhm,
    const std::vector<py::array_t<double>>& feats,  
    int min_size, int thres_per, const std::string& return_mode, int num_workers,
    py::function progress_callback) {

    ThreadPool pool(num_workers);
    std::vector<std::future<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>>> results;

    // Dispatch parallel tasks
    for (size_t i = 0; i < feats.size(); ++i) {
        // Convert the input Python object (NumPy array) to Eigen::MatrixXd
        Eigen::MatrixXd loc = locs[i].cast<Eigen::MatrixXd>();

        // Convert inputs using appropriate conversion functions
        Eigen::VectorXd feat = feats[i].cast<Eigen::VectorXd>();

        // Enqueue the task
        results.emplace_back(pool.enqueue(topological_comp_res, loc, spatial_type, fwhm, feat, min_size, thres_per, return_mode));
        
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
PYBIND11_MODULE(parallelize, m) {  // Module name within the STopover package
    m.def("parallel_topological_comp", &parallel_topological_comp, "Parallelized topological_comp_res function",
          py::arg("locs"), py::arg("spatial_type") = "visium", py::arg("fwhm") = 2.5, 
          py::arg("feats"), py::arg("min_size") = 5, py::arg("thres_per") = 30, py::arg("return_mode") = "all", 
          py::arg("num_workers") = 4, py::arg("progress_callback"));
    
    m.def("parallel_jaccard_composite", &parallel_jaccard_composite, "Parallelized jaccard_composite function",
          py::arg("CCx_loc_sums"), py::arg("CCy_loc_sums"), py::arg("feat_xs"), py::arg("feat_ys"), 
          py::arg("num_workers") = 4, py::arg("progress_callback"));
}