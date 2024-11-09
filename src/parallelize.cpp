#include "parallelize.h"
#include "topological_comp.h"
#include "jaccard.h"

#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>        // For automatic conversion of STL containers
#include <pybind11/numpy.h>      // For handling NumPy arrays
#include <future>                // For std::future
#include <vector>
#include <utility>               // For std::pair

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

// Helper function to convert py::array_t<double> to Eigen::VectorXd
Eigen::VectorXd array_to_vector(const py::array_t<double>& array) {
    // Ensure the array is one-dimensional
    if (array.ndim() != 1) {
        throw std::invalid_argument("All input arrays must be one-dimensional.");
    }

    // Get buffer info
    py::buffer_info buf = array.request();

    // Create Eigen::Map without copying data
    Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(static_cast<double*>(buf.ptr), buf.shape[0]);

    return vec;
}

// Parallel function for topological_comp_res with type conversion and progress callback
std::vector<Eigen::VectorXd> parallel_topological_comp(
    const std::vector<py::object>& locs, 
    const std::string& spatial_type, double fwhm,
    const std::vector<py::array_t<double>>& feats,  
    int min_size, int thres_per, const std::string& return_mode, int num_workers,
    py::function progress_callback) {

    ThreadPool pool(num_workers);
    std::vector<std::future<std::pair<size_t, Eigen::VectorXd>>> results;

    // Dispatch parallel tasks
    for (size_t i = 0; i < feats.size(); ++i) {
        // Capture index and inputs by value
        size_t index = i;
        py::object loc_obj = locs[i];
        py::array_t<double> feat_array = feats[i];

        // Enqueue the task
        results.emplace_back(pool.enqueue([=]() -> std::pair<size_t, Eigen::VectorXd> {
            // Convert inputs inside the task to avoid issues with object lifetimes
            Eigen::MatrixXd loc = loc_obj.cast<Eigen::MatrixXd>();
            Eigen::VectorXd feat = array_to_vector(feat_array);

            // Call the computation function
            Eigen::VectorXd res = topological_comp_res(loc, spatial_type, fwhm, feat, min_size, thres_per, return_mode);
            return {index, res};
        }));

        // Call the progress callback
        if (progress_callback) {
            progress_callback();
        }
    }

    // Collect the results in the correct order
    std::vector<Eigen::VectorXd> output(feats.size());
    for (auto& result_future : results) {
        auto result_pair = result_future.get();
        size_t index = result_pair.first;
        Eigen::VectorXd res = result_pair.second;
        output[index] = res;
    }

    return output;
}

// Updated parallel_jaccard_composite function to handle lists of NumPy arrays
std::vector<double> parallel_jaccard_composite_py(
    py::list CCx_loc_sums_list, 
    py::list CCy_loc_sums_list,
    py::list feat_xs_list, 
    py::list feat_ys_list, 
    int num_workers,
    py::function progress_callback) {

    // Check that all lists have the same size
    size_t list_size = CCx_loc_sums_list.size();
    if (CCy_loc_sums_list.size() != list_size ||
        feat_xs_list.size() != list_size ||
        feat_ys_list.size() != list_size) {
        throw std::invalid_argument("All input lists must have the same length.");
    }

    // Convert each NumPy array in the lists to Eigen::VectorXd
    std::vector<Eigen::VectorXd> CCx_loc_sums_vec;
    std::vector<Eigen::VectorXd> CCy_loc_sums_vec;
    std::vector<Eigen::VectorXd> feat_xs_vec;
    std::vector<Eigen::VectorXd> feat_ys_vec;

    CCx_loc_sums_vec.reserve(list_size);
    CCy_loc_sums_vec.reserve(list_size);
    feat_xs_vec.reserve(list_size);
    feat_ys_vec.reserve(list_size);

    for (size_t i = 0; i < list_size; ++i) {
        try {
            // Convert each array to Eigen::VectorXd
            Eigen::VectorXd CCx_sum = array_to_vector(CCx_loc_sums_list[i].cast<py::array_t<double>>());
            Eigen::VectorXd CCy_sum = array_to_vector(CCy_loc_sums_list[i].cast<py::array_t<double>>());
            Eigen::VectorXd feat_x = array_to_vector(feat_xs_list[i].cast<py::array_t<double>>());
            Eigen::VectorXd feat_y = array_to_vector(feat_ys_list[i].cast<py::array_t<double>>());

            CCx_loc_sums_vec.push_back(CCx_sum);
            CCy_loc_sums_vec.push_back(CCy_sum);
            feat_xs_vec.push_back(feat_x);
            feat_ys_vec.push_back(feat_y);
        }
        catch (const py::cast_error& e) {
            throw std::invalid_argument("All elements in input lists must be NumPy arrays of type float64.");
        }
    }

    // Initialize ThreadPool
    ThreadPool pool(num_workers);
    std::vector<std::future<std::pair<size_t, double>>> results;
    results.reserve(list_size);

    for (size_t i = 0; i < list_size; ++i) {
        // Capture by value to ensure thread safety
        size_t index = i;
        Eigen::VectorXd CCx_sum = CCx_loc_sums_vec[i];
        Eigen::VectorXd CCy_sum = CCy_loc_sums_vec[i];
        Eigen::VectorXd feat_x = feat_xs_vec[i];
        Eigen::VectorXd feat_y = feat_ys_vec[i];

        results.emplace_back(pool.enqueue([=]() -> std::pair<size_t, double> {
            double jaccard_index = jaccard_composite(CCx_sum, CCy_sum, feat_x, feat_y);
            return {index, jaccard_index};
        }));

        // Update progress
        if (progress_callback) {
            progress_callback();
        }
    }

    // Collect the results
    std::vector<double> output(list_size, 0.0);
    for (auto& result_future : results) {
        auto result_pair = result_future.get();
        size_t index = result_pair.first;
        double value = result_pair.second;
        output[index] = value;
    }

    return output;
}

// Expose to Python via Pybind11
PYBIND11_MODULE(parallelize, m) {  // Module name within the STopover package
    m.def("parallel_topological_comp", &parallel_topological_comp, "Parallelized topological_comp_res function",
          py::arg("locs"), py::arg("spatial_type") = "visium", py::arg("fwhm") = 2.5, 
          py::arg("feats"), py::arg("min_size") = 5, py::arg("thres_per") = 30, py::arg("return_mode") = "all", 
          py::arg("num_workers") = 4, py::arg("progress_callback"));

    m.def("parallel_jaccard_composite", &parallel_jaccard_composite_py, "Parallelized jaccard_composite function accepting lists of NumPy arrays",
          py::arg("CCx_loc_sums"), py::arg("CCy_loc_sums"), py::arg("feat_xs"), 
          py::arg("feat_ys"), py::arg("num_workers") = 4, py::arg("progress_callback"));
}