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
#include <fstream>
#include <chrono>
#include <iomanip>

namespace py = pybind11;

// Create a thread-safe logging function
std::mutex log_mutex;
void log_message(const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex);
    std::ofstream log_file("parallelize_debug.log", std::ios_base::app);
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    log_file << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") 
             << " - " << message << std::endl;
}

// ---------------------------------------------------------------------
// ThreadPool Implementation
// ---------------------------------------------------------------------
ThreadPool::ThreadPool(size_t threads) : stop(false) {
    log_message("Creating ThreadPool with " + std::to_string(threads) + " threads");
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this, i] {
            log_message("Thread " + std::to_string(i) + " started");
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this]{
                        return this->stop.load() || !this->tasks.empty();
                    });
                    if (this->stop.load() && this->tasks.empty()) {
                        log_message("Thread " + std::to_string(i) + " stopping");
                        return;
                    }
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                this->active_count++;
                try {
                    log_message("Thread " + std::to_string(i) + " executing task");
                    task();
                    log_message("Thread " + std::to_string(i) + " completed task");
                } catch (const std::exception &e) {
                    log_message("ERROR in thread " + std::to_string(i) + ": " + e.what());
                } catch (...) {
                    log_message("ERROR in thread " + std::to_string(i) + ": Unknown exception");
                }
                this->active_count--;
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
    log_message("Starting parallel_extract_adjacency with " + std::to_string(locs.size()) + 
                " locations, spatial_type=" + spatial_type + 
                ", fwhm=" + std::to_string(fwhm) + 
                ", num_workers=" + std::to_string(num_workers));
    
    // Reduce number of workers to avoid potential resource issues
    num_workers = std::max(1, std::min(num_workers, 4));
    log_message("Using " + std::to_string(num_workers) + " workers");
    
    ThreadPool pool(num_workers);
    std::vector<std::future<std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd>>> results;
    std::atomic<int> count{0};

    for (size_t loc_idx = 0; loc_idx < locs.size(); ++loc_idx) {
        log_message("Processing location " + std::to_string(loc_idx));
        
        // First check if the Python object is valid
        if (locs[loc_idx].is_none()) {
            log_message("ERROR: Location " + std::to_string(loc_idx) + " is None");
            results.emplace_back(std::async(std::launch::deferred, []() {
                return std::make_tuple(Eigen::SparseMatrix<double>(), Eigen::MatrixXd());
            }));
            continue;
        }
        
        try {
            // Try to extract numpy array info before casting
            log_message("Checking location " + std::to_string(loc_idx) + " type");
            
            // Check if it's a numpy array
            bool is_array = py::isinstance<py::array>(locs[loc_idx]);
            log_message("Location " + std::to_string(loc_idx) + " is " + 
                       (is_array ? "a numpy array" : "not a numpy array"));
            
            if (is_array) {
                py::array arr = locs[loc_idx].cast<py::array>();
                log_message("Array info: shape=(" + 
                           std::to_string(arr.shape(0)) + "," + 
                           (arr.ndim() > 1 ? std::to_string(arr.shape(1)) : "1") + 
                           "), dtype=" + std::string(py::str(arr.dtype())));
            }
            
            // Make a copy of the location to avoid potential reference issues
            py::object loc_copy = locs[loc_idx];
            
            // Process in a separate thread with proper error handling
            results.emplace_back(pool.enqueue([loc_copy, spatial_type, fwhm, loc_idx]() {
                try {
                    log_message("Thread: Converting location " + std::to_string(loc_idx));
                    
                    // Release the GIL during heavy computation
                    py::gil_scoped_release release;
                    
                    // Convert to Eigen matrix
                    Eigen::MatrixXd loc;
                    {
                        // Reacquire GIL for the conversion
                        py::gil_scoped_acquire acquire;
                        loc = loc_copy.cast<Eigen::MatrixXd>();
                    }
                    
                    log_message("Thread: Successfully converted location " + std::to_string(loc_idx) + 
                               " with shape (" + std::to_string(loc.rows()) + 
                               ", " + std::to_string(loc.cols()) + ")");
                    
                    // Validate the location matrix
                    if (loc.rows() == 0 || loc.cols() != 2) {
                        log_message("Thread: ERROR: Invalid location matrix shape: expected (n, 2), got (" + 
                                   std::to_string(loc.rows()) + ", " + 
                                   std::to_string(loc.cols()) + ")");
                        return std::make_tuple(Eigen::SparseMatrix<double>(), Eigen::MatrixXd());
                    }
                    
                    log_message("Thread: Starting extract_adjacency_spatial for location " + 
                               std::to_string(loc_idx));
                    auto result = extract_adjacency_spatial(loc, spatial_type, fwhm);
                    log_message("Thread: Completed extract_adjacency_spatial for location " + 
                               std::to_string(loc_idx));
                    return result;
                } catch (const std::exception& e) {
                    log_message("Thread: ERROR in processing location " + 
                               std::to_string(loc_idx) + ": " + std::string(e.what()));
                    // Return empty matrices instead of crashing
                    return std::make_tuple(Eigen::SparseMatrix<double>(), Eigen::MatrixXd());
                } catch (...) {
                    log_message("Thread: UNKNOWN ERROR in processing location " + 
                               std::to_string(loc_idx));
                    return std::make_tuple(Eigen::SparseMatrix<double>(), Eigen::MatrixXd());
                }
            }));
            
            log_message("Enqueued location " + std::to_string(loc_idx));
        } catch (const std::exception& e) {
            log_message("ERROR: Exception during enqueue for location " + 
                       std::to_string(loc_idx) + ": " + std::string(e.what()));
            // Add a placeholder future with empty result
            results.emplace_back(std::async(std::launch::deferred, []() {
                return std::make_tuple(Eigen::SparseMatrix<double>(), Eigen::MatrixXd());
            }));
        } catch (...) {
            log_message("ERROR: Unknown exception during enqueue for location " + 
                       std::to_string(loc_idx));
            results.emplace_back(std::async(std::launch::deferred, []() {
                return std::make_tuple(Eigen::SparseMatrix<double>(), Eigen::MatrixXd());
            }));
        }
        
        if (progress_callback && (++count % 10 == 0)) {
            try {
                py::gil_scoped_acquire acquire;
                progress_callback();
                log_message("Called progress_callback after " + std::to_string(count) + " locations");
            } catch (const std::exception& e) {
                log_message("ERROR in progress callback: " + std::string(e.what()));
            } catch (...) {
                log_message("UNKNOWN ERROR in progress callback");
            }
        }
    }
    
    log_message("All locations enqueued, collecting results");
    std::vector<std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd>> output;
    output.reserve(results.size());
    
    for (size_t i = 0; i < results.size(); ++i) {
        try {
            log_message("Getting result for index " + std::to_string(i));
            output.emplace_back(results[i].get());
            log_message("Successfully got result for index " + std::to_string(i));
        } catch (const std::exception& e) {
            log_message("ERROR getting result at index " + std::to_string(i) + 
                       ": " + std::string(e.what()));
            output.emplace_back(std::make_tuple(Eigen::SparseMatrix<double>(), Eigen::MatrixXd()));
        } catch (...) {
            log_message("UNKNOWN ERROR getting result at index " + std::to_string(i));
            output.emplace_back(std::make_tuple(Eigen::SparseMatrix<double>(), Eigen::MatrixXd()));
        }
    }
    
    log_message("Completed parallel_extract_adjacency with " + std::to_string(output.size()) + " results");
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
    log_message("Starting parallel_topological_comp with " + std::to_string(feats.size()) + 
                " features, spatial_type=" + spatial_type + 
                ", min_size=" + std::to_string(min_size) + 
                ", thres_per=" + std::to_string(thres_per) + 
                ", return_mode=" + return_mode + 
                ", num_workers=" + std::to_string(num_workers));
    
    // Limit number of workers to avoid resource issues
    num_workers = std::max(1, std::min(num_workers, 4));
    log_message("Using " + std::to_string(num_workers) + " workers");
    
    ThreadPool pool(num_workers);
    std::vector<std::future<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>>> results;
    std::atomic<int> count{0};

    for (size_t i = 0; i < feats.size(); ++i) {
        log_message("Processing feature " + std::to_string(i));
        
        try {
            // Make copies of the Python objects to avoid reference issues
            py::array_t<double> feat_copy = feats[i];
            py::object A_matrix_copy = A_matrices[i];
            py::array_t<double> mask_copy = masks[i];
            
            results.emplace_back(pool.enqueue([feat_copy, A_matrix_copy, mask_copy, 
                                              spatial_type, min_size, thres_per, return_mode, i]() {
                try {
                    log_message("Thread: Converting feature " + std::to_string(i));
                    
                    // Release the GIL during heavy computation
                    py::gil_scoped_release release;
                    
                    // Convert to Eigen types
                    Eigen::VectorXd feat;
                    Eigen::SparseMatrix<double> A_matrix;
                    Eigen::MatrixXd mask;
                    
                    {
                        // Reacquire GIL for the conversion
                        py::gil_scoped_acquire acquire;
                        
                        // Convert feature
                        feat = feat_copy.cast<Eigen::VectorXd>();
                        log_message("Thread: Converted feature " + std::to_string(i) + 
                                   " with size " + std::to_string(feat.size()));
                        
                        // Convert adjacency matrix
                        try {
                            A_matrix = scipy_sparse_to_eigen_sparse(A_matrix_copy);
                            log_message("Thread: Converted adjacency matrix " + std::to_string(i) + 
                                       " with size (" + std::to_string(A_matrix.rows()) + 
                                       ", " + std::to_string(A_matrix.cols()) + ")");
                        } catch (const std::exception& e) {
                            log_message("Thread: ERROR converting adjacency matrix " + std::to_string(i) + 
                                       ": " + std::string(e.what()));
                            throw;
                        }
                        
                        // Convert mask
                        try {
                            mask = mask_copy.cast<Eigen::MatrixXd>();
                            log_message("Thread: Converted mask " + std::to_string(i) + 
                                       " with shape (" + std::to_string(mask.rows()) + 
                                       ", " + std::to_string(mask.cols()) + ")");
                        } catch (const std::exception& e) {
                            log_message("Thread: ERROR converting mask " + std::to_string(i) + 
                                       ": " + std::string(e.what()));
                            throw;
                        }
                    }
                    
                    // Validate inputs
                    if (feat.size() == 0) {
                        log_message("Thread: ERROR: Empty feature vector for index " + std::to_string(i));
                        return std::make_tuple(std::vector<std::vector<int>>(), Eigen::SparseMatrix<int>());
                    }
                    
                    if (A_matrix.rows() == 0 || A_matrix.cols() == 0) {
                        log_message("Thread: ERROR: Empty adjacency matrix for index " + std::to_string(i));
                        return std::make_tuple(std::vector<std::vector<int>>(), Eigen::SparseMatrix<int>());
                    }
                    
                    // Call topological_comp_res with proper error handling
                    log_message("Thread: Starting topological_comp_res for feature " + std::to_string(i));
                    auto result = topological_comp_res(feat, A_matrix, mask, spatial_type, min_size, thres_per, return_mode);
                    log_message("Thread: Completed topological_comp_res for feature " + std::to_string(i));
                    return result;
                } catch (const std::exception& e) {
                    log_message("Thread: ERROR in processing feature " + std::to_string(i) + 
                               ": " + std::string(e.what()));
                    return std::make_tuple(std::vector<std::vector<int>>(), Eigen::SparseMatrix<int>());
                } catch (...) {
                    log_message("Thread: UNKNOWN ERROR in processing feature " + std::to_string(i));
                    return std::make_tuple(std::vector<std::vector<int>>(), Eigen::SparseMatrix<int>());
                }
            }));
            
            log_message("Enqueued feature " + std::to_string(i));
        } catch (const std::exception& e) {
            log_message("ERROR: Exception during enqueue for feature " + std::to_string(i) + 
                       ": " + std::string(e.what()));
            results.emplace_back(std::async(std::launch::deferred, []() {
                return std::make_tuple(std::vector<std::vector<int>>(), Eigen::SparseMatrix<int>());
            }));
        } catch (...) {
            log_message("ERROR: Unknown exception during enqueue for feature " + std::to_string(i));
            results.emplace_back(std::async(std::launch::deferred, []() {
                return std::make_tuple(std::vector<std::vector<int>>(), Eigen::SparseMatrix<int>());
            }));
        }
        
        if (progress_callback && (++count % 10 == 0)) {
            try {
                py::gil_scoped_acquire acquire;
                progress_callback();
                log_message("Called progress_callback after " + std::to_string(count) + " features");
            } catch (const std::exception& e) {
                log_message("ERROR in progress callback: " + std::string(e.what()));
            } catch (...) {
                log_message("UNKNOWN ERROR in progress callback");
            }
        }
    }
    
    log_message("All features enqueued, collecting results");
    std::vector<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>> output;
    output.reserve(results.size());
    
    for (size_t i = 0; i < results.size(); ++i) {
        try {
            log_message("Getting result for feature " + std::to_string(i));
            output.emplace_back(results[i].get());
            log_message("Successfully got result for feature " + std::to_string(i));
        } catch (const std::exception& e) {
            log_message("ERROR getting result for feature " + std::to_string(i) + 
                       ": " + std::string(e.what()));
            output.emplace_back(std::make_tuple(std::vector<std::vector<int>>(), Eigen::SparseMatrix<int>()));
        } catch (...) {
            log_message("UNKNOWN ERROR getting result for feature " + std::to_string(i));
            output.emplace_back(std::make_tuple(std::vector<std::vector<int>>(), Eigen::SparseMatrix<int>()));
        }
    }
    
    log_message("Completed parallel_topological_comp with " + std::to_string(output.size()) + " results");
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