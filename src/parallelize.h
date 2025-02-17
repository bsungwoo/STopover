#ifndef PARALLELIZE_H
#define PARALLELIZE_H

#include <future>
#include <vector>
#include <tuple>
#include <string>
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace py = pybind11;

// ---------------------------------------------------------------------
// ThreadPool Class Declaration
// ---------------------------------------------------------------------
class ThreadPool {
public:
    // Constructor: spawns a given number of worker threads.
    ThreadPool(size_t threads);

    // Destructor: joins all threads.
    ~ThreadPool();

    // Enqueue a task into the thread pool; returns a future.
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    // Worker threads.
    std::vector<std::thread> workers;
    // Task queue.
    std::queue<std::function<void()>> tasks;
    // Synchronization.
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// Inline implementation of the template enqueue method.
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        // Wrap the task in a lambda to catch exceptions.
        [func = std::forward<F>(f), ... args = std::forward<Args>(args)]() -> return_type {
            try {
                return func(args...);
            } catch (const std::exception& e) {
                // Optionally log or handle the exception.
                throw;  // Rethrow the exception.
            }
        }
    );
    
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

// ---------------------------------------------------------------------
// Parallel Function Declarations
// ---------------------------------------------------------------------

// Parallelized extract_adjacency_spatial function.
// Accepts a vector of Python objects (each convertible to Eigen::MatrixXd),
// the spatial type, fwhm value, number of worker threads, and an optional progress callback.
std::vector<std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd>> parallel_extract_adjacency(
    const std::vector<py::object>& locs, 
    const std::string& spatial_type, double fwhm, int num_workers,
    py::function progress_callback);

// Parallelized topological_comp_res function.
// Accepts vectors of features (as py::array_t<double>), SciPy sparse matrices (as py::object),
// masks (as py::array_t<double>), and other parameters.
std::vector<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>> parallel_topological_comp(
    const std::vector<py::array_t<double>>& feats,  
    const std::vector<py::object>& A_matrices,      
    const std::vector<py::array_t<double>>& masks,
    const std::string& spatial_type, int min_size, int thres_per, const std::string& return_mode, int num_workers,
    py::function progress_callback);

// Parallelized jaccard_composite function.
// Accepts vectors of NumPy arrays for connected component locations and feature values.
std::vector<double> parallel_jaccard_composite(
    const std::vector<py::array_t<double>>& CCx_loc_sums, 
    const std::vector<py::array_t<double>>& CCy_loc_sums,
    const std::vector<py::array_t<double>>& feat_xs, 
    const std::vector<py::array_t<double>>& feat_ys, 
    int num_workers,
    py::function progress_callback);

#endif // PARALLELIZE_H