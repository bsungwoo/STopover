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
    explicit ThreadPool(size_t threads);
    ~ThreadPool();

    // Enqueue a task into the thread pool; returns a future.
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result_t<F, Args...>>;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic_bool stop;
};

// Inline implementation of the template enqueue method.
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result_t<F, Args...>>
{
    using return_type = typename std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        // Wrap the task in a lambda that catches exceptions.
        [func = std::forward<F>(f), ... args = std::forward<Args>(args)]() -> return_type {
            try {
                return func(args...);
            } catch (const std::exception &e) {
                // You can log the exception message here if desired.
                throw; // rethrow after catching
            }
        }
    );
    std::future<return_type> res = task->get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (stop.load())
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
std::vector<std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd>> parallel_extract_adjacency(
    const std::vector<py::object>& locs,
    const std::string& spatial_type,
    double fwhm,
    int num_workers,
    py::function progress_callback);

// Parallelized topological_comp_res function.
std::vector<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>> parallel_topological_comp(
    const std::vector<py::array_t<double>>& feats,
    const std::vector<py::object>& A_matrices,
    const std::vector<py::array_t<double>>& masks,
    const std::string& spatial_type,
    int min_size,
    int thres_per,
    const std::string& return_mode,
    int num_workers,
    py::function progress_callback);

// Parallelized jaccard_composite function.
std::vector<double> parallel_jaccard_composite(
    const std::vector<py::array_t<double>>& CCx_loc_sums,
    const std::vector<py::array_t<double>>& CCy_loc_sums,
    const std::vector<py::array_t<double>>& feat_xs,
    const std::vector<py::array_t<double>>& feat_ys,
    int num_workers,
    py::function progress_callback);

#endif // PARALLELIZE_H