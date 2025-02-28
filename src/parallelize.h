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
#include <pybind11/stl.h>
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
        -> std::future<typename std::result_of<F(Args...)>::type>;

    // Add a method to get the number of active tasks
    size_t active_tasks() const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return tasks.size();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
    std::atomic<size_t> active_count{0}; // Track active tasks
};

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        [func = std::forward<F>(f), ... args = std::forward<Args>(args)]() -> return_type {
            try {
                return func(args...);
            } catch (const std::exception &e) {
                // Optionally log the error message: e.what()
                throw; // rethrow the exception
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
    const std::vector<py::array_t<double>>& locs,
    const std::string& spatial_type,
    double fwhm,
    int num_workers,
    py::function progress_callback);

// Parallelized topological_comp_res function.
std::vector<py::object> parallel_topological_comp(
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
    const std::vector<py::array_t<int>>& cc_1_list,
    const std::vector<py::array_t<int>>& cc_2_list,
    const std::string& jaccard_type,
    int num_workers,
    py::function progress_callback);

#endif // PARALLELIZE_H