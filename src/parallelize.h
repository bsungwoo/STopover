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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace py = pybind11;

// ThreadPool class definition
class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool();  // Destructor declaration

    // Enqueue a task
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    // Worker threads
    std::vector<std::thread> workers;

    // Task queue
    std::queue<std::function<void()>> tasks;

    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// Implementation of ThreadPool::enqueue (inline for template)
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

// Parallel function for topological_comp_res
std::vector<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<double>>> parallel_topological_comp(
    const std::vector<py::object>& locs, 
    const std::string& spatial_type, double fwhm,
    const std::vector<py::array_t<double>>& feats,  
    int min_size, int thres_per, const std::string& return_mode, int num_workers,
    py::function progress_callback);

// Parallel function for jaccard_composite
std::vector<double> parallel_jaccard_composite(
    const std::vector<py::array_t<double>>& CCx_loc_sums, 
    const std::vector<py::array_t<double>>& CCy_loc_sums,
    const std::vector<py::array_t<double>>& feat_xs, 
    const std::vector<py::array_t<double>>& feat_ys, 
    int num_workers,
    py::function progress_callback);

#endif // PARALLELIZE_H