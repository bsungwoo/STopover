#ifndef PARALLELIZE_H
#define PARALLELIZE_H

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <queue>
#include <vector>
#include <thread>
#include <functional>
#include <future>
#include <mutex>
#include <condition_variable>
#include "type_conversion.h"  // Include the conversion header
#include "topological_comp.h"
#include "jaccard.h"

namespace py = pybind11;

// Thread Pool Class
class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool();

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// Template function definition
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

// Function Declarations

// Parallel function for extract_adjacency_spatial
std::vector<std::tuple<Eigen::SparseMatrix<int>, Eigen::MatrixXd>> parallel_extract_adjacency(
    const std::vector<py::object>& locs,
    const std::string& spatial_type, double fwhm, int num_workers,
    py::function progress_callback);

// Parallel function for topological_comp_res
std::vector<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>> parallel_topological_comp(
    const std::vector<py::array_t<double>>& feats,
    const std::vector<py::object>& A_matrices,
    const std::vector<py::array_t<double>>& masks,
    const std::string& spatial_type, int min_size, int thres_per, const std::string& return_mode, int num_workers,
    py::function progress_callback);

// Parallel function for jaccard_composite
std::vector<double> parallel_jaccard_composite(
    const std::vector<py::array_t<double>>& CCx_loc_sums,
    const std::vector<py::array_t<double>>& CCy_loc_sums,
    const std::vector<py::array_t<double>>& feat_xs,
    const std::vector<py::array_t<double>>& feat_ys, int num_workers,
    py::function progress_callback);

#endif // PARALLELIZE_H