#ifndef PARALLELIZE_H
#define PARALLELIZE_H

#include <future>
#include <vector>
#include <tuple>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>

// ThreadPool class definition
class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool();

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

// Function declarations with consistent types

// Parallel function for extract_adjacency_spatial
std::vector<std::tuple<Eigen::SparseMatrix<double>, Eigen::MatrixXd>> parallel_extract_adjacency(
    const std::vector<pybind11::object>& locs, 
    const std::string& spatial_type, double fwhm, int num_workers,
    pybind11::function progress_callback);

// Parallel function for topological_comp_res
std::vector<std::tuple<std::vector<std::vector<int>>, Eigen::SparseMatrix<int>>> parallel_topological_comp(
    const std::vector<pybind11::array_t<double>>& feats,  
    const std::vector<pybind11::object>& A_matrices,      
    const std::vector<pybind11::array_t<double>>& masks,
    const std::string& spatial_type, int min_size, int thres_per, const std::string& return_mode, int num_workers,
    pybind11::function progress_callback);

// Parallel function for jaccard_composite
std::vector<double> parallel_jaccard_composite(
    const std::vector<pybind11::array_t<double>>& CCx_loc_sums, 
    const std::vector<pybind11::array_t<double>>& CCy_loc_sums,
    const std::vector<pybind11::array_t<double>>& feat_xs, 
    const std::vector<pybind11::array_t<double>>& feat_ys, 
    int num_workers,
    pybind11::function progress_callback);

// Include the implementations for ThreadPool.enqueue
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

#endif // PARALLELIZE_H