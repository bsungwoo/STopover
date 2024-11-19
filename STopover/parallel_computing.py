import numpy as np
import tqdm
import threading
from .parallelize import parallel_topological_comp, parallel_jaccard_composite

def default_log_callback(message):
    print(f"C++ Log: {message}", end='')  # 'end' to avoid adding extra newlines

def parallel_with_progress_topological_comp(
    locs, feats, spatial_type="visium", fwhm=2.5,
    min_size=5, thres_per=30, return_mode="all", num_workers=4,
    log_callback=None, max_tasks_in_queue=1000
):
    """
    Parallel computation for topological component extraction.
    Progress is shown using a tqdm progress bar.

    Args:
        locs (list): List of locations (NumPy arrays) to compute adjacency matrix for.
        feats (list): List of feature arrays (NumPy arrays).
        spatial_type (str): Type of spatial data.
        fwhm (float): Full-width half-maximum for Gaussian smoothing.
        min_size (int): Minimum size of connected component.
        thres_per (int): Percentile threshold for filtering connected components.
        return_mode (str): Return mode.
        num_workers (int): Number of parallel workers.
        log_callback (callable, optional): Function to handle log messages from C++.
        max_tasks_in_queue (int, optional): Maximum number of tasks allowed in the queue.

    Returns:
        list: A list of topological components for each feature.
    """
    # Reshape feats to one-dimensional arrays
    feats = [feat.reshape(-1) if feat.ndim > 1 else feat for feat in feats]
    
    # Optionally, verify shapes
    for i, feat in enumerate(feats):
        if feat.ndim != 1:
            raise ValueError(f"feats[{i}] is not one-dimensional after reshape.")
    
    # Define a default log_callback if none is provided
    if log_callback is None:
        log_callback = default_log_callback

    # Initialize a semaphore to limit the number of tasks in the queue
    semaphore = threading.Semaphore(max_tasks_in_queue)
    
    # Initialize a list to hold threading.Thread objects
    threads = []
    
    # Initialize a lock for thread-safe result assignment
    result_lock = threading.Lock()
    result = [None] * len(feats)  # Pre-allocate list for results

    # Create a progress bar
    with tqdm.tqdm(total=len(feats)) as pbar:
        # Define a Python callback function to update progress and release semaphore
        def update_progress(index):
            pbar.update(1)
            semaphore.release()  # Release semaphore to allow more tasks to be enqueued

        # Define a wrapper function to enqueue tasks
        def enqueue_task(index, loc, feat):
            try:
                semaphore.acquire()  # Acquire semaphore before enqueuing
                # Enqueue the task and get the future
                future = parallel_topological_comp(
                    locs=[loc],
                    spatial_type=spatial_type,
                    fwhm=fwhm,
                    feats=[feat],
                    min_size=min_size,
                    thres_per=thres_per,
                    return_mode=return_mode,
                    num_workers=1,  # Use single worker per enqueue to prevent over-subscription
                    progress_callback=lambda: update_progress(index),
                    log_callback=log_callback
                )
                # Retrieve the result
                topo_result = future[0]  # Since we passed a single task
                with result_lock:
                    result[index] = topo_result
            except Exception as e:
                log_callback(f"\nException during task {index}: {e}\n")
                semaphore.release()  # Ensure semaphore is released even on exception

        # Launch threads to enqueue tasks
        for index, (loc, feat) in enumerate(zip(locs, feats)):
            thread = threading.Thread(target=enqueue_task, args=(index, loc, feat))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    return result


def parallel_with_progress_jaccard_composite(
    CCx_loc_sums, CCy_loc_sums, feat_xs=None, feat_ys=None,
    jaccard_type="default", num_workers=4,
    log_callback=None, max_tasks_in_queue=1000
):
    """
    Parallel computation for Jaccard composite index.
    Progress is shown using a tqdm progress bar.

    Args:
        CCx_loc_sums (list or np.ndarray): List of NumPy arrays for connected component location sums for feature x.
        CCy_loc_sums (list or np.ndarray): List of NumPy arrays for connected component location sums for feature y.
        feat_xs (list or np.ndarray, optional): List of NumPy arrays for feature x values.
        feat_ys (list or np.ndarray, optional): List of NumPy arrays for feature y values.
        jaccard_type (str, optional): Type of Jaccard index to calculate. Either "default" or "weighted".
        num_workers (int): Number of parallel workers.
        log_callback (callable, optional): Function to handle log messages from C++.
        max_tasks_in_queue (int, optional): Maximum number of tasks allowed in the queue.

    Returns:
        list: A list of Jaccard composite indices.
    """
    # If feat_xs or feat_ys are not provided, initialize them with zero arrays matching the length
    if feat_xs is None:
        feat_xs = [np.array([0.0]) for _ in range(len(CCx_loc_sums))]
    if feat_ys is None:
        feat_ys = [np.array([0.0]) for _ in range(len(CCx_loc_sums))]
    
    # Ensure that all input lists have the same length
    if not (len(CCx_loc_sums) == len(CCy_loc_sums) == len(feat_xs) == len(feat_ys)):
        raise ValueError("All input lists must have the same length.")
    
    # Convert inputs to lists of NumPy arrays if they are not already
    if isinstance(CCx_loc_sums, np.ndarray):
        CCx_loc_sums = CCx_loc_sums.tolist()
    if isinstance(CCy_loc_sums, np.ndarray):
        CCy_loc_sums = CCy_loc_sums.tolist()
    if isinstance(feat_xs, np.ndarray):
        feat_xs = feat_xs.tolist()
    if isinstance(feat_ys, np.ndarray):
        feat_ys = feat_ys.tolist()
    
    # Define a default log_callback if none is provided
    if log_callback is None:
        log_callback = default_log_callback

    # Initialize a semaphore to limit the number of tasks in the queue
    semaphore = threading.Semaphore(max_tasks_in_queue)
    
    # Initialize a list to hold threading.Thread objects
    threads = []
    
    # Initialize a lock for thread-safe result assignment
    result_lock = threading.Lock()
    result = [None] * len(CCx_loc_sums)  # Pre-allocate list for results

    # Create a progress bar
    with tqdm.tqdm(total=len(CCx_loc_sums)) as pbar:
        # Define a Python callback function to update progress and release semaphore
        def update_progress(index):
            pbar.update(1)
            semaphore.release()  # Release semaphore to allow more tasks to be enqueued

        # Define a wrapper function to enqueue tasks
        def enqueue_task(index, CCx_sum, CCy_sum, feat_x, feat_y):
            try:
                semaphore.acquire()  # Acquire semaphore before enqueuing
                # Enqueue the task and get the future
                future = parallel_jaccard_composite(
                    CCx_loc_sums=[CCx_sum],
                    CCy_loc_sums=[CCy_sum],
                    feat_xs=[feat_x],
                    feat_ys=[feat_y],
                    jaccard_type=jaccard_type,
                    num_workers=1,  # Use single worker per enqueue to prevent over-subscription
                    progress_callback=lambda: update_progress(index),
                    log_callback=log_callback
                )
                # Retrieve the result
                jaccard_result = future[0]  # Since we passed a single task
                with result_lock:
                    result[index] = jaccard_result
            except Exception as e:
                log_callback(f"\nException during task {index}: {e}\n")
                semaphore.release()  # Ensure semaphore is released even on exception

        # Launch threads to enqueue tasks
        for index, (CCx_sum, CCy_sum, feat_x, feat_y) in enumerate(zip(CCx_loc_sums, CCy_loc_sums, feat_xs, feat_ys)):
            thread = threading.Thread(target=enqueue_task, args=(index, CCx_sum, CCy_sum, feat_x, feat_y))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    return result


# Example Usage:
# Assuming you have the locs, feats, adjacency matrices, masks, and other inputs properly formatted.
# locs = [numpy arrays for locations]
# feats = [numpy arrays for features]
# A_matrices = [scipy sparse matrices for adjacency matrices]
# masks = [numpy arrays for masks]
# CCx_loc_sums = [numpy arrays for CCx locations]
# CCy_loc_sums = [numpy arrays for CCy locations]

# Example calls:
# result_extract_adjacency = parallel_with_progress_extract_adjacency(locs)
# result_topological_comp = parallel_with_progress_topological_comp(locs, feats)
# result_jaccard_composite = parallel_with_progress_jaccard_composite(CCx_loc_sums, CCy_loc_sums)