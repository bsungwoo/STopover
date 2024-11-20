import numpy as np
import tqdm
import gc
from .parallelize import parallel_topological_comp_omp, parallel_jaccard_composite

def default_log_callback(message):
    print(f"C++ Log: {message}", end='')  # 'end' to avoid adding extra newlines

def parallel_with_progress_topological_comp(
    locs, feats, spatial_type="visium", fwhm=2.5,
    min_size=5, thres_per=30, return_mode="all", num_workers=0,
    log_callback_func=None
):
    """
    Parallel computation for topological component extraction using OpenMP.
    
    Args:
        locs (list): List of locations (NumPy arrays).
        feats (list): List of feature arrays (NumPy arrays).
        spatial_type (str): Type of spatial data.
        fwhm (float): Full-width half-maximum for Gaussian smoothing.
        min_size (int): Minimum size of connected component.
        thres_per (int): Percentile threshold for filtering connected components.
        return_mode (str): Return mode.
        num_workers (int): Number of parallel workers (0 to auto-detect).
        log_callback_func (callable, optional): Function to handle log messages from C++.
    
    Returns:
        list: A list of topological components for each feature.
    """
    # Assign log_callback_func early to ensure it's not None during exception handling
    if log_callback_func is None:
        log_callback_func = default_log_callback

    # Reshape feats to one-dimensional arrays if necessary
    feats = [feat.reshape(-1) if feat.ndim > 1 else feat for feat in feats]

    # Verify shapes
    for i, feat in enumerate(feats):
        if feat.ndim != 1:
            log_callback_func(f"Error: feats[{i}] is not one-dimensional after reshape.\n")
            raise ValueError(f"feats[{i}] is not one-dimensional after reshape.")

    try:
        # Convert all NumPy arrays to contiguous arrays of type float64
        locs = [np.ascontiguousarray(loc, dtype=np.float64) for loc in locs]
        feats = [np.ascontiguousarray(feat, dtype=np.float64) for feat in feats]
    except Exception as e:
        log_callback_func(f"Error during conversion to contiguous arrays: {e}\n")
        raise

    # Total number of tasks
    total_tasks = len(locs)

    # Initialize the progress bar
    with tqdm.tqdm(total=total_tasks, desc="Processing Topological Components") as pbar:
        # Define a progress callback function that updates the progress bar
        def progress_callback(n):
            pbar.update(n)

        try:
            # Call the C++ function, which handles parallelism with OpenMP
            output = parallel_topological_comp_omp(
                locs_eigen=locs,
                feats_eigen=feats,
                spatial_type=spatial_type,
                fwhm=fwhm,
                min_size=min_size,
                thres_per=thres_per,
                return_mode=return_mode,
                num_workers=num_workers,
                progress_callback=progress_callback,
                log_callback=log_callback_func
            )
        except Exception as e:
            log_callback_func(f"Error during topological_comp computation: {e}\n")
            raise

    # Optionally, trigger garbage collection
    gc.collect()

    return output


def parallel_with_progress_jaccard_composite(
    CCx_loc_sums, CCy_loc_sums, feat_xs=None, feat_ys=None,
    jaccard_type="default", num_workers=0,
    log_callback_func=None
):
    """
    Parallel computation for Jaccard composite index.
    Delegates internal batching to C++.
    
    Args:
        CCx_loc_sums (list or np.ndarray): List of NumPy arrays for connected component location sums for feature x.
        CCy_loc_sums (list or np.ndarray): List of NumPy arrays for connected component location sums for feature y.
        feat_xs (list or np.ndarray, optional): List of NumPy arrays for feature x values.
        feat_ys (list or np.ndarray, optional): List of NumPy arrays for feature y values.
        jaccard_type (str, optional): Type of Jaccard index to calculate. Either "default" or "weighted".
        num_workers (int): Number of parallel workers (0 to auto-detect).
        log_callback_func (callable, optional): Function to handle log messages from C++.
    
    Returns:
        list: A list of Jaccard composite indices.
    """
    # Assign log_callback_func early to ensure it's not None during exception handling
    if log_callback_func is None:
        log_callback_func = default_log_callback

    # Initialize feat_xs and feat_ys if not provided
    if feat_xs is None:
        feat_xs = [np.array([0.0], dtype=np.float64) for _ in range(len(CCx_loc_sums))]
    if feat_ys is None:
        feat_ys = [np.array([0.0], dtype=np.float64) for _ in range(len(CCy_loc_sums))]

    # Validate input lengths
    if not (len(CCx_loc_sums) == len(CCy_loc_sums) == len(feat_xs) == len(feat_ys)):
        log_callback_func("Error: All input lists must have the same length.\n")
        raise ValueError("All input lists must have the same length.")

    try:
        # Convert all NumPy arrays to contiguous arrays of type float64
        CCx_loc_sums = [np.ascontiguousarray(arr, dtype=np.float64) for arr in CCx_loc_sums]
        CCy_loc_sums = [np.ascontiguousarray(arr, dtype=np.float64) for arr in CCy_loc_sums]
        feat_xs = [np.ascontiguousarray(arr, dtype=np.float64) for arr in feat_xs]
        feat_ys = [np.ascontiguousarray(arr, dtype=np.float64) for arr in feat_ys]
    except Exception as e:
        log_callback_func(f"Error during conversion to contiguous arrays: {e}\n")
        raise

    # Total number of tasks
    total_tasks = len(CCx_loc_sums)

    # Initialize the progress bar
    with tqdm.tqdm(total=total_tasks, desc="Processing Jaccard Composite") as pbar:
        # Define a progress callback function that updates the progress bar
        def progress_callback(n):
            pbar.update(n)

        try:
            # Call the C++ function, which handles parallelism with OpenMP
            output_j = parallel_jaccard_composite(
                CCx_loc_sums=CCx_loc_sums,
                CCy_loc_sums=CCy_loc_sums,
                feat_xs=feat_xs,
                feat_ys=feat_ys,
                jaccard_type=jaccard_type,
                num_workers=num_workers,
                progress_callback=progress_callback,
                log_callback=log_callback_func
            )
        except Exception as e:
            log_callback_func(f"Error during jaccard_composite computation: {e}\n")
            raise

    # Optionally, trigger garbage collection
    gc.collect()

    return output_j