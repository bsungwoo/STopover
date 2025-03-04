import numpy as np
import tqdm
import gc
from .parallelize import parallel_topological_comp, parallel_jaccard_composite
import datetime

def default_log_callback(message):
    """Enhanced log callback that writes to both console and file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] C++ Log: {message}"
    
    # Print to console
    print(formatted_message)
    
    # Also write to a log file
    with open("stopover_parallel.log", "a") as log_file:
        log_file.write(formatted_message + "\n")

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
    
    log_callback_func(f"Starting parallel_with_progress_topological_comp with {len(locs)} items")
    log_callback_func(f"Parameters: spatial_type={spatial_type}, fwhm={fwhm}, min_size={min_size}, "
                     f"thres_per={thres_per}, return_mode={return_mode}, num_workers={num_workers}")

    # Reshape feats to one-dimensional arrays if necessary
    feats = [feat.reshape(-1) if feat.ndim > 1 else feat for feat in feats]

    # Verify shapes
    for i, feat in enumerate(feats):
        if feat.ndim != 1:
            error_msg = f"Error: feats[{i}] is not one-dimensional after reshape."
            log_callback_func(error_msg)
            raise ValueError(error_msg)

    try:
        # Convert all NumPy arrays to contiguous arrays of type float64
        log_callback_func("Converting arrays to contiguous float64 format")
        locs = [np.ascontiguousarray(loc, dtype=np.float64) for loc in locs]
        feats = [np.ascontiguousarray(feat, dtype=np.float64) for feat in feats]
        
        # Log shapes for debugging
        for i, (loc, feat) in enumerate(zip(locs, feats)):
            log_callback_func(f"Item {i}: loc shape={loc.shape}, feat shape={feat.shape}")
            
    except Exception as e:
        error_msg = f"Error during conversion to contiguous arrays: {e}"
        log_callback_func(error_msg)
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
            output = parallel_topological_comp(
                locs=locs,
                feats=feats,
                spatial_type=spatial_type,
                fwhm=fwhm,
                min_size=min_size,
                thres_per=thres_per,
                return_mode=return_mode,
                num_workers=num_workers,
                progress_callback=progress_callback,
                log_callback=None
            )
        except Exception as e:
            error_msg = f"Error during topological_comp computation: {e}"
            log_callback_func(error_msg)
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
                log_callback=None
            )
        except Exception as e:
            log_callback_func(f"Error during jaccard_composite computation: {e}\n")
            raise

    # Optionally, trigger garbage collection
    gc.collect()

    return output_j