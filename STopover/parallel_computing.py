import numpy as np
import tqdm
from .parallelize import parallel_topological_comp, parallel_jaccard_composite

def default_log_callback(message):
    print(f"C++ Log: {message}", end='')  # 'end' to avoid adding extra newlines

def parallel_with_progress_topological_comp(
    locs, feats, spatial_type="visium", fwhm=2.5,
    min_size=5, thres_per=30, return_mode="all", num_workers=4,
    log_callback_func=None
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
        num_workers (int): Number of parallel workers (default: 4).
        log_callback_func (callable, optional): Function to handle log messages from C++.

    Returns:
        list: A list of topological components for each feature.
    """
    # Reshape feats to one-dimensional arrays
    feats = [feat.reshape(-1) if feat.ndim > 1 else feat for feat in feats]

    # Verify shapes
    for i, feat in enumerate(feats):
        if feat.ndim != 1:
            raise ValueError(f"feats[{i}] is not one-dimensional after reshape.")

    # Define a default log_callback if none is provided
    if log_callback_func is None:
        log_callback_func = default_log_callback

    # Initialize the overall result list
    total_tasks = len(locs)
    results = [None] * total_tasks  # Pre-allocate list for results

    # Create a progress bar
    with tqdm.tqdm(total=total_tasks) as pbar:
        try:
            # Call the C++ function for all tasks
            batch_results = parallel_topological_comp(
                locs=locs,
                spatial_type=spatial_type,
                fwhm=fwhm,
                feats=feats,
                min_size=min_size,
                thres_per=thres_per,
                return_mode=return_mode,
                num_workers=num_workers,  # 0 to auto-detect
                progress_callback=lambda: pbar.update(1),
                log_callback=log_callback_func
            )

            # Assign results
            for i, res in enumerate(batch_results):
                results[i] = res

        except Exception as e:
            log_callback_func(f"\nException during topological_comp: {e}\n")
            raise

    return results

def parallel_with_progress_jaccard_composite(
    CCx_loc_sums, CCy_loc_sums, feat_xs=None, feat_ys=None,
    jaccard_type="default", num_workers=4,
    log_callback_func=None
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
        num_workers (int): Number of parallel workers (default: 4).
        log_callback_func (callable, optional): Function to handle log messages from C++.

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
    if log_callback_func is None:
        log_callback_func = default_log_callback

    # Initialize the overall result list
    total_tasks = len(CCx_loc_sums)
    results = [None] * total_tasks  # Pre-allocate list for results

    # Create a progress bar
    with tqdm.tqdm(total=total_tasks) as pbar:
        try:
            # Call the C++ function for all tasks
            batch_results = parallel_jaccard_composite(
                CCx_loc_sums=CCx_loc_sums,
                CCy_loc_sums=CCy_loc_sums,
                feat_xs=feat_xs,
                feat_ys=feat_ys,
                jaccard_type=jaccard_type,
                num_workers=num_workers,  # 0 to auto-detect
                progress_callback=lambda: pbar.update(1),
                log_callback=log_callback_func
            )

            # Assign results
            for i, res in enumerate(batch_results):
                results[i] = res

        except Exception as e:
            log_callback_func(f"\nException during jaccard_composite: {e}\n")
            raise

    return results