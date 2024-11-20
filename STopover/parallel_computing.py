import numpy as np
import tqdm
import gc
from .parallelize import parallel_topological_comp, parallel_jaccard_composite

def default_log_callback(message):
    print(f"C++ Log: {message}", end='')  # 'end' to avoid adding extra newlines

def create_batches(data, batch_size):
    """
    Splits the data into smaller batches.

    Args:
        data (list): The data to split into batches.
        batch_size (int): The number of items per batch.

    Yields:
        list: A batch of data items.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def parallel_with_progress_topological_comp(
    locs, feats, spatial_type="visium", fwhm=2.5,
    min_size=5, thres_per=30, return_mode="all", num_workers=0,
    log_callback_func=None, batch_size=500
):
    """
    Batched parallel computation for topological component extraction.
    Processes each batch sequentially, waiting for one batch to complete before moving to the next.

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
        batch_size (int, optional): Maximum number of tasks per batch.

    Returns:
        list: A list of topological components for each feature.
    """
    # Reshape feats to one-dimensional arrays
    feats = [feat.reshape(-1) if feat.ndim > 1 else feat for feat in feats]

    # Verify shapes
    for i, feat in enumerate(feats):
        if feat.ndim != 1:
            raise ValueError(f"feats[{i}] is not one-dimensional after reshape.")

   # Assign log_callback_func early to ensure it's not None during exception handling
    if log_callback_func is None:
        log_callback_func = default_log_callback

    # Create batches
    batches_locs = list(create_batches(locs, batch_size))
    batches_feats = list(create_batches(feats, batch_size))

    # Initialize result container
    total_tasks = len(locs)
    results = [None] * total_tasks

    # Process batches sequentially
    with tqdm.tqdm(total=total_tasks) as pbar:
        for batch_idx, (batch_locs, batch_feats) in enumerate(zip(batches_locs, batches_feats)):                
            # Define a Python callback function to update progress
            def update_progress():
                pbar.update(1)
            try:
                batch_results = parallel_topological_comp(
                    locs=batch_locs,
                    feats=batch_feats,
                    spatial_type=spatial_type,
                    fwhm=fwhm,
                    min_size=min_size,
                    thres_per=thres_per,
                    return_mode=return_mode,
                    num_workers=num_workers,  # 0 to auto-detect
                    progress_callback=update_progress,
                    log_callback=log_callback_func,
                )

                # Store batch results in the main results list
                start_idx = batch_idx * batch_size
                for i, res in enumerate(batch_results):
                    results[start_idx + i] = res

            except Exception as e:
                log_callback_func(f"\nException during batch {batch_idx}: {e}\n")
                raise

            # Clear memory after each batch
            gc.collect()

    return results


def parallel_with_progress_jaccard_composite(
    CCx_loc_sums, CCy_loc_sums, feat_xs=None, feat_ys=None,
    jaccard_type="default", num_workers=0,
    log_callback_func=None, batch_size=500
):
    """
    Batched parallel computation for Jaccard composite index.
    Processes each batch sequentially, waiting for one batch to complete before moving to the next.

    Args:
        CCx_loc_sums (list or np.ndarray): List of NumPy arrays for connected component location sums for feature x.
        CCy_loc_sums (list or np.ndarray): List of NumPy arrays for connected component location sums for feature y.
        feat_xs (list or np.ndarray, optional): List of NumPy arrays for feature x values.
        feat_ys (list or np.ndarray, optional): List of NumPy arrays for feature y values.
        jaccard_type (str, optional): Type of Jaccard index to calculate. Either "default" or "weighted".
        num_workers (int): Number of parallel workers (0 to auto-detect).
        log_callback_func (callable, optional): Function to handle log messages from C++.
        batch_size (int, optional): Maximum number of tasks per batch.

    Returns:
        list: A list of Jaccard composite indices.
    """
    # Assign log_callback_func early to ensure it's not None during exception handling
    if log_callback_func is None:
        log_callback_func = default_log_callback

    # Initialize feat_xs and feat_ys if not provided
    if feat_xs is None:
        feat_xs = [np.array([0.0]) for _ in range(len(CCx_loc_sums))]
    if feat_ys is None:
        feat_ys = [np.array([0.0]) for _ in range(len(CCy_loc_sums))]

    # Validate input lengths
    if not (len(CCx_loc_sums) == len(CCy_loc_sums) == len(feat_xs) == len(feat_ys)):
        log_callback_func("Error: All input lists must have the same length.\n")
        raise ValueError("All input lists must have the same length.")

    try:
        # Convert all NumPy arrays to Python lists
        CCx_loc_sums = [arr.tolist() for arr in CCx_loc_sums]
        CCy_loc_sums = [arr.tolist() for arr in CCy_loc_sums]
        feat_xs = [arr.tolist() for arr in feat_xs]
        feat_ys = [arr.tolist() for arr in feat_ys]
    except Exception as e:
        log_callback_func(f"Error during conversion of NumPy arrays to lists: {e}\n")
        raise

    # Create batches
    batches_CCx = list(create_batches(CCx_loc_sums, batch_size))
    batches_CCy = list(create_batches(CCy_loc_sums, batch_size))
    batches_feat_x = list(create_batches(feat_xs, batch_size))
    batches_feat_y = list(create_batches(feat_ys, batch_size))

    # Initialize results list
    total_tasks = len(CCx_loc_sums)
    results = [None] * total_tasks

    # Initialize progress bar
    with tqdm.tqdm(total=total_tasks, desc="Processing Jaccard Composite") as pbar:
        for batch_idx, (batch_CCx, batch_CCy, batch_feat_x, batch_feat_y) in enumerate(
            zip(batches_CCx, batches_CCy, batches_feat_x, batches_feat_y)
        ):
            # Define a Python callback function to update progress
            def update_progress():
                pbar.update(len(batch_CCx))

            try:
                batch_results = parallel_jaccard_composite(
                    CCx_loc_sums=batch_CCx,
                    CCy_loc_sums=batch_CCy,
                    feat_xs=batch_feat_x,
                    feat_ys=batch_feat_y,
                    jaccard_type=jaccard_type,
                    num_workers=num_workers,  # 0 to auto-detect
                    progress_callback=update_progress,
                    log_callback=log_callback_func,
                )

                # Store batch results in the main results list
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(batch_results)
                results[start_idx:end_idx] = batch_results

            except Exception as e:
                # Ensure that log_callback_func is callable before invoking
                if callable(log_callback_func):
                    log_callback_func(f"\nException during batch {batch_idx}: {e}\n")
                else:
                    print(f"\nException during batch {batch_idx}: {e}\n")
                raise
            # Clear memory after each batch
            gc.collect()

    return results