import numpy as np
import tqdm
from .parallelize import parallel_topological_comp, parallel_jaccard_composite

def parallel_with_progress_topological_comp(locs, feats, spatial_type="visium", fwhm=2.5,
                                            min_size=5, thres_per=30, return_mode="all", num_workers=4):
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
        
    Returns:
        list: A list of topological components for each feature.
    """
    # Create a progress bar
    with tqdm.tqdm(total=len(feats)) as pbar:
        # Define a Python callback function to update progress
        def update_progress():
            pbar.update(1)
        
        # Call the C++ function in parallel, passing the progress callback
        result = parallel_topological_comp(locs, spatial_type, fwhm, feats, min_size, thres_per, return_mode, num_workers, update_progress)

    return result


def parallel_with_progress_jaccard_composite(CCx_loc_sums, CCy_loc_sums, feat_xs=None, feat_ys=None, num_workers=4):
    """
    Parallel computation for Jaccard composite index.
    Progress is shown using a tqdm progress bar.
    
    Args:
        CCx_loc_sums (list or np.ndarray): List of NumPy arrays for connected component location sums for feature x.
        CCy_loc_sums (list or np.ndarray): List of NumPy arrays for connected component location sums for feature y.
        feat_xs (list or np.ndarray, optional): List of NumPy arrays for feature x values.
        feat_ys (list or np.ndarray, optional): List of NumPy arrays for feature y values.
        num_workers (int, optional): Number of parallel workers.
        
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
    
    # Create a progress bar
    with tqdm.tqdm(total=len(CCx_loc_sums)) as pbar:
        # Define a Python callback function to update progress
        def update_progress():
            pbar.update(1)
        
        # Call the C++ function in parallel, passing the progress callback
        result = parallel_jaccard_composite(
            CCx_loc_sums=CCx_loc_sums, 
            CCy_loc_sums=CCy_loc_sums,
            feat_xs=feat_xs,
            feat_ys=feat_ys,
            num_workers=num_workers,
            progress_callback=update_progress
        )

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
# result_topological_comp = parallel_with_progress_topological_comp(feats, A_matrices, masks)
# result_jaccard_composite = parallel_with_progress_jaccard_composite(CCx_loc_sums, CCy_loc_sums)