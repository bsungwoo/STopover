import numpy as np
import tqdm
from .parallelize import parallel_extract_adjacency, parallel_topological_comp, parallel_jaccard_composite

def parallel_with_progress_extract_adjacency(locs, spatial_type="visium", fwhm=2.5, num_workers=4):
    """
    Parallel computation for extracting the adjacency matrix and Gaussian smoothing mask.
    Progress is shown using a tqdm progress bar.
    
    Args:
        locs (list): List of locations (NumPy arrays) to compute adjacency for.
        spatial_type (str): Type of spatial data.
        fwhm (float): Full-width half-maximum for Gaussian smoothing.
        num_workers (int): Number of parallel workers.
        
    Returns:
        list: A list of tuples (adjacency matrix, Gaussian mask) for each location.
    """
    with tqdm.tqdm(total=len(locs)) as pbar:
        # Using a lambda for the callback so that each task updates the progress bar.
        progress_callback = lambda: pbar.update(1)
        result = parallel_extract_adjacency(locs, spatial_type, fwhm, num_workers, progress_callback)
    return result


def parallel_with_progress_topological_comp(feats, A_matrices, masks, spatial_type="visium",
                                            min_size=5, thres_per=30, return_mode="all", num_workers=4):
    """
    Parallel computation for topological component extraction.
    Progress is shown using a tqdm progress bar.
    
    Args:
        feats (list): List of feature arrays (NumPy arrays).
        A_matrices (list): List of adjacency matrices (scipy sparse CSR matrices).
        masks (list): List of Gaussian smoothing masks (NumPy arrays).
        spatial_type (str): Type of spatial data.
        min_size (int): Minimum size of connected component.
        thres_per (int): Percentile threshold for filtering connected components.
        return_mode (str): Return mode.
        num_workers (int): Number of parallel workers.
        
    Returns:
        list: A list of topological component outputs for each feature.
    """
    with tqdm.tqdm(total=len(feats)) as pbar:
        progress_callback = lambda: pbar.update(1)
        result = parallel_topological_comp(
            feats, A_matrices, masks, spatial_type, min_size, thres_per, return_mode, num_workers, progress_callback
        )
    return result


def parallel_with_progress_jaccard_composite(CCx_loc_sums, CCy_loc_sums, feat_xs=None, feat_ys=None, num_workers=4):
    """
    Parallel computation for Jaccard composite index.
    Progress is shown using a tqdm progress bar.
    
    Args:
        CCx_loc_sums (list): List of connected component location arrays (NumPy arrays) for feature x.
        CCy_loc_sums (list): List of connected component location arrays (NumPy arrays) for feature y.
        feat_xs (list): List of feature value arrays (NumPy arrays) for feature x. If None, empty arrays are used.
        feat_ys (list): List of feature value arrays (NumPy arrays) for feature y. If None, empty arrays are used.
        num_workers (int): Number of parallel workers.
        
    Returns:
        list: A list of Jaccard composite indices.
    """
    # Use empty matrices if features are not provided.
    feat_xs = feat_xs if feat_xs is not None else [np.empty((0, 0)) for _ in CCx_loc_sums]
    feat_ys = feat_ys if feat_ys is not None else [np.empty((0, 0)) for _ in CCy_loc_sums]

    with tqdm.tqdm(total=len(CCx_loc_sums)) as pbar:
        progress_callback = lambda: pbar.update(1)
        result = parallel_jaccard_composite(CCx_loc_sums, CCy_loc_sums, feat_xs, feat_ys, num_workers, progress_callback)
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