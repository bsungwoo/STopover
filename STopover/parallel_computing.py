import numpy as np
import tqdm
from .parallelize import parallel_extract_adjacency, parallel_topological_comp, parallel_jaccard_composite
import scipy.sparse as sparse

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
    try:
        print(f"Starting parallel_extract_adjacency with {len(locs)} locations")
        
        # Ensure locations are properly formatted
        for i, loc in enumerate(locs):
            if not isinstance(loc, np.ndarray):
                locs[i] = np.array(loc, dtype=np.float64)
            
            # Ensure the array is contiguous in memory
            if not locs[i].flags.c_contiguous:
                locs[i] = np.ascontiguousarray(locs[i], dtype=np.float64)
        
        with tqdm.tqdm(total=len(locs)) as pbar:
            # Using a lambda for the callback so that each task updates the progress bar.
            progress_callback = lambda: pbar.update(1)
            result = parallel_extract_adjacency(locs, spatial_type, fwhm, num_workers, progress_callback)
        return result
    except Exception as e:
        print(f"ERROR in parallel_with_progress_extract_adjacency: {str(e)}")
        # Return empty results instead of crashing
        return [(sparse.csr_matrix((0, 0)), np.empty((0, 0))) for _ in range(len(locs))]


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


def parallel_with_progress_jaccard_composite(cc_1_list, cc_2_list, jaccard_type="default", num_workers=4):
    """
    Parallel computation for calculating Jaccard composite indices.
    Progress is shown using a tqdm progress bar.
    
    Args:
        cc_1_list (list): List of connected components for the first feature.
        cc_2_list (list): List of connected components for the second feature.
        jaccard_type (str): Type of Jaccard index to compute.
        num_workers (int): Number of parallel workers.
        
    Returns:
        list: A list of Jaccard indices.
    """
    try:
        # Validate inputs
        if len(cc_1_list) != len(cc_2_list):
            print(f"Error: cc_1_list length ({len(cc_1_list)}) doesn't match cc_2_list length ({len(cc_2_list)})")
            return [0.0] * len(cc_1_list)
            
        # Check for empty arrays
        for i, (cc1, cc2) in enumerate(zip(cc_1_list, cc_2_list)):
            if cc1.size == 0 or cc2.size == 0:
                print(f"Warning: Empty array at index {i}")
                
        # Convert arrays to the right type
        cc_1_list_int = []
        cc_2_list_int = []
        
        for i, (cc1, cc2) in enumerate(zip(cc_1_list, cc_2_list)):
            try:
                cc_1_list_int.append(np.asarray(cc1, dtype=np.int32))
                cc_2_list_int.append(np.asarray(cc2, dtype=np.int32))
            except Exception as e:
                print(f"Error converting arrays at index {i}: {str(e)}")
                cc_1_list_int.append(np.zeros((1, 1), dtype=np.int32))
                cc_2_list_int.append(np.zeros((1, 1), dtype=np.int32))
        
        with tqdm.tqdm(total=len(cc_1_list)) as pbar:
            # Using a lambda for the callback so that each task updates the progress bar.
            progress_callback = lambda: pbar.update(1)
            result = parallel_jaccard_composite(cc_1_list_int, cc_2_list_int, jaccard_type, num_workers, progress_callback)
        return result
    except Exception as e:
        print(f"ERROR in parallel_with_progress_jaccard_composite: {str(e)}")
        # Return zeros instead of crashing
        return [0.0] * len(cc_1_list)

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