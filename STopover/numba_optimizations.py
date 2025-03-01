"""
Numba optimizations for STopover package
This module provides JIT-compiled implementations of computationally intensive functions
"""
import numpy as np
from scipy import sparse
import numba
from numba import jit, prange, int32, float64, boolean
from numba.typed import List
from concurrent.futures import ProcessPoolExecutor
import os

# Check if numba is available
try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    print("Numba not found. Using standard Python implementation.")

# Enable parallel processing in numba if available
if _HAS_NUMBA:
    numba.set_num_threads(min(os.cpu_count(), 8))  # Limit to avoid oversubscription

@jit(nopython=True, parallel=True, cache=True)
def extract_connected_components_numba(U, indices, indptr, data, threshold):
    """
    Numba-optimized implementation of connected component extraction
    
    Parameters:
    -----------
    U : numpy.ndarray
        Gene expression profiles (1D array)
    indices, indptr, data : numpy.ndarray
        CSR matrix components of adjacency matrix
    threshold : numpy.ndarray
        Threshold values for filtration
    
    Returns:
    --------
    CC_list : list of numpy.ndarray
        List of connected components
    E_data : numpy.ndarray
        Connectivity matrix data
    E_indices : numpy.ndarray
        Connectivity matrix indices
    E_indptr : numpy.ndarray
        Connectivity matrix indptr
    duration : numpy.ndarray
        Birth and death times of CCs
    history : list of numpy.ndarray
        History of CCs
    """
    n = len(U)
    num_thresholds = len(threshold)
    
    # Initialize outputs
    CC_list = List()
    duration = np.zeros((n, 2), dtype=np.float64)
    history = List()
    
    # Initialize connectivity matrix in CSR format
    E_data = np.zeros(n*n, dtype=np.float64)
    E_indices = np.zeros(n*n, dtype=np.int32)
    E_indptr = np.zeros(n+1, dtype=np.int32)
    
    # For each threshold
    cc_idx = 0
    for t_idx in range(num_thresholds):
        t = threshold[t_idx]
        
        # Find nodes that satisfy threshold
        nodes = np.zeros(n, dtype=np.int32)
        num_nodes = 0
        for i in range(n):
            if U[i] >= t:
                nodes[num_nodes] = i
                num_nodes += 1
        
        if num_nodes == 0:
            continue
        
        # Create visited array
        visited = np.zeros(n, dtype=np.int32)
        
        # Find connected components
        for node_idx in range(num_nodes):
            node = nodes[node_idx]
            if visited[node] == 0:
                # BFS to find connected component
                cc = np.zeros(n, dtype=np.int32)
                cc_size = 0
                
                queue = np.zeros(n, dtype=np.int32)
                queue_start = 0
                queue_end = 1
                queue[0] = node
                visited[node] = 1
                
                while queue_start < queue_end:
                    current = queue[queue_start]
                    queue_start += 1
                    
                    cc[cc_size] = current
                    cc_size += 1
                    
                    # Add neighbors
                    for j in range(indptr[current], indptr[current+1]):
                        neighbor = indices[j]
                        if visited[neighbor] == 0 and U[neighbor] >= t:
                            queue[queue_end] = neighbor
                            queue_end += 1
                            visited[neighbor] = 1
                
                # Add connected component to list
                CC_list.append(cc[:cc_size])
                
                # Update duration
                duration[cc_idx, 0] = t
                
                # Update history
                history.append(np.array([], dtype=np.int32))
                
                # Update connectivity matrix
                E_indptr[cc_idx+1] = E_indptr[cc_idx] + 1
                E_indices[E_indptr[cc_idx]] = cc_idx
                E_data[E_indptr[cc_idx]] = t
                
                cc_idx += 1
                
                # Check if we've reached the maximum number of CCs
                if cc_idx >= n:
                    break
        
        # Check if we've reached the maximum number of CCs
        if cc_idx >= n:
            break
    
    # Trim outputs to actual size
    return CC_list, E_data[:E_indptr[cc_idx]], E_indices[:E_indptr[cc_idx]], E_indptr[:cc_idx+1], duration[:cc_idx], history[:cc_idx]

@jit(nopython=True, parallel=True, cache=True)
def compute_jaccard_similarity_numba(CCx, CCy):
    """
    Numba-optimized implementation of Jaccard similarity calculation
    
    Parameters:
    -----------
    CCx, CCy : numpy.ndarray
        Binary matrices where each row represents a connected component
    
    Returns:
    --------
    J : numpy.ndarray
        Jaccard similarity matrix
    """
    n_x = CCx.shape[1]
    n_y = CCy.shape[1]
    
    # Initialize output
    J = np.zeros((n_x, n_y), dtype=np.float64)
    
    # Calculate Jaccard similarity for each pair of CCs
    for i in prange(n_x):
        for j in range(n_y):
            # Get binary vectors
            x = CCx[:, i]
            y = CCy[:, j]
            
            # Calculate intersection and union
            intersection = np.sum((x != 0) & (y != 0))
            union = np.sum((x != 0) | (y != 0))
            
            # Calculate Jaccard similarity
            if union > 0:
                J[i, j] = intersection / union
    
    return J

@jit(nopython=True, parallel=True, cache=True)
def compute_weighted_jaccard_similarity_numba(CCx, CCy, feat_x, feat_y):
    """
    Numba-optimized implementation of weighted Jaccard similarity calculation
    
    Parameters:
    -----------
    CCx, CCy : numpy.ndarray
        Binary matrices where each row represents a connected component
    feat_x, feat_y : numpy.ndarray
        Feature values for each connected component
    
    Returns:
    --------
    J : numpy.ndarray
        Weighted Jaccard similarity matrix
    """
    n_x = CCx.shape[1]
    n_y = CCy.shape[1]
    n_spots = CCx.shape[0]
    
    # Initialize output
    J = np.zeros((n_x, n_y), dtype=np.float64)
    
    # Normalize feature values
    feat_x_min = np.min(feat_x)
    feat_x_max = np.max(feat_x)
    feat_y_min = np.min(feat_y)
    feat_y_max = np.max(feat_y)
    
    feat_x_norm = np.zeros_like(feat_x)
    feat_y_norm = np.zeros_like(feat_y)
    
    if feat_x_max > feat_x_min:
        feat_x_norm = (feat_x - feat_x_min) / (feat_x_max - feat_x_min)
    
    if feat_y_max > feat_y_min:
        feat_y_norm = (feat_y - feat_y_min) / (feat_y_max - feat_y_min)
    
    # Calculate weighted Jaccard similarity for each pair of CCs
    for i in prange(n_x):
        for j in range(n_y):
            sum_min = 0.0
            sum_max = 0.0
            
            for k in range(n_spots):
                # Check if spot is in either CC
                in_x = CCx[k, i] != 0
                in_y = CCy[k, j] != 0
                
                if in_x or in_y:
                    # Get feature values
                    val_x = feat_x_norm[k] if in_x else 0.0
                    val_y = feat_y_norm[k] if in_y else 0.0
                    
                    sum_min += min(val_x, val_y)
                    sum_max += max(val_x, val_y)
            
            if sum_max > 0:
                J[i, j] = sum_min / sum_max
    
    return J

@jit(nopython=True, cache=True)
def gaussian_filter_numba(A, sigma):
    """
    Numba-optimized implementation of Gaussian filtering
    
    Parameters:
    -----------
    A : numpy.ndarray
        Distance matrix
    sigma : float
        Sigma parameter for Gaussian kernel
    
    Returns:
    --------
    mask : numpy.ndarray
        Gaussian mask
    """
    mask = np.exp(-(A**2) / (2 * sigma**2))
    return mask

@jit(nopython=True)
def extract_adjacency_spatial_numba(loc, spatial_type='visium', fwhm=2.5):
    """
    Numba-optimized version of extract_adjacency_spatial for Visium data
    """
    sigma = fwhm / 2.355
    
    if spatial_type == 'visium':
        # Calculate pairwise distances
        n = loc.shape[0]
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((loc[i] - loc[j])**2))
                A[i, j] = round(dist, 4)
                A[j, i] = A[i, j]
        
        # Replace distances > fwhm with infinity
        for i in range(n):
            for j in range(n):
                if A[i, j] > fwhm:
                    A[i, j] = np.inf
        
        # Gaussian smoothing
        arr_mod = np.zeros_like(A)
        for i in range(n):
            for j in range(n):
                if A[i, j] != np.inf:
                    arr_mod[i, j] = 1/(2*np.pi*sigma**2)*np.exp(-(A[i, j]**2)/(2*sigma**2))
        
        # Find minimum non-zero distance
        min_distance = np.inf
        for i in range(n):
            for j in range(n):
                if 0 < A[i, j] < min_distance:
                    min_distance = A[i, j]
        
        # Convert to adjacency matrix (1 if adjacent, 0 otherwise)
        for i in range(n):
            for j in range(n):
                if 0 < A[i, j] <= min_distance:
                    A[i, j] = 1
                else:
                    A[i, j] = 0
        
        return A, arr_mod
    else:
        # For other spatial types, we'll use the original function
        return None, None

@jit(nopython=True)
def compute_jaccard_similarity_numba(CCx_loc_mat, CCy_loc_mat):
    """
    Numba-optimized version of jaccard_composite for standard jaccard calculation
    """
    # Calculate Jaccard similarity
    n = len(CCx_loc_mat)
    intersection = 0
    union = 0
    
    for i in range(n):
        if CCx_loc_mat[i] > 0 and CCy_loc_mat[i] > 0:
            intersection += 1
        if CCx_loc_mat[i] > 0 or CCy_loc_mat[i] > 0:
            union += 1
    
    if union == 0:
        return 0.0
    
    return intersection / union

@jit(nopython=True)
def compute_weighted_jaccard_similarity_numba(CCx_loc_mat, CCy_loc_mat, feat_x_val, feat_y_val):
    """
    Numba-optimized version of jaccard_composite for weighted jaccard calculation
    """
    # Calculate weighted Jaccard similarity
    n = len(CCx_loc_mat)
    intersection_sum = 0.0
    union_sum = 0.0
    
    for i in range(n):
        x_val = feat_x_val[i, 0] if CCx_loc_mat[i] > 0 else 0
        y_val = feat_y_val[i, 0] if CCy_loc_mat[i] > 0 else 0
        
        intersection_sum += min(x_val, y_val)
        union_sum += max(x_val, y_val)
    
    if union_sum == 0:
        return 0.0
    
    return intersection_sum / union_sum

def topological_comp_res_numba(feat=None, A=None, mask=None,
                              spatial_type='visium', min_size=5, thres_per=30, return_mode='all'):
    """
    Numba-optimized version of topological_comp_res
    
    This function has the same signature and behavior as the original topological_comp_res
    but uses Numba-optimized functions where possible.
    """
    # This is a wrapper function that will call the original function
    # but use Numba-optimized functions for the computationally intensive parts
    
    # For now, we'll just call the original function
    # In the future, we can optimize specific parts of this function
    from .topological_comp import (
        extract_connected_comp,
        extract_connected_loc_mat,
        filter_connected_loc_exp,
        topological_comp_res
    )
    
    return topological_comp_res(
        feat=feat, A=A, mask=mask,
        spatial_type=spatial_type, min_size=min_size, 
        thres_per=thres_per, return_mode=return_mode
    )

def optimized_parallel_processing(func, items, **kwargs):
    """
    Parallel processing function that works with both numba and non-numba functions
    """
    num_workers = kwargs.get('num_workers', os.cpu_count())
    progress_bar = kwargs.get('progress_bar', True)
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        
        # Show progress bar if requested
        results = []
        try:
            from tqdm import tqdm
            for future in tqdm(futures, total=len(futures)):
                results.append(future.result())
        except ImportError:
            for future in futures:
                results.append(future.result())
    
    return results 