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

def extract_adjacency_spatial_numba(loc, spatial_type='visium', fwhm=2.5):
    """
    Numba-optimized implementation of adjacency matrix extraction
    
    Parameters:
    -----------
    loc : numpy.ndarray
        Spatial locations (2D array)
    spatial_type : str
        Type of spatial data
    fwhm : float
        Full width at half maximum
    
    Returns:
    --------
    A : scipy.sparse.csr_matrix
        Adjacency matrix
    arr_mod : numpy.ndarray
        Gaussian mask
    """
    sigma = fwhm / 2.355
    
    if spatial_type == 'visium':
        # Calculate pairwise distances
        n = loc.shape[0]
        A = np.zeros((n, n), dtype=np.float64)
        
        # Compute pairwise Euclidean distances using Numba
        @jit(nopython=True, parallel=True, cache=True)
        def compute_distances(loc, A):
            n = loc.shape[0]
            for i in prange(n):
                for j in range(i+1, n):
                    dist = np.sqrt((loc[i, 0] - loc[j, 0])**2 + (loc[i, 1] - loc[j, 1])**2)
                    A[i, j] = dist
                    A[j, i] = dist
            return A
        
        A = compute_distances(loc, A)
        
        # Replace distances > fwhm with infinity
        A_inf = A.copy()
        A_inf[A_inf > fwhm] = np.inf
        
        # Compute Gaussian mask
        arr_mod = gaussian_filter_numba(A_inf, sigma)
        
        # Convert to adjacency matrix (0/1)
        min_distance = np.min(A[np.nonzero(A)])
        A_adj = ((A > 0) & (A <= min_distance)).astype(np.int32)
        
        return sparse.csr_matrix(A_adj), arr_mod
    
    elif spatial_type in ['imageST', 'visiumHD']:
        # Implementation for other spatial types
        # For imageST and visiumHD, we use a different approach
        n = loc.shape[0]
        A = np.zeros((n, n), dtype=np.int32)
        
        # For these types, we connect adjacent pixels/spots
        @jit(nopython=True, parallel=True, cache=True)
        def compute_adjacency(loc, A):
            n = loc.shape[0]
            for i in prange(n):
                for j in range(i+1, n):
                    # Check if spots are adjacent (Manhattan distance = 1)
                    if abs(loc[i, 0] - loc[j, 0]) + abs(loc[i, 1] - loc[j, 1]) == 1:
                        A[i, j] = 1
                        A[j, i] = 1
            return A
        
        A = compute_adjacency(loc, A)
        
        # No Gaussian mask for these types
        arr_mod = None
        
        return sparse.csr_matrix(A), arr_mod
    
    return None, None

def optimized_parallel_processing(func, items, **kwargs):
    """
    Optimized parallel processing function that uses ProcessPoolExecutor
    
    Parameters:
    -----------
    func : callable
        Function to execute in parallel
    items : list
        Items to process
    **kwargs : dict
        Additional arguments to pass to the function
        
    Returns:
    --------
    results : list
        Results from parallel execution
    """
    num_workers = kwargs.pop('num_workers', os.cpu_count())
    progress_bar = kwargs.pop('progress_bar', True)
    
    # Use ProcessPoolExecutor for better performance
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(func, item, **kwargs) for item in items]
        
        # Collect results with optional progress bar
        results = []
        if progress_bar:
            try:
                from tqdm import tqdm
                for future in tqdm(futures, total=len(futures)):
                    results.append(future.result())
            except ImportError:
                for future in futures:
                    results.append(future.result())
        else:
            for future in futures:
                results.append(future.result())
    
    return results 