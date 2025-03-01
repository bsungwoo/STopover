"""
Numba optimizations for STopover package
This module provides JIT-compiled implementations of computationally intensive functions
"""
import numpy as np
import pandas as pd
from scipy import sparse
import numba
from numba import jit, prange, int32, float64, boolean
from numba.typed import List
from concurrent.futures import ProcessPoolExecutor
import os
from numpy.matlib import repmat

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
    Numba-optimized version of extract_adjacency_spatial for different spatial data types
    """
    sigma = fwhm / 2.355
    n = loc.shape[0]
    
    if spatial_type == 'visium':
        # Calculate pairwise distances
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
    
    elif spatial_type == 'imageST':
        # For imageST, we use a different approach based on grid neighbors
        # Assuming loc contains grid coordinates
        A = np.zeros((n, n))
        
        # Create adjacency matrix based on 4-connectivity (von Neumann neighborhood)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if points are adjacent in the grid (Manhattan distance = 1)
                    if abs(loc[i, 0] - loc[j, 0]) + abs(loc[i, 1] - loc[j, 1]) == 1:
                        A[i, j] = 1
        
        # Create mask for Gaussian smoothing
        arr_mod = np.zeros_like(A)
        for i in range(n):
            for j in range(n):
                # Calculate Euclidean distance
                dist = np.sqrt(np.sum((loc[i] - loc[j])**2))
                if dist <= fwhm:
                    arr_mod[i, j] = 1/(2*np.pi*sigma**2)*np.exp(-(dist**2)/(2*sigma**2))
        
        return A, arr_mod
    
    elif spatial_type == 'visiumHD':
        # For visiumHD, similar to visium but with potentially higher density
        # Calculate pairwise distances
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
        # For visiumHD, we might want to use a slightly different threshold
        # to account for the higher density
        threshold = min_distance * 1.1  # Slightly more permissive
        for i in range(n):
            for j in range(n):
                if 0 < A[i, j] <= threshold:
                    A[i, j] = 1
                else:
                    A[i, j] = 0
        
        return A, arr_mod
    
    else:
        # For unknown spatial types, return empty matrices
        # This will cause the code to fall back to the non-Numba version
        return np.zeros((n, n)), np.zeros((n, n))

@jit(nopython=True)
def compute_jaccard_similarity_numba(CCx_loc_mat, CCy_loc_mat):
    """
    Numba-optimized version of jaccard_composite for default jaccard calculation
    """
    # Calculate Jaccard similarity
    intersection = 0
    union = 0
    
    for i in range(len(CCx_loc_mat)):
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

@jit(nopython=True)
def find_connected_components_numba(A, t, min_size):
    """
    Numba-optimized version of finding connected components
    """
    n = A.shape[0]
    visited = np.zeros(n, dtype=np.int32)
    components = []
    
    for i in range(n):
        if visited[i] == 0 and A[i, i] >= t:
            # Start a new component
            component = []
            stack = [i]
            visited[i] = 1
            
            while stack:
                node = stack.pop()
                component.append(node)
                
                # Find neighbors
                for j in range(n):
                    if A[node, j] >= t and visited[j] == 0:
                        stack.append(j)
                        visited[j] = 1
            
            if len(component) >= min_size:
                components.append(component)
    
    return components

def make_original_dendrogram_cc_numba(U, A, threshold, min_size):
    """
    Numba-optimized version of make_original_dendrogram_cc
    """
    # Convert inputs to numpy arrays if they're not already
    if isinstance(U, list):
        U = np.array(U)
    if isinstance(A, sparse.spmatrix):
        A = A.toarray()
    if isinstance(threshold, list):
        threshold = np.array(threshold)
    
    # Initialize variables
    n = len(U)
    max_cc = n * 2  # Maximum possible number of connected components
    
    # Initialize arrays with proper types
    CC = np.zeros((max_cc, n), dtype=np.float64)
    E = np.zeros((max_cc, max_cc), dtype=np.float64)
    duration = np.zeros((max_cc, 2), dtype=np.float64)
    history = np.zeros((max_cc, 3), dtype=np.float64)
    
    # Track connected components
    ncc = 0  # Number of connected components
    ck_cc = {}  # Dictionary to track component IDs
    
    # Process each threshold
    for t_idx, t in enumerate(threshold):
        # Find connected components at this threshold
        components = find_connected_components_numba(A, t, min_size)
        
        for component in components:
            # Create a key for this component
            cc_key = tuple(sorted(component))
            
            # Check if this component already exists
            if cc_key in ck_cc:
                continue
            
            # Add new component
            for idx in component:
                CC[ncc, idx] = U[idx]
            
            # Track this component
            ck_cc[cc_key] = ncc
            
            # Set birth time
            duration[ncc, 0] = float(t)
            
            # Update connectivity matrix
            E[ncc, ncc] = float(t)
            
            # Update history
            history[ncc, 0] = float(t)  # Birth time
            history[ncc, 1] = float(ncc)  # Component ID
            history[ncc, 2] = float(len(component))  # Component size
            
            ncc += 1
    
    # Trim arrays to actual size
    CC = CC[:ncc]
    E = E[:ncc, :ncc]
    duration = duration[:ncc]
    history = history[:ncc]
    
    return CC, E, duration, history

def topological_comp_res_numba(feat=None, A=None, mask=None,
                              spatial_type='visium', min_size=5, thres_per=30, return_mode='all'):
    """
    Numba-optimized version of topological_comp_res
    
    This function has the same signature and behavior as the original topological_comp_res
    but uses Numba-optimized functions where possible.
    """
    # Check feasibility of the dataset
    if not (return_mode in ['all','cc_loc','jaccard_cc_list']):
        raise ValueError("'return_mode' should be among 'all', 'cc_loc', or 'jaccard_cc_list'")

    # If no dataset is given, then feature x and feature y should be provided as numpy arrays
    if (spatial_type=='visium') and (A is None or mask is None):
        raise ValueError("'A' and 'mask' should be provided")
    if isinstance(feat, sparse.spmatrix): feat = feat.toarray()
    elif isinstance(feat, np.ndarray): pass
    else: raise ValueError("Values for 'feat' should be provided as numpy ndarray or scipy sparse matrix")

    # Calculate adjacency matrix and mask if data is provided
    # Gaussian smoothing with zero padding
    p = len(feat)
    
    # Check shapes and fix if necessary
    if spatial_type == 'visium':
        # Convert to dense arrays if needed
        if isinstance(A, sparse.spmatrix):
            A_dense = A.toarray()
        else:
            A_dense = A
            
        # Ensure mask has the right shape
        if mask.shape != (p, p):
            # Fall back to standard implementation
            from .topological_comp import topological_comp_res
            return topological_comp_res(
                feat=feat, A=A, mask=mask,
                spatial_type=spatial_type, min_size=min_size, 
                thres_per=thres_per, return_mode=return_mode
            )
            
        # Reshape feat for broadcasting
        feat_reshaped = feat.reshape(-1, 1)
        # Apply smoothing
        smooth = np.zeros(p)
        for i in range(p):
            for j in range(p):
                smooth[i] += mask[i, j] * feat_reshaped[j, 0]
                
        # Normalize
        if np.sum(smooth) > 0:
            smooth = smooth / np.sum(smooth) * np.sum(feat)
    else:
        # Already smoothed features as input
        smooth = feat.flatten()

    ## Estimate dendrogram for feat
    t = smooth * (smooth > 0)
    # Find nonzero unique value and sort in descending order
    threshold = np.flip(np.sort(np.setdiff1d(t, 0), axis=None))
    
    try:
        # Try using the Numba-optimized version
        CC, E, duration, history = make_original_dendrogram_cc_numba(
            smooth, A_dense if spatial_type == 'visium' else A, threshold, min_size
        )
        
        # Import necessary functions for the rest of the processing
        from .topological_comp import (
            extract_connected_loc_mat,
            filter_connected_loc_exp
        )
        
        # Extract location of connected components as arrays
        CC_loc_mat = extract_connected_loc_mat(CC, num_spots=len(feat))
        CC_loc_mat = filter_connected_loc_exp(CC_loc_mat, feat=feat, thres_per=thres_per)
        
        if return_mode=='all': return CC, CC_loc_mat
        elif return_mode=='cc_loc': return CC_loc_mat
        else: return CC
        
    except Exception as e:
        # If there's an error, fall back to the original implementation
        print(f"Numba optimization failed: {e}. Falling back to standard implementation.")
        from .topological_comp import topological_comp_res
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