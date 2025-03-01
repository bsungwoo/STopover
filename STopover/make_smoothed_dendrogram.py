"""
Created on Sat Apr 3 18:39:34 2021
Last modified on Thurs March 31 09:16:00 2022
Numba optimization added for performance improvement

@author: 
Original matlab code for graph filtration: Hyekyoung Lee
Translation to python and addition of visualization code: Sungwoo Bae with the help of matlab2python

"""
import numpy as np
from scipy import sparse
import copy

# Check if numba is available
try:
    import numba
    from numba import jit, prange, int32, float64, boolean
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    print("Numba not found. Using standard Python implementation.")

@jit(nopython=True, cache=True)
def _filter_components_numba(cc_sizes, min_size):
    """
    Numba-optimized function to filter components by size
    
    Parameters:
    -----------
    cc_sizes : numpy.ndarray
        Sizes of connected components
    min_size : int
        Minimum size threshold
        
    Returns:
    --------
    keep_indices : numpy.ndarray
        Indices of components to keep
    new_indices_map : numpy.ndarray
        Mapping from old indices to new indices (-1 for filtered out components)
    """
    n = len(cc_sizes)
    keep_mask = np.zeros(n, dtype=np.bool_)
    
    # Identify components to keep
    for i in range(n):
        if cc_sizes[i] >= min_size:
            keep_mask[i] = True
    
    # Count kept components
    keep_count = np.sum(keep_mask)
    
    # Create arrays for indices
    keep_indices = np.zeros(keep_count, dtype=np.int32)
    new_indices_map = np.full(n, -1, dtype=np.int32)
    
    # Fill arrays
    idx = 0
    for i in range(n):
        if keep_mask[i]:
            keep_indices[idx] = i
            new_indices_map[i] = idx
            idx += 1
    
    return keep_indices, new_indices_map

@jit(nopython=True, cache=True)
def _update_connectivity_numba(E_dense, keep_indices, new_indices_map):
    """
    Numba-optimized function to update connectivity matrix
    
    Parameters:
    -----------
    E_dense : numpy.ndarray
        Original connectivity matrix
    keep_indices : numpy.ndarray
        Indices of components to keep
    new_indices_map : numpy.ndarray
        Mapping from old indices to new indices
        
    Returns:
    --------
    nE_dense : numpy.ndarray
        Updated connectivity matrix
    """
    n_new = len(keep_indices)
    nE_dense = np.zeros((n_new, n_new), dtype=E_dense.dtype)
    
    # Update connectivity matrix
    for i in range(n_new):
        old_i = keep_indices[i]
        for j in range(n_new):
            old_j = keep_indices[j]
            nE_dense[i, j] = E_dense[old_i, old_j]
    
    return nE_dense

def make_smoothed_dendrogram(CC, E, duration, history, min_size=5, use_numba=True):
    """
    Generate smoothed dendrogram by removing small connected components
    
    Parameters:
    -----------
    CC : list of lists
        Connected components
    E : numpy.ndarray or scipy.sparse.spmatrix
        Connectivity matrix
    duration : numpy.ndarray
        Birth and death times of CCs
    history : list of lists
        History of CCs
    min_size : int
        Minimum size of CC to consider
    use_numba : bool
        Whether to use Numba-optimized implementation
    
    Returns:
    --------
    nCC : list of lists
        Smoothed connected components
    nE : numpy.ndarray or scipy.sparse.spmatrix
        Smoothed connectivity matrix
    nduration : numpy.ndarray
        Smoothed birth and death times
    nhistory : list of lists
        Smoothed history of CCs
    """
    # Handle empty input case
    if len(CC) == 0:
        if is_sparse:
            return [], sparse.csr_matrix((0, 0)), np.zeros((0, 2)), []
        else:
            return [], np.zeros((0, 0)), np.zeros((0, 2)), []
    
    # Convert sparse matrix to dense if necessary
    is_sparse = isinstance(E, sparse.spmatrix)
    if is_sparse:
        E_dense = E.toarray()
    else:
        E_dense = E.copy()
    
    # Get component sizes
    cc_sizes = np.array([len(cc) for cc in CC], dtype=np.int32)
    
    # Use Numba optimization if available
    if use_numba and _HAS_NUMBA:
        # Filter components using Numba
        keep_indices, new_indices_map = _filter_components_numba(cc_sizes, min_size)
        
        # Handle case where no components meet the size threshold
        if len(keep_indices) == 0:
            if is_sparse:
                return [], sparse.csr_matrix((0, 0)), np.zeros((0, 2)), []
            else:
                return [], np.zeros((0, 0)), np.zeros((0, 2)), []
        
        # Update connectivity matrix using Numba
        nE_dense = _update_connectivity_numba(E_dense, keep_indices, new_indices_map)
        
        # Create new CC list (can't be done in Numba due to list of lists)
        nCC = [copy.deepcopy(CC[i]) for i in keep_indices]
        
        # Update duration array
        nduration = duration[keep_indices].copy()
        
        # Update history (can't be done in Numba due to list of lists)
        nhistory = []
        for i in keep_indices:
            new_hist = []
            for idx in history[i]:
                new_idx = new_indices_map[idx]
                if new_idx >= 0:  # Only include indices that weren't filtered out
                    new_hist.append(int(new_idx))  # Convert to int to ensure compatibility
            nhistory.append(new_hist)
    else:
        # Standard Python implementation
        nCC = []
        nduration = []
        nhistory = []
        
        # Track indices mapping
        idx_map = {}
        
        # First pass: identify CCs to keep
        for i, cc in enumerate(CC):
            if len(cc) >= min_size:
                idx_map[i] = len(nCC)
                nCC.append(copy.deepcopy(cc))
                nduration.append(duration[i].copy())
                nhistory.append([])  # Placeholder, will update in second pass
        
        # Handle case where no components meet the size threshold
        if len(nCC) == 0:
            if is_sparse:
                return [], sparse.csr_matrix((0, 0)), np.zeros((0, 2)), []
            else:
                return [], np.zeros((0, 0)), np.zeros((0, 2)), []
        
        # Convert nduration to numpy array
        nduration = np.array(nduration)
        
        # Create new connectivity matrix
        nE_dense = np.zeros((len(nCC), len(nCC)), dtype=E_dense.dtype)
        
        # Second pass: update connectivity matrix and history
        for old_i, new_i in idx_map.items():
            for old_j, new_j in idx_map.items():
                nE_dense[new_i, new_j] = E_dense[old_i, old_j]
            
            # Update history
            new_hist = []
            for idx in history[old_i]:
                if idx in idx_map:
                    new_hist.append(idx_map[idx])
            nhistory[new_i] = new_hist
    
    # Convert back to sparse matrix if necessary
    if is_sparse:
        nE = sparse.csr_matrix(nE_dense)
    else:
        nE = nE_dense
    
    return nCC, nE, nduration, nhistory