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
def _calculate_dendrogram_coordinates_numba(duration, children):
    """
    Numba-optimized implementation of dendrogram coordinate calculation
    
    Parameters:
    -----------
    duration : numpy.ndarray
        Birth and death times of CCs
    children : numpy.ndarray
        Children of each CC (2D array where each row contains indices of children)
    children_counts : numpy.ndarray
        Number of children for each CC
    
    Returns:
    --------
    vertical_x : numpy.ndarray
        X-coordinates of vertical lines
    vertical_y : numpy.ndarray
        Y-coordinates of vertical lines
    horizontal_x : numpy.ndarray
        X-coordinates of horizontal lines
    horizontal_y : numpy.ndarray
        Y-coordinates of horizontal lines
    dots : numpy.ndarray
        Coordinates of dots
    layer : numpy.ndarray
        Layer information for each CC
    """
    n = len(duration)
    
    # Initialize outputs
    vertical_x = np.zeros((n, 2), dtype=np.float64)
    vertical_y = np.zeros((n, 2), dtype=np.float64)
    horizontal_x = np.zeros((n, 2), dtype=np.float64)
    horizontal_y = np.zeros((n, 2), dtype=np.float64)
    dots = np.zeros((n, 2), dtype=np.float64)
    layer = np.zeros(n, dtype=np.int32)
    
    # Calculate coordinates
    for i in range(n):
        # Set birth and death times
        birth = duration[i, 0]
        death = duration[i, 1]
        
        # Set vertical line coordinates
        vertical_x[i, 0] = i
        vertical_x[i, 1] = i
        vertical_y[i, 0] = birth
        vertical_y[i, 1] = death
        
        # Set dot coordinates
        dots[i, 0] = i
        dots[i, 1] = birth
        
        # Set layer information
        if children[i, 0] == -1:  # No children
            layer[i] = 0
        else:
            max_child_layer = 0
            for j in range(n):
                if children[i, j] == -1:
                    break
                child_idx = children[i, j]
                if layer[child_idx] > max_child_layer:
                    max_child_layer = layer[child_idx]
            layer[i] = max_child_layer + 1
        
        # Set horizontal line coordinates
        if children[i, 0] != -1:  # Has children
            min_child_idx = n
            max_child_idx = -1
            
            for j in range(n):
                if children[i, j] == -1:
                    break
                child_idx = children[i, j]
                if child_idx < min_child_idx:
                    min_child_idx = child_idx
                if child_idx > max_child_idx:
                    max_child_idx = child_idx
            
            horizontal_x[i, 0] = min_child_idx
            horizontal_x[i, 1] = max_child_idx
            horizontal_y[i, 0] = death
            horizontal_y[i, 1] = death
    
    return vertical_x, vertical_y, horizontal_x, horizontal_y, dots, layer

def make_dendrogram_bar(CC, E, duration, history, use_numba=True):
    """
    Calculate coordinates for dendrogram visualization
    
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
    use_numba : bool
        Whether to use Numba-optimized implementation
    
    Returns:
    --------
    vertical_x : numpy.ndarray
        X-coordinates of vertical lines
    vertical_y : numpy.ndarray
        Y-coordinates of vertical lines
    horizontal_x : numpy.ndarray
        X-coordinates of horizontal lines
    horizontal_y : numpy.ndarray
        Y-coordinates of horizontal lines
    dots : numpy.ndarray
        Coordinates of dots
    layer : numpy.ndarray
        Layer information for each CC
    """
    # Handle empty input case
    if len(CC) == 0:
        empty_result = np.zeros((0, 2))
        return empty_result, empty_result, empty_result, empty_result, empty_result, np.zeros(0, dtype=int)
    
    # Use Numba optimization if available
    if use_numba and _HAS_NUMBA:
        # Convert history to format compatible with Numba
        n = len(CC)
        max_children = max([len(h) for h in history], default=0)
        
        # Create 2D array for children
        children_array = np.full((n, max(max_children, 1)), -1, dtype=np.int32)
        
        for i, h in enumerate(history):
            for j, child in enumerate(h):
                children_array[i, j] = child
        
        # Call Numba-optimized function
        vertical_x, vertical_y, horizontal_x, horizontal_y, dots, layer = _calculate_dendrogram_coordinates_numba(
            duration, children_array
        )
    else:
        # Standard Python implementation
        n = len(CC)
        
        # Initialize outputs
        vertical_x = np.zeros((n, 2))
        vertical_y = np.zeros((n, 2))
        horizontal_x = np.zeros((n, 2))
        horizontal_y = np.zeros((n, 2))
        dots = np.zeros((n, 2))
        layer = np.zeros(n, dtype=int)
        
        # Calculate coordinates
        for i in range(n):
            # Set birth and death times
            birth = duration[i, 0]
            death = duration[i, 1]
            
            # Set vertical line coordinates
            vertical_x[i, 0] = i
            vertical_x[i, 1] = i
            vertical_y[i, 0] = birth
            vertical_y[i, 1] = death
            
            # Set dot coordinates
            dots[i, 0] = i
            dots[i, 1] = birth
            
            # Set layer information
            if i >= len(history) or not history[i]:
                layer[i] = 0
            else:
                layer[i] = max([layer[child] for child in history[i]], default=0) + 1
            
            # Set horizontal line coordinates
            if i < len(history) and history[i]:
                min_child_idx = min(history[i])
                max_child_idx = max(history[i])
                
                horizontal_x[i, 0] = min_child_idx
                horizontal_x[i, 1] = max_child_idx
                horizontal_y[i, 0] = death
                horizontal_y[i, 1] = death
    
    return vertical_x, vertical_y, horizontal_x, horizontal_y, dots, layer
