"""
Created on Sat Apr 3 18:39:34 2021
Last modified on Thurs March 31 09:16:00 2022

@author: 
Original matlab code for graph filtration: Hyekyoung Lee
Translation to python and addition of visualization code: Sungwoo Bae with the help of matlab2python
Numba optimization: Added for performance improvement

"""
import numpy as np
from scipy import sparse
import numba
from numba import jit, prange, int32, float64, boolean

# Check if numba is available
try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    print("Numba not found. Using standard Python implementation.")

# The algorithm is identical to networkx python module
def extract_connected_nodes(edge_list, sel_node_idx):
    '''
    ## Extract node indices of a connected component which contains a selected node
    (Algorithm is identical to python networkx _plain_bfs function)
    ### Input
    edge_list: list containing array of all nodes connected with each node
    idx: index of the selected node (from 0 to node number)

    ### Output
    set of nodes constituting the connected components containing the selected node
    '''
    cc_set = set()
    next_neighbor = {sel_node_idx}
    while next_neighbor:
        curr_neighbor = next_neighbor
        next_neighbor = set()
        cc_set.update(curr_neighbor)
        for node in curr_neighbor:
            next_neighbor.update(n for n in edge_list[node] if n not in cc_set)
    return cc_set

@jit(nopython=True, cache=True)
def _extract_connected_nodes_numba(edge_indices, edge_indptr, sel_node_idx, n_nodes):
    """
    Numba-optimized version of extract_connected_nodes
    
    Parameters:
    -----------
    edge_indices : numpy.ndarray
        CSR format indices array for edge list
    edge_indptr : numpy.ndarray
        CSR format indptr array for edge list
    sel_node_idx : int
        Index of the selected node
    n_nodes : int
        Total number of nodes
        
    Returns:
    --------
    cc_array : numpy.ndarray
        Array of node indices in the connected component
    cc_size : int
        Size of the connected component
    """
    # Initialize visited array
    visited = np.zeros(n_nodes, dtype=np.bool_)
    
    # Initialize queue
    queue = np.zeros(n_nodes, dtype=np.int32)
    queue_start = 0
    queue_end = 1
    queue[0] = sel_node_idx
    visited[sel_node_idx] = True
    
    # BFS
    while queue_start < queue_end:
        node = queue[queue_start]
        queue_start += 1
        
        # Process neighbors
        for i in range(edge_indptr[node], edge_indptr[node + 1]):
            neighbor = edge_indices[i]
            if not visited[neighbor]:
                visited[neighbor] = True
                queue[queue_end] = neighbor
                queue_end += 1
    
    # Create result array
    cc_array = np.zeros(queue_end, dtype=np.int32)
    for i in range(queue_end):
        cc_array[i] = queue[i]
    
    return cc_array, queue_end

def make_original_dendrogram_cc(U, A, threshold, min_size, use_numba=False):
    """
    Make original dendrogram for connected components
    
    Parameters
    ----------
    U : array-like
        Feature values
    A : array-like
        Adjacency matrix
    threshold : array-like
        Threshold values
    min_size : int
        Minimum size of connected components
    use_numba : bool, optional
        Whether to use Numba optimization, by default False
        
    Returns
    -------
    CC : array-like
        Connected components
    E : array-like
        Edge matrix
    duration : array-like
        Duration of connected components
    history : array-like
        History of connected components
    """
    # Disable Numba for now due to compatibility issues
    use_numba = False
    
    # Convert inputs to numpy arrays if they aren't already
    U = np.asarray(U, dtype=float)
    if isinstance(A, sparse.spmatrix):
        A = A.toarray()
    A = np.asarray(A, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    
    # Initialize variables
    n = len(U)
    max_cc = min(10000, n)  # Maximum number of connected components
    
    # Initialize arrays
    CC = [[] for _ in range(max_cc)]  # List of lists to store connected components
    E = np.zeros((max_cc, max_cc), dtype=float)  # Edge matrix
    duration = np.zeros((max_cc, 2), dtype=float)  # Duration of connected components
    history = np.zeros((max_cc, 3), dtype=float)  # History of connected components
    
    # Dictionary to keep track of connected components
    ck_cc = {}
    ncc = 0  # Number of connected components
    
    # Process each threshold
    for t in threshold:
        # Create binary mask
        mask = U >= t
        
        # Find connected components
        if use_numba:
            # Use Numba-optimized function if available
            try:
                from .numba_optimizations import find_connected_components_numba
                components = find_connected_components_numba(mask, A)
            except:
                # Fall back to standard implementation
                components = []
                visited = np.zeros(n, dtype=bool)
                for i in range(n):
                    if mask[i] and not visited[i]:
                        component = []
                        queue = [i]
                        visited[i] = True
                        while queue:
                            node = queue.pop(0)
                            component.append(node)
                            for j in range(n):
                                if A[node, j] > 0 and mask[j] and not visited[j]:
                                    queue.append(j)
                                    visited[j] = True
                        if len(component) >= min_size:
                            components.append(component)
        else:
            # Standard implementation
            components = []
            visited = np.zeros(n, dtype=bool)
            for i in range(n):
                if mask[i] and not visited[i]:
                    component = []
                    queue = [i]
                    visited[i] = True
                    while queue:
                        node = queue.pop(0)
                        component.append(node)
                        for j in range(n):
                            if A[node, j] > 0 and mask[j] and not visited[j]:
                                queue.append(j)
                                visited[j] = True
                    if len(component) >= min_size:
                        components.append(component)
        
        # Process each connected component
        for component in components:
            # Sort component
            component.sort()
            
            # Create key for component
            cc_key = tuple(component)
            
            # Check if component already exists
            if cc_key in ck_cc:
                continue
            
            # Add component to CC
            CC[ncc] = component
            
            # Add to dictionary
            ck_cc[cc_key] = ncc
            
            # Set birth time - this is where the error was occurring
            # Make sure t is a scalar, not a sequence
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
