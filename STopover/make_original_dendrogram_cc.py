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

def make_original_dendrogram_cc(U, A, threshold=None, min_size=5, use_numba=True):
    '''
    ## Make dendrogram of connected components from gene expression profile and adjacency matrix
    
    ### Input
    U: gene expression profile (1D array)
    A: adjacency matrix (sparse matrix)
    threshold: threshold values for filtration (default: percentile values from 0 to 100 with step 5)
    min_size: minimum size of connected components to be considered (default: 5)
    use_numba: whether to use numba optimization (default: True)
    
    ### Output
    CC: list of connected components
    E: connectivity matrix
    duration: birth and death times of CCs
    history: history of CCs
    '''
    # Set default threshold if not provided
    if threshold is None:
        threshold = np.percentile(U, np.arange(0, 105, 5))
    
    # Convert adjacency matrix to CSR format if not already
    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)
    
    # Extract edge list from adjacency matrix
    n_nodes = A.shape[0]
    edge_list = [[] for _ in range(n_nodes)]
    
    # Fill edge list
    rows, cols = A.nonzero()
    for i, j in zip(rows, cols):
        edge_list[i].append(j)
    
    # For Numba optimization, convert edge list to CSR format
    if use_numba and _HAS_NUMBA:
        edge_indices = []
        edge_indptr = [0]
        
        for neighbors in edge_list:
            edge_indices.extend(neighbors)
            edge_indptr.append(len(edge_indices))
        
        edge_indices = np.array(edge_indices, dtype=np.int32)
        edge_indptr = np.array(edge_indptr, dtype=np.int32)
    
    # Initialize outputs
    CC = []  # Connected components
    E = sparse.lil_matrix((n_nodes, n_nodes))  # Connectivity matrix
    duration = np.zeros((n_nodes, 2))  # Birth and death times
    history = []  # History of CCs
    
    # Dictionary to check if a CC already exists
    ck_cc = {}
    
    # Number of CCs
    ncc = 0
    
    # For each threshold
    for i, t in enumerate(threshold):
        # Find nodes that satisfy threshold
        nodes = np.where(U >= t)[0]
        
        if len(nodes) == 0:
            continue
        
        # Find connected components
        visited = np.zeros(n_nodes, dtype=bool)
        
        for node in nodes:
            if not visited[node]:
                # Extract connected component
                if use_numba and _HAS_NUMBA:
                    cc_array, cc_size = _extract_connected_nodes_numba(edge_indices, edge_indptr, node, n_nodes)
                    cc = cc_array[:cc_size].tolist()
                else:
                    cc = list(extract_connected_nodes(edge_list, node))
                
                # Mark nodes as visited
                for n in cc:
                    visited[n] = True
                
                # Check if CC already exists
                cc_key = str(sorted(cc))
                if cc_key in ck_cc:
                    continue
                
                # Add new CC
                CC.append(cc)
                ck_cc[cc_key] = ncc
                
                # Set birth time
                duration[ncc, 0] = t
                
                # Update connectivity matrix
                E[ncc, ncc] = t
                
                # Increment CC counter
                ncc += 1
        
        # Find CCs that die at this threshold
        if i < len(threshold) - 1:
            next_t = threshold[i + 1]
            next_nodes = np.where(U >= next_t)[0]
            
            if len(next_nodes) == 0:
                # All remaining CCs die at this threshold
                for j in range(ncc):
                    if duration[j, 1] == 0:
                        duration[j, 1] = t
                continue
            
            # Find CCs that will merge at next threshold
            next_visited = np.zeros(n_nodes, dtype=bool)
            next_ccs = []
            
            for node in next_nodes:
                if not next_visited[node]:
                    # Extract connected component
                    if use_numba and _HAS_NUMBA:
                        cc_array, cc_size = _extract_connected_nodes_numba(edge_indices, edge_indptr, node, n_nodes)
                        cc = cc_array[:cc_size].tolist()
                    else:
                        cc = list(extract_connected_nodes(edge_list, node))
                    
                    # Mark nodes as visited
                    for n in cc:
                        next_visited[n] = True
                    
                    next_ccs.append(cc)
            
            # Find CCs that will merge
            for next_cc in next_ccs:
                # Find CCs that contain nodes in next_cc
                tind1 = []
                
                for j in range(ncc):
                    if duration[j, 1] == 0:  # CC is still alive
                        # Check if CC shares nodes with next_cc
                        if any(node in next_cc for node in CC[j]):
                            tind1.append(j)
                
                if len(tind1) == 1:
                    # Set death time if not already set
                    if duration[tind1[0], 1] == 0:
                        duration[tind1[0], 1] = t
                elif len(tind1) > 1:
                    # Create new CC
                    ncc += 1
                    CC_tind1 = [node for cc in tind1 for node in CC[cc]]
                    CC.append(list(set(CC_tind1)))
                    ck_cc[str(sorted(CC[-1]))] = ncc - 1  # Use 0-based indexing
                    
                    # Set birth and death times
                    duration[ncc-1, 0] = t
                    for idx in tind1:
                        duration[idx, 1] = t
                    
                    # Set connectivity
                    E_mod = np.eye(len(tind1)) * E[tind1, :][:, tind1].toarray() + (1 - np.eye(len(tind1))) * t
                    for ind_e, e in enumerate(tind1):
                        E[tind1, e] = E_mod[:, ind_e]
                    
                    E[ncc-1, tind1] = t
                    E[tind1, ncc-1] = t
                    E[ncc-1, ncc-1] = t
                    
                    # Update history
                    history.append(tind1)
    
    # Remove empty CCs
    valid_indices = []
    for index, cc in enumerate(CC):
        if len(cc) > 0:
            valid_indices.append(index)
    
    CC = [CC[i] for i in valid_indices]
    history = [history[i] for i in valid_indices if i < len(history)]
    E = E[valid_indices, :][:, valid_indices].copy()
    duration = duration[valid_indices, :]
    
    return CC, E, duration, history
