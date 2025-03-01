"""
Memory optimizations for STopover package
This module provides memory-efficient implementations of data structures
"""
import numpy as np
from scipy import sparse
import numba
from numba import jit, prange, int8, int32, float64

# Check if numba is available
try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    print("Numba not found. Using standard Python implementation.")

def sparse_connected_components(CC, n_spots):
    """
    Convert connected components to sparse format for memory efficiency
    
    Parameters:
    -----------
    CC : list
        List of connected components
    n_spots : int
        Total number of spots
    
    Returns:
    --------
    CC_sparse : list of scipy.sparse.csr_matrix
        Sparse representation of connected components
    """
    CC_sparse = []
    
    for cc in CC:
        # Create sparse vector
        data = np.ones(len(cc), dtype=np.int8)
        indices = np.array(cc, dtype=np.int32)
        indptr = np.array([0, len(cc)], dtype=np.int32)
        
        # Create sparse matrix
        cc_sparse = sparse.csr_matrix((data, indices, indptr), shape=(1, n_spots))
        CC_sparse.append(cc_sparse)
    
    return CC_sparse

def merge_sparse_connected_components(CC_sparse):
    """
    Merge sparse connected components into a single sparse matrix
    
    Parameters:
    -----------
    CC_sparse : list of scipy.sparse.csr_matrix
        Sparse representation of connected components
    
    Returns:
    --------
    CC_merged : scipy.sparse.csr_matrix
        Merged sparse matrix of connected components
    """
    if not CC_sparse:
        return None
    
    # Vertically stack sparse matrices
    CC_merged = sparse.vstack(CC_sparse)
    
    return CC_merged

@jit(nopython=True, cache=True)
def create_binary_matrix_numba(CC, n_spots):
    """
    Numba-optimized function to create a binary matrix from connected components
    
    Parameters:
    -----------
    CC : list of arrays
        List of connected components
    n_spots : int
        Total number of spots
    
    Returns:
    --------
    matrix : numpy.ndarray
        Binary matrix where each row represents a connected component
    """
    n_cc = len(CC)
    matrix = np.zeros((n_cc, n_spots), dtype=np.int8)
    
    for i in range(n_cc):
        for j in range(len(CC[i])):
            matrix[i, CC[i][j]] = 1
    
    return matrix

@jit(nopython=True, parallel=True, cache=True)
def chunk_process_numba(data, func, chunk_size=1000):
    """
    Process data in chunks using Numba for better performance
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data to process
    func : function
        Function to apply to each chunk
    chunk_size : int
        Size of each chunk
    
    Returns:
    --------
    result : numpy.ndarray
        Processed data
    """
    n = len(data)
    result = np.zeros_like(data)
    
    for i in prange(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk = data[i:end]
        result[i:end] = func(chunk)
    
    return result

def compress_sparse_matrix(matrix):
    """
    Compress a sparse matrix to reduce memory usage
    
    Parameters:
    -----------
    matrix : scipy.sparse.spmatrix
        Sparse matrix to compress
    
    Returns:
    --------
    compressed : scipy.sparse.csr_matrix
        Compressed sparse matrix
    """
    # Convert to CSR format if not already
    if not isinstance(matrix, sparse.csr_matrix):
        matrix = matrix.tocsr()
    
    # Use smaller data types where possible
    matrix.data = matrix.data.astype(np.int8)
    
    # Remove explicit zeros
    matrix.eliminate_zeros()
    
    # Sort indices for better compression
    matrix.sort_indices()
    
    return matrix

@jit(nopython=True, cache=True)
def memory_efficient_jaccard_numba(x, y):
    """
    Memory-efficient implementation of Jaccard similarity using Numba
    
    Parameters:
    -----------
    x, y : numpy.ndarray
        Binary arrays
    
    Returns:
    --------
    j : float
        Jaccard similarity
    """
    # Process in chunks to reduce memory usage
    chunk_size = 1000
    n = len(x)
    
    intersection_count = 0
    union_count = 0
    
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        x_chunk = x[i:end]
        y_chunk = y[i:end]
        
        intersection_count += np.sum((x_chunk != 0) & (y_chunk != 0))
        union_count += np.sum((x_chunk != 0) | (y_chunk != 0))
    
    if union_count > 0:
        return intersection_count / union_count
    else:
        return 0.0

def chunk_processing(data, chunk_size=1000):
    """
    Process data in chunks to reduce memory usage
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data to process
    chunk_size : int
        Size of each chunk
    
    Returns:
    --------
    chunks : list of numpy.ndarray
        List of data chunks
    """
    n = len(data)
    chunks = []
    
    for i in range(0, n, chunk_size):
        chunk = data[i:min(i+chunk_size, n)]
        chunks.append(chunk)
    
    return chunks 