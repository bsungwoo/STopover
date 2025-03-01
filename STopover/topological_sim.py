"""
Created on Sat Apr 3 18:39:34 2021
Last modified on Thurs March 31 09:16:00 2022
Numba optimization added for performance improvement

@author: 
Original matlab code for graph filtration: Hyekyoung Lee
Translation to python and addition of visualization code: Sungwoo Bae with the help of matlab2python
"""
import numpy as np
import pandas as pd

import os
import time
import parmap

from .topological_comp import extract_adjacency_spatial, topological_comp_res
from .topological_comp import extract_connected_loc_mat
from .jaccard import jaccard_composite

from .numba_optimizations import (
    extract_adjacency_spatial_numba, 
    compute_jaccard_similarity_numba, 
    compute_weighted_jaccard_similarity_numba,
    optimized_parallel_processing
)
from .memory_optimizations import sparse_connected_components, merge_sparse_connected_components, chunk_processing

# Check if numba is available
try:
    import numba
    from numba import jit, prange, int32, float64, boolean
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    print("Numba not found. Using standard Python implementation.")

def topological_sim(data, loc, spatial_type='visium', fwhm=2.5, threshold=None, min_size=5, 
                   num_workers=None, progress_bar=True, use_numba=True, memory_efficient=True):
    '''
    ## Compute topological similarity between samples
    
    ### Input
    data: gene expression data (n_samples x n_spots x n_genes)
    loc: spatial coordinates (n_spots x 2)
    spatial_type: type of spatial data ('visium', 'ST', 'imageST', 'visiumHD')
    fwhm: full width at half maximum for Gaussian kernel (default: 2.5)
    threshold: threshold values for filtration (default: percentile values from 0 to 100 with step 5)
    min_size: minimum size of connected components to be considered (default: 5)
    num_workers: number of workers for parallel processing (default: number of CPU cores)
    progress_bar: whether to show progress bar (default: True)
    use_numba: whether to use numba optimization (default: True)
    memory_efficient: whether to use memory-efficient implementation (default: True)
    
    ### Output
    df_top_total: dataframe with topological similarity measures
    data_mod: modified data (for visualization)
    '''
    # Set default number of workers
    if num_workers is None:
        num_workers = os.cpu_count()
    
    # Start timer
    start_time = time.time()
    
    # Convert to numpy array if not already
    if isinstance(data, pd.DataFrame):
        data = data.values
    if isinstance(loc, pd.DataFrame):
        loc = loc.values
    
    # Extract adjacency matrix
    if use_numba and _HAS_NUMBA:
        A, data_mod = extract_adjacency_spatial_numba(loc, spatial_type, fwhm)
    else:
        A, data_mod = extract_adjacency_spatial(loc, spatial_type, fwhm)
    
    print("Adjacency matrix extraction: %.2f seconds" % (time.time()-start_time))
    
    # Initialize dataframe for topological similarity
    n_samples = data.shape[0]
    df_top_total = pd.DataFrame(index=range(n_samples*(n_samples-1)//2), 
                               columns=['sample1', 'sample2', 'J_comp'])
    
    # Compute topological components for each sample
    CC_list = []
    E_list = []
    duration_list = []
    history_list = []
    
    for i in range(n_samples):
        # Extract feature values for current sample
        feat_val = data[i]
        
        # Compute topological components
        CC, E, duration, history = topological_comp_res(
            feat_val, A, threshold, min_size, use_numba=use_numba
        )
        
        # Store results
        CC_list.append(CC)
        E_list.append(E)
        duration_list.append(duration)
        history_list.append(history)
    
    print("Topological component computation: %.2f seconds" % (time.time()-start_time))
    
    # Extract connected location matrices
    CCx_loc_mat_list = []
    
    for i in range(n_samples):
        # Extract feature values for current sample
        feat_val = data[i]
        
        # Extract connected location matrix
        CCx_loc_mat = extract_connected_loc_mat(
            CC_list[i], loc, feat_val, use_numba=use_numba
        )
        
        # Reshape to 2D if necessary
        if CCx_loc_mat.ndim == 1:
            CCx_loc_mat = CCx_loc_mat.reshape(-1, 1)
        
        CCx_loc_mat_list.append(CCx_loc_mat)
    
    # Prepare pairwise comparisons
    CCxy_loc_mat_list = []
    idx = 0
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            # Extract connected location matrices
            CCx_loc_mat = CCx_loc_mat_list[i]
            CCy_loc_mat = CCx_loc_mat_list[j]
            
            # Store sample indices
            df_top_total.loc[idx, 'sample1'] = i
            df_top_total.loc[idx, 'sample2'] = j
            
            # Store location matrices for Jaccard computation
            if memory_efficient:
                # Use memory-efficient implementation
                CCx_sparse = sparse_connected_components(CCx_loc_mat, CCx_loc_mat.shape[0])
                CCy_sparse = sparse_connected_components(CCy_loc_mat, CCy_loc_mat.shape[0])
                CCxy_loc_mat_list.append((CCx_sparse, CCy_sparse))
            else:
                # Use standard implementation
                CCxy_loc_mat_list.append((CCx_loc_mat, CCy_loc_mat))
            
            idx += 1
    
    print("Connected location matrix extraction: %.2f seconds" % (time.time()-start_time))
    
    # Define compute_jaccard function based on optimization settings
    def compute_jaccard(CCxy_tuple):
        CCx, CCy = CCxy_tuple
        
        if memory_efficient:
            # Use memory-efficient implementation
            if use_numba and _HAS_NUMBA:
                # Use Numba-optimized implementation
                return compute_weighted_jaccard_similarity_numba(CCx, CCy)
            else:
                # Use standard memory-efficient implementation
                return merge_sparse_connected_components(CCx, CCy)
        else:
            # Use standard implementation
            if use_numba and _HAS_NUMBA:
                # Use Numba-optimized implementation
                return compute_jaccard_similarity_numba(CCx, CCy)
            else:
                # Use standard implementation
                return jaccard_composite(CCx, CCy)
    
    # Compute Jaccard similarity for each pair
    if use_numba and _HAS_NUMBA:
        output_j = optimized_parallel_processing(
            compute_jaccard, 
            CCxy_loc_mat_list,
            num_workers=int(max(1, min(os.cpu_count(), num_workers//1.5))), 
            progress_bar=progress_bar
        )
    else:
        output_j = parmap.starmap(
            compute_jaccard, 
            CCxy_loc_mat_list,
            pm_pbar=progress_bar, 
            pm_processes=int(max(1, min(os.cpu_count(), num_workers//1.5)))
        )
    
    # Create dataframe for J metrics
    output_j = pd.DataFrame(output_j, columns=['J_comp'])
    # Create dataframe with pairwise topological similarity measures
    df_top_total = pd.concat([df_top_total.iloc[:,:-1], output_j], axis=1)

    print("End of the whole process: %.2f seconds" % (time.time()-start_time))

    return df_top_total, data_mod