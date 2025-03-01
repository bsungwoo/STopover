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
import scipy.sparse as sparse

from .topological_comp import extract_adjacency_spatial, topological_comp_res
from .topological_comp import extract_connected_loc_mat
from .jaccard import jaccard_composite

from .numba_optimizations import (
    extract_adjacency_spatial_numba, 
    compute_jaccard_similarity_numba, 
    compute_weighted_jaccard_similarity_numba,
    topological_comp_res_numba
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

def extract_adjacency_wrapper(loc, spatial_type='visium', fwhm=2.5):
    """Wrapper function for extract_adjacency_spatial that works with both numba and non-numba versions"""
    if _HAS_NUMBA:
        try:
            # Disable Numba threading when using multiprocessing
            old_threading_layer = numba.get_num_threads()
            numba.set_num_threads(1)
            result = extract_adjacency_spatial_numba(loc, spatial_type, fwhm)
            numba.set_num_threads(old_threading_layer)
            return result
        except:
            return extract_adjacency_spatial(loc, spatial_type=spatial_type, fwhm=fwhm)
    else:
        return extract_adjacency_spatial(loc, spatial_type=spatial_type, fwhm=fwhm)

def jaccard_wrapper(args, jaccard_type='default'):
    """Wrapper function for jaccard similarity calculation that works with both numba and non-numba versions"""
    if _HAS_NUMBA:
        try:
            # Disable Numba threading when using multiprocessing
            old_threading_layer = numba.get_num_threads()
            numba.set_num_threads(1)
            
            if jaccard_type == 'weighted' and len(args) == 4:
                CCx_loc_mat, CCy_loc_mat, feat_x_val, feat_y_val = args
                result = compute_weighted_jaccard_similarity_numba(CCx_loc_mat, CCy_loc_mat, feat_x_val, feat_y_val)
            else:
                CCx_loc_mat, CCy_loc_mat = args[0], args[1]
                result = compute_jaccard_similarity_numba(CCx_loc_mat, CCy_loc_mat)
                
            numba.set_num_threads(old_threading_layer)
            return result
        except:
            return jaccard_composite(*args)
    else:
        return jaccard_composite(*args)

def topological_sim_pairs_(data, feat_pairs, spatial_type='visium', group_list=None, group_name='batch',
                          fwhm=2.5, min_size=5, thres_per=30, jaccard_type='default', J_result_name='result',
                          num_workers=os.cpu_count(), progress_bar=True, use_numba=True):
    '''
    ## Calculate Jaccard index between topological connected components of feature pairs and return dataframe
        : if the group is given, divide the spatial data according to the group and calculate topological overlap separately in each group

    ### Input
    * data: spatial data (format: anndata) containing log-normalized gene expression
    * feat_pairs: 
        list of features with the format [('A','B'),('C','D')] or the pandas equivalent
        -> (A and C) should be same data format: all in metadata (.obs.columns) or all in gene names(.var.index)
        -> (C and D) should be same data format: all in metadata (.obs.columns) or all in gene names(.var.index)
        -> If the data format is not same the majority of the data format will be automatically searched
        -> and the rest of the features with different format will be removed from the pairs
    * spatial_type: type of the spatial data (should be either 'visium', 'imageST', or 'visiumHD')
    * group_name:
        the column name for the groups saved in metadata(.obs)
        spatial data is divided according to the group and calculate topological overlap separately in each group
    * group_list: list of the elements in the group 
    * jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)
    * J_result_name: the name of the jaccard index data file name
    * num_workers: number of workers to use for multiprocessing
    * progress_bar: whether to show the progress bar during multiprocessing
    * use_numba: whether to use Numba optimization (default: True)

    ### Output
    * df_top_total: dataframe that contains spatial overlap measures represented by (Jmax, Jmean, Jmmx, Jmmy) for the feature pairs 
    and average value for the feature across the spatial spots (if group is provided, then calculate average for the spots in each group)
    * data_mod: AnnData with summed location of all connected components in metadata(.obs) across all feature pairs
    '''
    start_time = time.time()

    # Check if we should use Numba
    if use_numba and _HAS_NUMBA:
        # Import Numba-related functions
        try:
            from .numba_optimizations import (
                extract_adjacency_spatial_numba, 
                compute_jaccard_similarity_numba, 
                compute_weighted_jaccard_similarity_numba,
                topological_comp_res_numba
            )
            # Set Numba to use a single thread to avoid conflicts with multiprocessing
            import numba
            old_threading_layer = numba.get_num_threads()
            numba.set_num_threads(1)
        except ImportError:
            # If any function is missing, disable Numba
            use_numba = False
    
    # Check the format of the feature pairs
    if isinstance(feat_pairs, pd.DataFrame): df_feat = feat_pairs
    elif isinstance(feat_pairs, list): df_feat = pd.DataFrame(feat_pairs)
    else: raise ValueError("'feat_pairs' should be pandas dataframe or list")
    if df_feat.shape[1] != 2:
        raise ValueError("'feat_pairs' should be list format: [('A','B'),('C','D')] or equivalent pandas dataframe")

    # Check the format of the data and jaccard output type
    if spatial_type not in ['visium', 'imageST', 'visiumHD']: raise ValueError("'spatial_type' should be either 'visium', 'imageST', or 'visiumHD'")
    if jaccard_type not in ['default', 'weighted']: raise ValueError("'jaccard_type' should be either 'default' or 'weighted'")

    # Add group name if no group name is provided
    if group_list is None:
        try: group_list = data.obs[group_name].cat.categories
        except: group_list = [None]
    
    # Create dataframe for the similarity between feature 1 and 2 across the groups
    print('End of data preparation')
    print('Elapsed time: %.2f seconds ' % (time.time()-start_time))
    print('Calculation of adjacency matrix and mask')
    
    # Extract spatial coordinates
    loc_list = []
    for element in group_list:
        if element is None: loc_list.append(data.obsm['spatial'])
        else: loc_list.append(data[data.obs[group_name]==element].obsm['spatial'])
    
    # Extract adjacency matrix and mask
    adjacency_mask = []
    for loc in loc_list:
        if use_numba and _HAS_NUMBA:
            try:
                A, mask = extract_adjacency_spatial_numba(loc, spatial_type, fwhm)
                adjacency_mask.append((A, mask))
            except:
                A, mask = extract_adjacency_spatial(loc, spatial_type, fwhm)
                adjacency_mask.append((A, mask))
        else:
            A, mask = extract_adjacency_spatial(loc, spatial_type, fwhm)
            adjacency_mask.append((A, mask))
    
    # Extract feature values
    print('Calculation of connected components for each feature')
    val_list = []
    df_top_total = pd.DataFrame()  # Initialize df_top_total
    
    for num, element in enumerate(group_list):
        # Extract feature values
        if element is None: data_sub = data
        else: data_sub = data[data.obs[group_name]==element]
        
        # Extract feature values for each feature
        feat_list = pd.concat([df_feat.iloc[:,0], df_feat.iloc[:,1]], axis=0).drop_duplicates().tolist()
        val_arr = np.zeros((data_sub.shape[0], len(feat_list)))
        
        # Add index for feature 1 and 2
        # Create DataFrame with proper index to avoid ValueError
        df_top = pd.DataFrame({
            'Feat_1': [feat_list[0]] * len(df_feat),
            'Feat_2': [feat_list[1]] * len(df_feat)
        }, index=range(len(df_feat)))
        
        df_top['Index_1'] = df_top['Feat_1'].apply(lambda x: feat_list.index(x))
        df_top['Index_2'] = df_top['Feat_2'].apply(lambda x: feat_list.index(x))
        
        # Add mean value for feature 1 and 2
        df_top['Mean_1'] = df_top['Index_1'].apply(lambda x: np.mean(val_arr[:,x]))
        df_top['Mean_2'] = df_top['Index_2'].apply(lambda x: np.mean(val_arr[:,x]))
        
        # Add group information
        if element is not None: df_top[group_name] = element
        
        # Add to total dataframe
        df_top_total = pd.concat([df_top_total, df_top], axis=0)
        
        # Create feature-adjacency-mask pairs
        feat_A_mask_pair = []  # Initialize the list here
        for i, feat in enumerate(feat_list):
            feat_A_mask_pair.append((val_arr[:,i], sparse.csr_matrix(adjacency_mask[num][0]), adjacency_mask[num][1]))
    
    # Use sequential processing for connected components
    output_cc = []
    for args in feat_A_mask_pair:
        # Use Numba-optimized function if available
        if use_numba and _HAS_NUMBA:
            try:
                result = topological_comp_res_numba(
                    feat=args[0], A=args[1], mask=args[2],
                    spatial_type=spatial_type, min_size=min_size,
                    thres_per=thres_per, return_mode='cc_loc'
                )
            except:
                # Fall back to standard implementation
                result = topological_comp_res(
                    feat=args[0], A=args[1], mask=args[2],
                    spatial_type=spatial_type, min_size=min_size,
                    thres_per=thres_per, return_mode='cc_loc'
                )
        else:
            # Always use standard implementation if Numba is not available
            result = topological_comp_res(
                feat=args[0], A=args[1], mask=args[2],
                spatial_type=spatial_type, min_size=min_size,
                thres_per=thres_per, return_mode='cc_loc'
            )
        output_cc.append(result)

    # Make dataframe for the similarity between feature 1 and 2 across the groups
    print('Calculation of composite jaccard indexes between feature pairs')
    CCxy_loc_mat_list = []; output_cc_loc=[]
    feat_num_sum = 0
    for num, element in enumerate(group_list):
        df_subset = df_top_total[df_top_total[group_name]==element] if element is not None else df_top_total
        # Find the subset of the given data
        data_sub = data[data.obs[group_name]==element] if element is not None else data
        # Add the connected component location of all features in each group
        feat_num = val_list[num].shape[1]
        arr_cc_loc = np.concatenate(output_cc[feat_num_sum:(feat_num_sum+feat_num)], axis=1)
        df_cc_loc = pd.DataFrame(arr_cc_loc)
        feat_num_sum += feat_num

        # Reconstruct combined feature list for each group
        comb_feat_list = pd.concat([df_subset['Feat_1'], df_subset['Feat_2']],
                                    axis=0, ignore_index=True).drop_duplicates().tolist()
        # Assign column names and index
        df_cc_loc.columns = ['Comb_CC_'+str(i) for i in comb_feat_list]
        df_cc_loc.index = data_sub.obs.index
        output_cc_loc.append(df_cc_loc)

        for index in range(len(df_subset)):
            CCx_loc_mat = arr_cc_loc[:,df_subset['Index_1'].iloc[index]]
            CCy_loc_mat = arr_cc_loc[:,df_subset['Index_2'].iloc[index]]
            if jaccard_type!="default":
                feat_x_val = val_list[num][:,df_subset['Index_1'].iloc[index]].reshape((-1,1))
                feat_y_val = val_list[num][:,df_subset['Index_2'].iloc[index]].reshape((-1,1))
            if jaccard_type=="default": CCxy_loc_mat_list.append((CCx_loc_mat,CCy_loc_mat))
            else: CCxy_loc_mat_list.append((CCx_loc_mat,CCy_loc_mat,feat_x_val,feat_y_val))

    # Get the output for connected component location and save
    data_mod = data.copy()
    output_cc_loc = pd.concat(output_cc_loc, axis=0).fillna(0).astype(int).astype('category')
    # Check if there is overlapping columns
    import re
    pattern = re.compile("^.*_prev[0-9]+$")
    data_count = [int(i.split("_prev")[1]) for i in data_mod.obs.columns if pattern.match(i)]
    if len(data_count) > 0: data_count = sorted(data_count)[-1] + 1
    else: data_count = 1
    # Add the connected component location information to the .obs
    data_mod.obs = data_mod.obs.join(output_cc_loc, lsuffix='_prev'+str(data_count))

    # Get the output for jaccard - use sequential processing
    output_j = []
    for args in CCxy_loc_mat_list:
        if use_numba and _HAS_NUMBA and jaccard_type == 'default':
            try:
                CCx_loc_mat, CCy_loc_mat = args[0], args[1]
                result = compute_jaccard_similarity_numba(CCx_loc_mat, CCy_loc_mat)
            except:
                result = jaccard_composite(*args)
        elif use_numba and _HAS_NUMBA and jaccard_type == 'weighted':
            try:
                CCx_loc_mat, CCy_loc_mat, feat_x_val, feat_y_val = args
                result = compute_weighted_jaccard_similarity_numba(CCx_loc_mat, CCy_loc_mat, feat_x_val, feat_y_val)
            except:
                result = jaccard_composite(*args)
        else:
            result = jaccard_composite(*args)
        output_j.append(result)
    
    # Create dataframe for J metrics
    output_j = pd.DataFrame(output_j, columns=['J_comp'])
    # Create dataframe with pairwise topological similarity measures
    df_top_total = pd.concat([df_top_total.iloc[:,:-2], output_j], axis=1)

    # Save the result to the data.uns
    data_mod.uns[f"J_result_{J_result_name}"] = df_top_total

    # Restore Numba threading if we changed it
    if use_numba and _HAS_NUMBA:
        numba.set_num_threads(old_threading_layer)

    print("End of the whole process: %.2f seconds" % (time.time()-start_time))

    return df_top_total, data_mod