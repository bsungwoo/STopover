import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter
import os
import time
from .parallel_computing import (
    parallel_with_progress_extract_adjacency,
    parallel_with_progress_topological_comp,
    parallel_with_progress_jaccard_composite,
)

def topological_sim_pairs_(data, feat_pairs, spatial_type='visium', group_list=None, group_name='batch',
                           fwhm=2.5, min_size=5, thres_per=30, jaccard_type='default',
                           num_workers=os.cpu_count()):
    """
    Calculate Jaccard index between topological connected components of feature pairs.
    If groups are provided, data is processed per group.
    """
    start_time = time.time()

    # Validate and format feat_pairs.
    if isinstance(feat_pairs, pd.DataFrame):
        df_feat = feat_pairs.copy()
    elif isinstance(feat_pairs, list):
        df_feat = pd.DataFrame(feat_pairs)
    else:
        raise ValueError("'feat_pairs' should be a pandas DataFrame or a list")
    if df_feat.shape[1] != 2:
        raise ValueError("'feat_pairs' should be a list of pairs, e.g. [('A','B'), ('C','D')]")

    if spatial_type not in ['visium', 'imageST', 'visiumHD']:
        raise ValueError("'spatial_type' must be 'visium', 'imageST', or 'visiumHD'")
    if jaccard_type not in ['default', 'weighted']:
        raise ValueError("'jaccard_type' must be either 'default' or 'weighted'")

    # Determine group_list from data.obs if not provided.
    if group_list is None:
        try:
            group_list = data.obs[group_name].cat.categories
        except Exception:
            group_name, group_list = 'group', ['0']
            data.obs[group_name] = '0'
    else:
        if group_name not in data.obs.columns:
            raise ValueError(f"'{group_name}' not found in data.obs")
        if not (set(group_list) <= set(data.obs[group_name])):
            raise ValueError(f"Some elements in group_list not found in data.obs['{group_name}']")
        data = data[data.obs[group_name].isin(group_list)]

    # Determine the type of data.X.
    if isinstance(data.X, np.ndarray):
        data_type = 'array'
    elif sparse.isspmatrix(data.X):
        data_type = 'sparse'
    else:
        raise ValueError("'data.X' must be a numpy array or a scipy sparse matrix")

    # Rename columns if any are NaN.
    if any(pd.isnull(col) for col in data.obs.columns):
        data.obs = data.obs.rename(columns=str)

    df_top_total = pd.DataFrame()
    val_list, loc_list = [], []

    # Determine whether to use .obs or .var for features.
    obs_data_x = df_feat.iloc[:, 0].isin(data.obs.columns)
    var_data_x = df_feat.iloc[:, 0].isin(data.var.index)
    obs_data_y = df_feat.iloc[:, 1].isin(data.obs.columns)
    var_data_y = df_feat.iloc[:, 1].isin(data.var.index)
    obs_tf_x = obs_data_x.sum() > var_data_x.sum()
    obs_tf_y = obs_data_y.sum() > var_data_y.sum()

    if (not obs_tf_x) and (var_data_x.sum() == 0):
        raise ValueError("None of the first features are found")
    if (not obs_tf_y) and (var_data_y.sum() == 0):
        raise ValueError("None of the second features are found")

    # Keep only pairs that exist.
    if obs_tf_x and obs_tf_y:
        df_feat = df_feat[obs_data_x & obs_data_y].reset_index(drop=True)
    elif (not obs_tf_x) and obs_tf_y:
        df_feat = df_feat[var_data_x & obs_data_y].reset_index(drop=True)
    elif obs_tf_x and (not obs_tf_y):
        df_feat = df_feat[obs_data_x & var_data_y].reset_index(drop=True)
    else:
        df_feat = df_feat[var_data_x & var_data_y].reset_index(drop=True)

    # Process each group.
    for num, grp in enumerate(group_list):
        data_sub = data[data.obs[group_name] == grp]
        df_tmp = df_feat.copy()
        df_tmp.insert(0, group_name, grp)

        if obs_tf_x:
            data_x = data_sub.obs[df_tmp.iloc[:, 1].tolist()].to_numpy()
            df_tmp['Avg_1'] = np.mean(np.abs(data_x), axis=0)
        else:
            data_x = None
            if data_type == 'array':
                df_tmp['Avg_1'] = np.log1p(np.mean(np.expm1(data_sub[:, df_tmp.iloc[:, 1].tolist()].X), axis=0))
            else:
                df_tmp['Avg_1'] = np.asarray(np.log1p(np.mean(np.expm1(data_sub[:, df_tmp.iloc[:, 1].tolist()].X).toarray(), axis=0))).reshape(-1)

        if obs_tf_y:
            data_y = data_sub.obs[df_tmp.iloc[:, 2].tolist()].to_numpy()
            df_tmp['Avg_2'] = np.mean(np.abs(data_y), axis=0)
        else:
            data_y = None
            if data_type == 'array':
                df_tmp['Avg_2'] = np.log1p(np.mean(np.expm1(data_sub[:, df_tmp.iloc[:, 2].tolist()].X), axis=0))
            else:
                df_tmp['Avg_2'] = np.asarray(np.log1p(np.mean(np.expm1(data_sub[:, df_tmp.iloc[:, 2].tolist()].X).toarray(), axis=0))).reshape(-1)

        df_tmp = df_tmp[(df_tmp['Avg_1'] != 0) & (df_tmp['Avg_2'] != 0)]
        if df_tmp.empty:
            raise ValueError("After filtering, no valid feature pairs remain")
        if data_x is not None:
            df_tmp['Avg_1'] = np.mean(data_x, axis=0)[df_tmp.index]
        if data_y is not None:
            df_tmp['Avg_2'] = np.mean(data_y, axis=0)[df_tmp.index]
        df_tmp = df_tmp.reset_index(drop=True)

        # Build combined feature list.
        comb_feat_list = pd.concat([df_tmp.iloc[:, 1], df_tmp.iloc[:, 2]], axis=0).drop_duplicates().to_frame().set_index(0)
        comb_feat_list['index'] = range(len(comb_feat_list))
        df_x = comb_feat_list.loc[df_tmp.iloc[:, 1].drop_duplicates()]
        df_y = comb_feat_list.loc[df_tmp.iloc[:, 2].drop_duplicates()]
        df_xy = pd.concat([df_x, df_y], axis=1)
        df_xy.columns = ['index_x', 'index_y']
        df_tmp['Index_1'] = df_xy.loc[df_tmp.iloc[:, 1]].reset_index()['index_x'].astype(int)
        df_tmp['Index_2'] = df_xy.loc[df_tmp.iloc[:, 2]].reset_index()['index_y'].astype(int)

        if obs_tf_x != obs_tf_y:
            if obs_tf_x:
                val_x = data_sub.obs[comb_feat_list.index].to_numpy()
                if data_type == 'array':
                    val_element = np.concatenate((val_x, data_sub[:, comb_feat_list.index].X), axis=1)
                else:
                    val_element = np.concatenate((val_x, data_sub[:, comb_feat_list.index].X.toarray()), axis=1)
            else:
                val_y = data_sub.obs[comb_feat_list.index].to_numpy()
                if data_type == 'array':
                    val_element = np.concatenate((data_sub[:, comb_feat_list.index].X, val_y), axis=1)
                else:
                    val_element = np.concatenate((data_sub[:, comb_feat_list.index].X.toarray(), val_y), axis=1)
        else:
            if obs_tf_x:
                val_element = data_sub.obs[comb_feat_list.index].to_numpy()
            else:
                if data_type == 'array':
                    val_element = data_sub[:, comb_feat_list.index].X
                else:
                    val_element = data_sub[:, comb_feat_list.index].X.toarray()
        df_top_total = pd.concat([df_top_total, df_tmp], axis=0)

        # Get location info.
        try:
            df_loc = data_sub.obs.loc[:, ['array_col', 'array_row']]
        except Exception:
            raise ValueError("data.obs must contain 'array_col' and 'array_row'")
        if spatial_type == 'visium':
            df_loc['array_row'] = df_loc['array_row'] * np.sqrt(3) * 0.5
            df_loc['array_col'] = df_loc['array_col'] * 0.5
        loc_array = df_loc.to_numpy()
        # Debug: print shape of location array.
        print(f"Group {grp} loc shape: {loc_array.shape}")
        loc_list.append(loc_array)
        if spatial_type == 'visium':
            val_list.append(val_element)
        elif spatial_type in ['imageST', 'visiumHD']:
            sigma = fwhm / 2.355
            cols = df_loc['array_col'].values
            rows = df_loc['array_row'].values
            unique_rows = np.unique(rows)
            unique_cols = np.unique(cols)
            row_indices = np.searchsorted(unique_rows, rows)
            col_indices = np.searchsorted(unique_cols, cols)
            arr = np.zeros((len(unique_rows), len(unique_cols), val_element.shape[1]))
            arr[row_indices, col_indices, :] = val_element
            smooth = gaussian_filter(arr, sigma=(sigma, sigma, 0), truncate=2.355, mode='constant')
            smooth_sum = np.sum(smooth, axis=(0, 1), keepdims=True)
            val_element_sum = np.sum(val_element, axis=0, keepdims=True)
            smooth = smooth / smooth_sum * val_element_sum
            smooth_subset = smooth[row_indices, col_indices, :].reshape(-1, val_element.shape[1])
            val_list.append(smooth_subset)
    
    # Finalize dataframe columns.
    column_names = [group_name, 'Feat_1', 'Feat_2', 'Avg_1', 'Avg_2', 'Index_1', 'Index_2']
    df_top_total.columns = column_names
    df_top_total.index = range(df_top_total.shape[0])
    
    print('End of data preparation')
    print("Elapsed time: %.2f seconds" % (time.time() - start_time))

    # --- Parallel processing ---
    print("Calculation of adjacency matrix and mask")
    # Before parallel call, check that each loc has shape (n,2)
    for i, loc in enumerate(loc_list):
        if loc.ndim != 2 or loc.shape[1] != 2:
            raise ValueError(f"loc_list[{i}] has shape {loc.shape}; expected (n, 2)")
        else:
            print(f"loc_list[{i}] shape: {loc.shape}")
    
    try:
        # For debugging, you can temporarily pass progress_callback=lambda: None
        adjacency_mask = parallel_with_progress_extract_adjacency(
            loc_list, spatial_type=spatial_type, fwhm=fwhm,
            num_workers=max(1, int(num_workers // 1.5))
            #, progress_callback=lambda: None
        )
    except Exception as e:
        print("Error during parallel_extract_adjacency:", e)
        raise

    if spatial_type == 'visium':
        feat_A_mask_pair = [
            (feat[:, feat_idx].reshape((-1, 1)), adjacency_mask[grp_idx][0], adjacency_mask[grp_idx][1])
            for grp_idx, feat in enumerate(val_list)
            for feat_idx in range(feat.shape[1])
        ]
    else:
        feat_A_mask_pair = [
            (feat[:, feat_idx].reshape((-1, 1)), adjacency_mask[grp_idx], None)
            for grp_idx, feat in enumerate(val_list)
            for feat_idx in range(feat.shape[1])
        ]
    
    print("Calculation of connected components for each feature")
    output_cc = parallel_with_progress_topological_comp(
        feats=[feat[0] for feat in feat_A_mask_pair],
        A_matrices=[feat[1] for feat in feat_A_mask_pair],
        masks=[feat[2] for feat in feat_A_mask_pair],
        spatial_type=spatial_type,
        min_size=min_size, thres_per=thres_per, return_mode='cc_loc',
        num_workers=max(1, int(num_workers // 1.5))
        #, progress_callback=lambda: None
    )

    print('Calculation of composite jaccard indexes between feature pairs')
    CCxy_loc_mat_list = []
    output_cc_loc = []
    feat_num_sum = 0
    for num, grp in enumerate(group_list):
        data_sub = data[data.obs[group_name] == grp]
        feat_num = val_list[num].shape[1]
        
        # Get the relevant output_cc results
        group_output_cc = output_cc[feat_num_sum:(feat_num_sum + feat_num)]
        
        # Check if all elements have the same shape
        if not group_output_cc:
            print(f"Warning: No output_cc results for group {grp}")
            feat_num_sum += feat_num
            continue
            
        # Check if the results are tuples or arrays
        if isinstance(group_output_cc[0], tuple):
            # Extract the first element of each tuple (connected components)
            cc_arrays = [cc[0] for cc in group_output_cc]
        else:
            cc_arrays = group_output_cc
            
        # Check if we have arrays or lists of arrays
        if isinstance(cc_arrays[0], np.ndarray):
            # Direct arrays - concatenate them
            try:
                arr_cc_loc = np.concatenate(cc_arrays, axis=1)
            except ValueError as e:
                print(f"Error concatenating arrays: {e}")
                # Create an empty placeholder
                arr_cc_loc = np.zeros((1, 1))
        else:
            # Lists of arrays - need to handle differently
            try:
                # Flatten the lists into a single list of arrays
                flattened = []
                for cc_list in cc_arrays:
                    flattened.extend(cc_list)
                    
                # Convert to a 2D array if possible
                if flattened:
                    # Find max dimensions
                    max_rows = max(arr.shape[0] for arr in flattened if hasattr(arr, 'shape'))
                    
                    # Create a 2D array
                    arr_cc_loc = np.zeros((max_rows, len(flattened)))
                    
                    # Fill in the values
                    for i, arr in enumerate(flattened):
                        if hasattr(arr, 'shape') and arr.shape[0] > 0:
                            arr_cc_loc[:arr.shape[0], i] = arr
                else:
                    arr_cc_loc = np.zeros((1, 1))
            except Exception as e:
                print(f"Error processing lists of arrays: {e}")
                arr_cc_loc = np.zeros((1, 1))
        
        df_cc_loc = pd.DataFrame(arr_cc_loc)
        feat_num_sum += feat_num
        
        df_subset = df_top_total[df_top_total[group_name] == grp]
        comb_feat_list = pd.concat([df_subset['Feat_1'], df_subset['Feat_2']],
                                   axis=0, ignore_index=True).drop_duplicates().tolist()
        df_cc_loc.columns = ['Comb_CC_' + str(i) for i in comb_feat_list]
        df_cc_loc.index = data[data.obs[group_name] == grp].obs.index
        output_cc_loc.append(df_cc_loc)
        
        for idx in range(len(df_subset)):
            CCx_loc_mat = arr_cc_loc[:, df_subset['Index_1'].iloc[idx]]
            CCy_loc_mat = arr_cc_loc[:, df_subset['Index_2'].iloc[idx]]
            if jaccard_type != "default":
                feat_x_val = val_list[num][:, df_subset['Index_1'].iloc[idx]].reshape((-1, 1))
                feat_y_val = val_list[num][:, df_subset['Index_2'].iloc[idx]].reshape((-1, 1))
            if jaccard_type == "default":
                CCxy_loc_mat_list.append((CCx_loc_mat, CCy_loc_mat, None, None))
            else:
                CCxy_loc_mat_list.append((CCx_loc_mat, CCy_loc_mat, feat_x_val, feat_y_val))
    
    data_mod = data
    output_cc_loc = pd.concat(output_cc_loc, axis=0).fillna(0).astype(int).astype('category')
    import re
    pattern = re.compile("^.*_prev[0-9]+$")
    data_count = [int(i.split("_prev")[1]) for i in data_mod.obs.columns if pattern.match(i)]
    if data_count:
        data_count = sorted(data_count)[-1] + 1
    else:
        data_count = 1
    data_mod.obs = data_mod.obs.join(output_cc_loc, lsuffix='_prev' + str(data_count))
    
    try:
        output_j = parallel_with_progress_jaccard_composite(
            CCx_loc_sums=[feat[0] for feat in CCxy_loc_mat_list],
            CCy_loc_sums=[feat[1] for feat in CCxy_loc_mat_list],
            feat_xs=[feat[2] for feat in CCxy_loc_mat_list],
            feat_ys=[feat[3] for feat in CCxy_loc_mat_list],
            num_workers=max(1, int(num_workers // 1.5))
            #, progress_callback=lambda: None
        )
    except Exception as e:
        print("Error during parallel_jaccard_composite:", e)
        raise

    output_j = pd.DataFrame(output_j, columns=['J_comp'])
    df_top_total = pd.concat([df_top_total.iloc[:, :-2], output_j], axis=1)
    
    print("End of the whole process: %.2f seconds" % (time.time() - start_time))
    return df_top_total, data_mod