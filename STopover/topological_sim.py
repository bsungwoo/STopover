import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter

import os
import time
from .parallel_computing import *


def topological_sim_pairs_(data, feat_pairs, spatial_type = 'visium', group_list=None, group_name='batch',
                           fwhm=2.5, min_size=5, thres_per=30, jaccard_type='default',
                           num_workers=os.cpu_count()):
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
    * use_lr_db: whether to use list of features in L-R database (default = False)
    * lr_db_species: select species to utilize in L-R database (default = 'human')
    * db_name: name of the ligand-receptor database to use: either 'CellTalk', 'CellChat', or 'Omnipath' (default = 'CellTalk')

    * group_name:
        the column name for the groups saved in metadata(.obs)
        spatial data is divided according to the group and calculate topological overlap separately in each group
    * group_list: list of the elements in the group 
    * jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

    * J_result_name: the name of the jaccard index data file name
    * num_workers: number of workers to use for multiprocessing

    ### Output
    * df_top_total: dataframe that contains spatial overlap measures represented by (Jmax, Jmean, Jmmx, Jmmy) for the feature pairs 
    and average value for the feature across the spatial spots (if group is provided, then calculate average for the spots in each group)
    * data_mod: AnnData with summed location of all connected components in metadata(.obs) across all feature pairs
    '''
    start_time = time.time()

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
        except:
            group_name, group_list = 'group', ['0']
            data.obs[group_name] = '0'
    else:
        if group_name not in data.obs.columns:
            raise ValueError("'group_name' not found in columns of 'data.obs'")
        if not (set(group_list) <= set(data.obs[group_name])):
            raise ValueError("Some elements in 'group_list' not found in 'data.obs['"+str(group_name)+"']'")
        data = data[data.obs[group_name].isin(group_list)]

    # Determine the type of the data
    if isinstance(data.X, np.ndarray): data_type = 'array'
    elif isinstance(data.X, sparse.spmatrix): data_type = 'sparse'
    else: raise ValueError("'data.X' should be either numpy ndarray or scipy sparse matrix")

    # Change the column names if nan is included
    if np.nan in data.obs.columns: data.obs = data.obs.rename(columns=str)

    df_top_total = pd.DataFrame([])
    val_list, loc_list = [], []

    # Test the location of the data slot for first and second features (x and y) of the pairs
    obs_data_x, var_data_x = df_feat.iloc[:,0].isin(data.obs.columns), df_feat.iloc[:,0].isin(data.var.index)
    obs_data_y, var_data_y = df_feat.iloc[:,1].isin(data.obs.columns), df_feat.iloc[:,1].isin(data.var.index)
    obs_tf_x = obs_data_x.sum() > var_data_x.sum()
    obs_tf_y = obs_data_y.sum() > var_data_y.sum()

    if (not obs_tf_x) and (var_data_x.sum()==0):
        raise ValueError("None of the first features in the pairs are found")
    if (not obs_tf_y) and (var_data_y.sum()==0):
        raise ValueError("None of the second features in the pairs are found")

    # Filter the dataframe to contain only the existing data
    if obs_tf_x and obs_tf_y: df_feat = df_feat[obs_data_x & obs_data_y].reset_index(drop=True)
    elif (not obs_tf_x) and obs_tf_y: df_feat = df_feat[var_data_x & obs_data_y].reset_index(drop=True)
    elif obs_tf_x and (not obs_tf_y): df_feat = df_feat[obs_data_x & var_data_y].reset_index(drop=True)
    else: df_feat = df_feat[var_data_x & var_data_y].reset_index(drop=True)

    # Repeat the process for the proivded group_list
    for num, i in enumerate(group_list):
        data_sub = data[data.obs[group_name]==i]
        df_tmp = df_feat.copy()
        # Add the group name in the first column
        df_tmp.insert(0, group_name, i)

        # Check if the feature pairs are in the data and values are non-zero
        if obs_tf_x:
            # Load the data from the data_sub.obs
            data_x = data_sub.obs[df_tmp.iloc[:,1].tolist()].to_numpy()
            # Calculate average expression with absolute values
            df_tmp['Avg_1'] = np.mean(np.absolute(data_x), axis=0)
        else:
            data_x = None
            if data_type=='array':
                df_tmp['Avg_1'] = np.log1p(np.mean(np.expm1(data_sub[:,df_tmp.iloc[:,1].tolist()].X), axis=0))
            else:
                df_tmp['Avg_1'] = np.asarray(np.log1p(data_sub[:,df_tmp.iloc[:,1].tolist()].X.expm1().mean(axis = 0))).reshape(-1)
        if obs_tf_y:
            # Load the data from the data_sub.obs
            data_y = data_sub.obs[df_tmp.iloc[:,2].tolist()].to_numpy()
            # Calculate average expression with absolute values
            df_tmp['Avg_2'] = np.mean(np.absolute(data_y), axis=0)
        else:
            data_y = None
            if data_type=='array':
                df_tmp['Avg_2'] = np.log1p(np.mean(np.expm1(data_sub[:,df_tmp.iloc[:,2].tolist()].X), axis=0))
            else:
                df_tmp['Avg_2'] = np.asarray(np.log1p(data_sub[:,df_tmp.iloc[:,2].tolist()].X.expm1().mean(axis = 0))).reshape(-1)

        # Remove the features which have zero values only
        df_tmp = df_tmp[(df_tmp['Avg_1']!=0) & (df_tmp['Avg_2']!=0)]
        if df_tmp.shape[0] == 0: raise ValueError("In all feature pairs, more than one features has all-zero values")

        # Replace the absolute average values to the real average
        if data_x is not None: df_tmp['Avg_1'] = np.mean(data_x, axis=0)[df_tmp.index]
        if data_y is not None: df_tmp['Avg_2'] = np.mean(data_y, axis=0)[df_tmp.index]
        # Reset the index
        df_tmp = df_tmp.reset_index(drop=True)

        comb_feat_list = pd.concat([df_tmp.iloc[:,1],
                                    df_tmp.iloc[:,2]], axis=0).drop_duplicates().to_frame().set_index(0)
        comb_feat_list['index'] = range(len(comb_feat_list))
      
        # Find the index of the feature in Feat_1 and Feat_2 among the comb_feat_list.index
        df_x = comb_feat_list.loc[df_tmp.iloc[:,1].drop_duplicates()]
        df_y = comb_feat_list.loc[df_tmp.iloc[:,2].drop_duplicates()]
        comb_feat_list_x, comb_feat_list_y = df_x, df_y
        # Combine the dataframe df_x and df_y
        df_xy = pd.concat([df_x, df_y], axis=1)
        df_xy.columns = ['index_x','index_y']

        # Find the index for the Feature 1 and Feature 2: comb_feat_list as reference
        df_tmp['Index_1'] = df_xy.loc[df_tmp.iloc[:,1]].reset_index()['index_x'].astype(int)
        df_tmp['Index_2'] = df_xy.loc[df_tmp.iloc[:,2]].reset_index()['index_y'].astype(int)

        # Extract the non-overlapping feat list from group i and save index number corresponding to feature pairs
        if obs_tf_x != obs_tf_y:
            if obs_tf_x:
                val_x = data_sub.obs[comb_feat_list_x.index].to_numpy()
                if data_type=='array': val_element = np.concatenate((val_x, data_sub[:,comb_feat_list_y.index].X), axis=1)
                else: val_element = np.concatenate((val_x, data_sub[:,comb_feat_list_y.index].X.toarray()), axis=1)
            else:
                val_y = data_sub.obs[comb_feat_list_y.index].to_numpy()
                if data_type=='array': val_element = np.concatenate((data_sub[:,comb_feat_list_x.index].X, val_y), axis=1)
                else: val_element = np.concatenate((data_sub[:,comb_feat_list_x.index].X.toarray(), val_y), axis=1)
        else:
            # In case type of feature x and y is same
            # Define combined feature list for feat_1 and feat_2 and remove duplicates
            if obs_tf_x:
                val_element = data_sub.obs[comb_feat_list.index].to_numpy()
            else:
                if data_type=='array': val_element = data_sub[:,comb_feat_list.index].X
                else: val_element = data_sub[:,comb_feat_list.index].X.toarray()
        # Append the dataframe
        df_top_total = pd.concat([df_top_total, df_tmp], axis=0)

        # Add the location information of the spots
        try: df_loc = data_sub.obs.loc[:,['array_col','array_row']]
        except: raise ValueError("'data' should contain coordinates of spots in .obs as 'array_col' and 'array_row'")
        if spatial_type == 'visium':
            df_loc['array_row'] = df_loc['array_row']*np.sqrt(3)*0.5
            df_loc['array_col'] = df_loc['array_col']*0.5
        loc_list.append(df_loc.to_numpy())
        
        if spatial_type == 'visium':
            val_list.append(val_element)
        elif spatial_type in ['imageST', 'visiumHD']:
            sigma = fwhm / 2.355
            cols, rows = df_loc['array_col'].values, df_loc['array_row'].values

            # Convert val_element to a 3D array where each "slice" along the third axis corresponds to a feature
            # Map unique coordinates to indices
            unique_rows = np.unique(rows)
            unique_cols = np.unique(cols)
            row_indices = np.searchsorted(unique_rows, rows)
            col_indices = np.searchsorted(unique_cols, cols)
            # Initialize the array with the mapped dimensions
            arr = np.zeros((len(unique_rows), len(unique_cols), val_element.shape[1]))
            # Fill the array using the mapped indices
            arr[row_indices, col_indices, :] = val_element
            
            # Apply the Gaussian filter along the first two dimensions for each feature simultaneously
            smooth = gaussian_filter(arr, sigma=(sigma, sigma, 0), truncate=2.355, mode='constant')
            # Normalize the smoothed array
            smooth_sum = np.sum(smooth, axis=(0, 1), keepdims=True)
            val_element_sum = np.sum(val_element, axis=0, keepdims=True)
            smooth = smooth / smooth_sum * val_element_sum
                
            # Subset the smooth array using the original rows and cols indices
            smooth_subset = smooth[row_indices, col_indices, :]
            # Flatten the smoothed array along the first two dimensions
            smooth_subset = smooth_subset.reshape(-1, val_element.shape[1])    
            val_list.append(smooth_subset)

    # Make dataframe for the list of feature 1 and 2 across the groups
    column_names = [group_name,'Feat_1','Feat_2','Avg_1','Avg_2','Index_1','Index_2']
    df_top_total.columns = column_names
    df_top_total.index = range(df_top_total.shape[0])

    print('End of data preparation')
    print("Elapsed time: %.2f seconds " % (time.time()-start_time))

    # Start the multiprocessing for extracting adjacency matrix and mask
    loc_feat_pair = [(loc_list[grp_idx], feat[:,feat_idx].reshape((-1,1))) \
        for grp_idx, feat in enumerate(val_list) for feat_idx in range(feat.shape[1])]
    
    # Start the multiprocessing for finding connected components of each feature
    print("Calculation of connected components for each feature")
    output_cc = parallel_with_progress_topological_comp(locs = [feat[0] for feat in loc_feat_pair],
                                                        feats = [feat[1] for feat in loc_feat_pair],
                                                        spatial_type=spatial_type,
                                                        min_size=min_size, thres_per=thres_per, return_mode='cc_loc',
                                                        num_workers=int(max(1, min(os.cpu_count(), num_workers//1.5))))
    return output_cc, data
    # # Make dataframe for the similarity between feature 1 and 2 across the groups
    # print('Calculation of composite jaccard indexes between feature pairs')
    # CCxy_loc_mat_list = []; output_cc_loc=[]
    # feat_num_sum = 0
    # for num, element in enumerate(group_list):
    #     df_subset = df_top_total[df_top_total[group_name]==element]
    #     # Find the subset of the given data
    #     data_sub = data[data.obs[group_name]==element]
    #     # Add the connected component location of all features in each group
    #     feat_num = val_list[num].shape[1]
    #     arr_cc_loc = np.concatenate(output_cc[feat_num_sum:(feat_num_sum+feat_num)], axis=1)
    #     df_cc_loc = pd.DataFrame(arr_cc_loc)
    #     feat_num_sum += feat_num

    #     # Reconstruct combined feature list for each group
    #     comb_feat_list = pd.concat([df_subset['Feat_1'], df_subset['Feat_2']],
    #                                 axis=0, ignore_index=True).drop_duplicates().tolist()
    #     # Assign column names and index
    #     df_cc_loc.columns = ['Comb_CC_'+str(i) for i in comb_feat_list]
    #     df_cc_loc.index = data[data.obs[group_name]==group_list[num]].obs.index
    #     output_cc_loc.append(df_cc_loc)

    #     for index in range(len(df_subset)):
    #         CCx_loc_mat = arr_cc_loc[:,df_subset['Index_1'].iloc[index]]
    #         CCy_loc_mat = arr_cc_loc[:,df_subset['Index_2'].iloc[index]]
    #         if jaccard_type!="default":
    #             feat_x_val = val_list[num][:,df_subset['Index_1'].iloc[index]].reshape((-1,1))
    #             feat_y_val = val_list[num][:,df_subset['Index_2'].iloc[index]].reshape((-1,1))
    #         if jaccard_type=="default": CCxy_loc_mat_list.append((CCx_loc_mat,CCy_loc_mat,None,None))
    #         else: CCxy_loc_mat_list.append((CCx_loc_mat,CCy_loc_mat,feat_x_val,feat_y_val))

    # # Get the output for connected component location and save
    # data_mod = data
    # output_cc_loc = pd.concat(output_cc_loc, axis=0).fillna(0).astype(int).astype('category')
    # # Check if there is overlapping columns
    # import re
    # pattern = re.compile("^.*_prev[0-9]+$")
    # data_count = [int(i.split("_prev")[1]) for i in data_mod.obs.columns if pattern.match(i)]
    # if len(data_count) > 0: data_count = sorted(data_count)[-1] + 1
    # else: data_count = 1
    # # Add the connected component location information to the .obs
    # data_mod.obs = data_mod.obs.join(output_cc_loc, lsuffix='_prev'+str(data_count))

    # # Get the output for jaccard
    # output_j = parallel_with_progress_jaccard_composite(CCx_loc_sums=[feat[0] for feat in CCxy_loc_mat_list], 
    #                                                     CCy_loc_sums=[feat[1] for feat in CCxy_loc_mat_list],
    #                                                     feat_xs=[feat[2] for feat in CCxy_loc_mat_list],
    #                                                     feat_ys=[feat[3] for feat in CCxy_loc_mat_list],
    #                                                     num_workers=int(max(1, min(os.cpu_count(), num_workers//1.5))))
    # # Create dataframe for J metrics
    # output_j = pd.DataFrame(output_j, columns=['J_comp'])
    # # Create dataframe with pairwise topological similarity measures
    # df_top_total = pd.concat([df_top_total.iloc[:,:-2], output_j], axis=1)

    # print("End of the whole process: %.2f seconds" % (time.time()-start_time))

    # return df_top_total, data_mod