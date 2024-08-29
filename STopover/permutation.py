import numpy as np
import pandas as pd
from scipy import sparse

import os
import time
import parmap

from .topological_comp import extract_adjacency_spatial
from .topological_comp import topological_comp_res

from .jaccard import jaccard_composite


def shuffle_generator(arr, seed):
    """
    Generator function that yields a shuffled version of the input 1D NumPy array
    with a fixed shuffle pattern for a given seed.
    
    Parameters:
    arr (np.ndarray): A 1D NumPy array to shuffle.
    seed (int): The seed for the random number generator.
    
    Yields:
    np.ndarray: Shuffled version of the input array.
    """
    rng = np.random.default_rng(seed)  # Create a random generator with the given seed
    while True:
        # Shuffle the array using the fixed random seed
        shuffled_arr = rng.permutation(arr)  # This creates a shuffled copy of the array
        # Yield the shuffled array
        yield shuffled_arr
    

def calculate_p_value(group):
    """
    Calculate p-value based on the null distribution of J_comp_perm for a given group.
    """
    # Extract the observed value from the 'J_comp' column
    observed_value = group['J_comp'].iloc[0]
    
    # Calculate the right-tailed p-value: proportion of permuted values >= observed value
    p_value = np.mean(group['J_comp_perm'] >= observed_value)
    
    # Assign the p-value (it will be the same for all rows in the group, but you can assign it to the first row)
    return pd.Series({'p_value': p_value})


def run_permutation_test(data, feat_pairs, nperm=1000, seed=0, spatial_type = 'visium',
                         fwhm=2.5, min_size=5, thres_per=30, jaccard_type='default',
                         num_workers=os.cpu_count(), progress_bar=True):
    '''
    ## Calculate Jaccard index for given feature pairs and return dataframe
        -> if the group is given, divide the spatial data according to the group and calculate topological overlap separately in each group

    ### Input
    data: spatial data (format: anndata) containing log-normalized gene expression
    feat_pairs:
        list of features with the format [('A','B'),('C','D')] or the pandas equivalent
        -> (A and C) should be same data format: all in metadata (.obs.columns) or all in gene names(.var.index)
        -> (C and D) should be same data format: all in metadata (.obs.columns) or all in gene names(.var.index)
        -> If the data format is not same the majority of the data format will be automatically searched
        -> and the rest of the features with different format will be removed from the pairs
    nperm: number of the random permutation (default: 1000)
    seed: the seed for the random number generator (default: 0)
    spatial_type: type of the spatial data (should be either 'visium' or 'imageST')

    group_name:
        the column name for the groups saved in metadata(.obs)
        spatial data is divided according to the group and calculate topological overlap separately in each group
    group_list: list of the name of groups

    fwhm: full width half maximum value for the gaussian smoothing kernel as the multiple of the central distance between the adjacent spots/grids
    min_size: minimum size of a connected component
    thres_per: lower percentile value threshold to remove the connected components
    jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)
    num_workers: number of workers to use for multiprocessing
    progress_bar: whether to show the progress bar during multiprocessing

    ### Output
    df_top_total: dataframe that contains spatial overlap measures represented by (Jmax, Jmean, Jmmx, Jmmy) for the feature pairs
    and average value for the feature across the spatial spots (if group is provided, then calculate average for the spots in each group)
    data_mod: AnnData with summed location of all connected components in metadata(.obs) across all feature pairs
    '''
    start_time = time.time()

    # Check the format of the feature pairs
    if sum(1 if i.startswith('Comb_CC_') else 0 for i in data.obs.columns) == 0:
        raise ValueError("No CC locations in 'data': permutation test cannot be performed")

    import re
    pattern = re.compile("^J_.*_[0-9]$")
    adata_keys = sorted([i for i in data.uns.keys() if pattern.match(i)])
    if len(adata_keys) > 0:
        feat_ref = data.uns[adata_keys[-1]]
    else:
        raise ValueError("No previously calculated connected components")

    group_name = feat_ref.columns.values[0]
    if group_name in ['Feat_1','Feat_2','Avg_1','Avg_2','Index_1','Index_2', 
                      'fwhm', 'min_size','fwhm', 'percent']:
        raise NotImplementedError(f"Batch name not found from adata.uns['{adata_keys[-1]}']")

    df_feat = feat_ref
    if feat_pairs is not None:
        if isinstance(feat_pairs, pd.DataFrame):
            if feat_pairs.shape[1] != 2:
                raise ValueError("'feat_pairs' should be list format: [('A','B'),('C','D')] or equivalent pandas dataframe") 
            feat_pairs.columns.values[0], feat_pairs.columns.values[1] = 'Feat_1', 'Feat_2'
            df_feat = pd.merge(feat_ref, feat_pairs.iloc[:,:2], left_on=['Feat_1','Feat_2'], right_on=['Feat_1','Feat_2'], how='inner')
        elif isinstance(feat_pairs, list): 
            df_sub = pd.DataFrame(feat_pairs, columns=['Feat_1','Feat_2'])
            if df_sub.shape[1] != 2:
                raise ValueError("'feat_pairs' should be list format: [('A','B'),('C','D')] or equivalent pandas dataframe") 
            df_feat = pd.merge(feat_ref, df_sub, 
                               left_on=['Feat_1','Feat_2'], right_on=['Feat_1','Feat_2'], how='inner')
        else:
            raise ValueError("'feat_pairs' should be pandas dataframe or list")
    group_list = np.intersect1d(pd.unique(df_feat[group_name]), pd.unique(data.obs[group_name]))

    # Check the format of the data and jaccard output type
    if spatial_type not in ['visium', 'imageST']: raise ValueError("'spatial_type' should be either 'visium' or 'imageST'")
    if jaccard_type not in ['default', 'weighted']: raise ValueError("'jaccard_type' should be either 'default' or 'weighted'")

    # Determine the type of the data
    if isinstance(data.X, np.ndarray): data_type = 'array'
    elif isinstance(data.X, sparse.spmatrix): data_type = 'sparse'
    else: raise ValueError("'data.X' should be either numpy ndarray or scipy sparse matrix")

    df_top_total = pd.DataFrame([])
    val_list, loc_list = [], []

    # Test the location of the data slot for first and second features (x and y) of the pairs
    obs_data_x, var_data_x = df_feat.loc[:,'Feat_1'].isin(data.obs.columns), df_feat.loc[:,'Feat_1'].isin(data.var.index)
    obs_data_y, var_data_y = df_feat.loc[:,'Feat_2'].isin(data.obs.columns), df_feat.loc[:,'Feat_2'].isin(data.var.index)
    obs_tf_x = obs_data_x.sum() > var_data_x.sum()
    obs_tf_y = obs_data_y.sum() > var_data_y.sum()

    # Repeat the process for the proivded group_list
    for num, i in enumerate(group_list):
        data_sub = data[data.obs[group_name]==i]
        df_tmp = df_feat.loc[df_feat[group_name]==i]

        comb_feat_list = pd.concat([df_tmp.loc[:,"Feat_1"], df_tmp.loc[:,"Feat_2"]], axis=0).drop_duplicates().to_frame().set_index(0)
        comb_feat_list['index'] = range(len(comb_feat_list))

        # Find the index of the feature in Feat_1 and Feat_2 among the comb_feat_list.index
        df_x = comb_feat_list.loc[df_tmp.loc[:,"Feat_1"].drop_duplicates()]
        df_y = comb_feat_list.loc[df_tmp.loc[:,"Feat_2"].drop_duplicates()]
        comb_feat_list_x, comb_feat_list_y = df_x, df_y
        # Combine the dataframe df_x and df_y
        df_xy = pd.concat([df_x, df_y], axis=1)
        df_xy.columns = ['index_x','index_y']

        # Find the index for the Feature 1 and Feature 2: comb_feat_list as reference
        df_tmp['Index_1'] = df_xy.loc[df_tmp.iloc[:,1]].reset_index()['index_x'].astype(int).values
        df_tmp['Index_2'] = df_xy.loc[df_tmp.iloc[:,2]].reset_index()['index_y'].astype(int).values

        # Initialize the shuffle generator with the length of the DataFrame
        shuffle_gen = shuffle_generator(np.arange(len(data_sub)), seed=seed)

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
        val_list.append([val_element[next(shuffle_gen),:] for _ in range(nperm)])
        # Append the dataframe
        df_top_total = pd.concat([df_top_total, df_tmp], axis=0)

        # Add the location information of the spots
        try: df_loc = data_sub.obs.loc[:,['array_col','array_row']]
        except: raise ValueError("'data' should contain coordinates of spots in .obs as 'array_col' and 'array_row'")
        if spatial_type == 'visium':
            df_loc['array_row'] = df_loc['array_row']*np.sqrt(3)*0.5
            df_loc['array_col'] = df_loc['array_col']*0.5
        
        # Generate a list of location arrays from the DataFrame
        loc_list.append(df_loc.to_numpy())

    # Make dataframe for the list of feature 1 and 2 across the groups
    df_top_total.index = range(df_top_total.shape[0])
    print('End of data preparation')
    print("Elapsed time: %.2f seconds " % (time.time()-start_time))

    # Start the multiprocessing for extracting adjacency matrix and mask
    print("Calculation of adjacency matrix and mask")
    adjacency_mask = parmap.map(extract_adjacency_spatial, loc_list, spatial_type=spatial_type, fwhm=fwhm,
                                pm_pbar=progress_bar, pm_processes=int(max(1, min(os.cpu_count(), num_workers//1.5))), pm_chunksize=50)
    if spatial_type=='visium':
        feat_A_mask_pair = [(feat[perm_idx][:,feat_idx].reshape((-1,1)),
                            adjacency_mask[grp_idx][0],
                            adjacency_mask[grp_idx][1],None,None) \
                            for grp_idx, feat in enumerate(val_list) \
                            for perm_idx in range(nperm) for feat_idx in range(feat[0].shape[1])]
    else:
        feat_A_mask_pair = [(feat[perm_idx][:,feat_idx].reshape((-1,1)),
                            adjacency_mask[grp_idx][0],
                            adjacency_mask[grp_idx][1],
                            len(np.unique(loc_list[grp_idx][0,:])),
                            len(np.unique(loc_list[grp_idx][1,:]))) \
                            for grp_idx, feat in enumerate(val_list) \
                            for perm_idx in range(nperm) for feat_idx in range(feat[0].shape[1])]

    # Start the multiprocessing for finding connected components of each feature
    print("Calculation of connected components for each feature")
    output_cc = parmap.starmap(topological_comp_res, feat_A_mask_pair, spatial_type=spatial_type,
                               fwhm=fwhm, min_size=min_size, thres_per=thres_per, return_mode='cc_loc',
                               pm_pbar=progress_bar, pm_processes=int(max(1, min(os.cpu_count(), num_workers//1.5))))

    # Make dataframe for the similarity between feature 1 and 2 across the groups
    print('Calculation of composite jaccard indexes between feature pairs')
    CCxy_loc_mat_list, output_cc_loc, df_perm = [], [], pd.DataFrame([])
    feat_num_sum = 0
    for num, element in enumerate(group_list):
        df_subset = df_top_total[df_top_total[group_name]==element]
        # Find the subset of the given data
        data_sub = data[data.obs[group_name]==element]
        feat_num = (val_list[num][0].shape[1])
        for perm_idx in range(nperm):
            # Add the connected component location of all features in each group
            arr_cc_loc = np.concatenate(output_cc[(feat_num_sum+feat_num*perm_idx):(feat_num_sum+feat_num*(perm_idx+1))], axis=1)
            df_cc_loc = pd.DataFrame(arr_cc_loc)

            # Reconstruct combined feature list for each group
            comb_feat_list = pd.concat([df_subset['Feat_1'], df_subset['Feat_2']],
                                        axis=0, ignore_index=True).drop_duplicates().tolist()
            # Assign column names and index
            df_cc_loc.columns = ['Comb_CC_'+str(i) for i in comb_feat_list]
            df_cc_loc = df_cc_loc.assign(barcode=data[data.obs[group_name] == group_list[num]].obs.index,
                                         perm_idx=perm_idx)
            output_cc_loc.append(df_cc_loc)
            
            for index in range(len(df_subset)):
                CCx_loc_mat = arr_cc_loc[:,df_subset['Index_1'].iloc[index]]
                CCy_loc_mat = arr_cc_loc[:,df_subset['Index_2'].iloc[index]]
                if jaccard_type=="default": 
                    CCxy_loc_mat_list.append((CCx_loc_mat,CCy_loc_mat))
                else: 
                    feat_x_val = val_list[num][perm_idx][:,df_subset['Index_1'].iloc[index]].reshape((-1,1))
                    feat_y_val = val_list[num][perm_idx][:,df_subset['Index_2'].iloc[index]].reshape((-1,1))
                    CCxy_loc_mat_list.append((CCx_loc_mat,CCy_loc_mat,feat_x_val,feat_y_val))
        
        df_perm = pd.concat([df_perm, pd.concat([df_subset.loc[:,[group_name,"Feat_1","Feat_2","J_comp"]]] * nperm, 
                                                axis=0)], axis=0)
        feat_num_sum += feat_num*nperm

    # Get the output for connected component location and save
    data_mod = data
    output_cc_loc = pd.concat(output_cc_loc, axis=0).reset_index(drop=True)
    output_cc_loc[[i for i in output_cc_loc.columns if i not in ['barcode','perm_idx']]] = \
        output_cc_loc[[i for i in output_cc_loc.columns if i not in ['barcode','perm_idx']]].fillna(0).astype(pd.SparseDtype("int", fill_value=0)).astype('category')
    # Add the connected component location information to the .obs
    data_mod.uns[f"cc_loc_{adata_keys[-1]}_perm"] = output_cc_loc

    # Get the output for jaccard
    output_j = parmap.starmap(jaccard_composite, CCxy_loc_mat_list,
                              pm_pbar=progress_bar,
                              pm_processes=int(max(1, min(os.cpu_count(), num_workers)//1.5)))

    # Create a dataframe for J metrics and calculate p-values
    df_perm_fin = df_perm.assign(J_comp_perm=output_j).groupby([group_name, 'Feat_1', 'Feat_2']).apply(calculate_p_value).reset_index()

    # Now merge the p-values with df_top_total based on group_name, Feat_1, and Feat_2
    df_top_total = df_top_total.iloc[:,:-2].merge(df_perm_fin[[group_name, 'Feat_1', 'Feat_2', 'p_value']], 
                                                  on=[group_name, 'Feat_1', 'Feat_2'], how='left')

    print("End of the whole process: %.2f seconds" % (time.time()-start_time))

    return df_top_total, data_mod
