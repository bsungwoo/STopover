import numpy as np
import numpy.matlib
import pandas as pd
from scipy import sparse

import os
import time
import multiprocessing as mp
            
from .topological_comp import extract_adjacency_spatial
from .topological_comp import extract_connected_comp
from .topological_comp import extract_connected_loc_arr
from .topological_comp import add_connected_loc
from .topological_comp import filter_connected_loc_exp

from .jaccard import jaccard_and_connected_loc_



def topological_sim_pair(data=None, feat_x=None, feat_y=None, fwhm = 2.5, min_size = 5, thres_per=30, 
                         J_metric=False, return_mode='all', A=None, mask=None):
    '''
    ## Calculate topological connected components and similarity between feature x and y
    ### Input
    feat_x, feat_y: two features to investigate topological similarity 
    -> represents the name of the genes when data is provided
    -> represents the feature values as numpy array when data is not provided (number of spots * 1 array)
    fwhm: full width half maximum value for the gaussian smoothing kernal
    min_size: minimum size of a connected component
    thres_per: percentile expression threshold to remove the connected components
    J_metric: whether to calculate Jaccard index (Jmax, Jmean, Jmmx, Jmmy) between CCx and CCy pair

    return_mode: mode of return
    'all': return jaccard index result along with array for location of connected components
    'cc_loc': return array or Anndata containing location of connected components
    'jaccard_cc_list': return jaccard index result only (when J_metric is True, then return jaccard metrics, and when false, return jaccard similarity array with CC locations)

    A: spatial adjacency matrix, mask: mask for gaussian smoothing

    ### Output
    result:
    -> if J_metric is True, then return (Jmax, Jmean, Jmmx, Jmmy): maximum jaccard index, mean jaccard index and mean of maximum jaccard for CCx and CCy
    -> if J_metric is False, then return jaccard similarity array between CCx and CCy along with CCx and CCy locations as below format
    CCx location: list containing index of spots indicating location of connected components for feature x
    CCy location: list containing index of spots indicating location of connected components for feature y

    data_fin: AnnData with the summed location of all connected components in feature x and y in metadata(.obs)
    -> if data is None, then the summed location array of all connected components are returned
    '''
    # Check feasibility of the dataset
    if not (return_mode in ['all','cc_loc','jaccard_cc_list']):
        raise ValueError("'return_mode' should be among 'all', 'cc_loc', or 'jaccard_cc_list'")

    # If no dataset is given, then feature x and feature y should be provided as numpy arrays
    if data is None: 
        if (A is None) or (mask is None):
            raise ValueError("'A' and 'mask' should be provided when 'data' is None")
        if not isinstance(feat_x, np.ndarray):
            raise ValueError("Values for 'feat_x' and 'feat_y' should be provided as numpy arrays when 'data' is None")

    # Define feature list, feature value list, and connected component list
    feature_list = [feat_x, feat_y]
    value_list = []; CC = []

    ## Import dataset and check the feasibility
    if data is not None:
         for ind, feat in enumerate(feature_list):
            # Test the location of the data slot
            obs_data = feat in data.obs.columns
            var_data = feat in data.var.index
            if (not obs_data) and (not var_data): 
                if ind==0: raise ValueError("'feat_x' not found among gene names and data.obs of the data")
                else:
                    if feat is None: break
                    else: raise ValueError("'feat_y' not found among gene names and data.obs of the data")
            
            if obs_data:
                # Load feature values in .obs
                val = data.obs[feat].to_numpy().reshape((-1,1))
                value_list.append(val)
            if var_data:
                # Load feature expression
                if isinstance(data.X, np.ndarray): value_list.append(data[:,feat].X)
                elif isinstance(data.X, sparse.spmatrix): value_list.append(data[:,feat].X.toarray())
                else: raise ValueError("'data.X' should be either numpy ndarray or scipy sparse matrix")                
    else:
        if feat_y is None: value_list = [feat_x]
        else: value_list = [feat_x, feat_y]
    
    # Calculate adjacency matrix and mask if data is provided
    if data is not None:
        # Load location information of spots
        try: loc = data.obs.loc[:,['array_row','array_col']].to_numpy()
        except: raise ValueError("'data' should contain coordinates of spots in .obs as 'array_row' and 'array_col'")
        # Calculate spatial adjacency matrix and mask for gaussian smoothing
        A, mask = extract_adjacency_spatial(loc, fwhm=fwhm)

    for val in value_list:
        # Gaussian smoothing with zero padding
        p = len(val)
        smooth = np.sum(mask*np.matlib.repmat(val,1,p), axis=0)
        smooth = smooth/np.sum(smooth)*np.sum(val)

        ## Estimate dendrogram for feat_x
        t = smooth*(smooth>0)
        # Find nonzero unique value and sort in descending order
        threshold = np.flip(np.sort(np.setdiff1d(t, 0), axis=None))
        # Compute connected components for feat_x
        CC.append(extract_connected_comp(t, sparse.csr_matrix(A), threshold, num_spots=p, min_size=min_size))

    # Extract location of connected components as arrays
    if len(value_list) > 1:
        if data is None:
            CCx_loc_arr = extract_connected_loc_arr(CC[0], num_spots=len(value_list[0]))
            CCy_loc_arr = extract_connected_loc_arr(CC[1], num_spots=len(value_list[1]))
            CCx_loc_arr = filter_connected_loc_exp(CCx_loc_arr, feat=feat_x, thres_per=thres_per)
            CCy_loc_arr = filter_connected_loc_exp(CCy_loc_arr, feat=feat_y, thres_per=thres_per)
            data_fin = np.concatenate((CCx_loc_arr, CCy_loc_arr), axis=1)
        else:
            data_mod = data.copy()
            data_mod, CCx_loc_arr = add_connected_loc(data_mod, CC[0], title='CC', feat_name=feat_x, return_splitted_loc=True, 
                                                       thres_cc = True, thres_per=thres_per)
            data_fin, CCy_loc_arr = add_connected_loc(data_mod, CC[1], title='CC', feat_name=feat_y, return_splitted_loc=True,
                                                       thres_cc = True, thres_per=thres_per)
    else:
        if data is None:
            CCx_loc_arr = extract_connected_loc_arr(CC[0], num_spots=len(value_list[0]))
            data_fin = filter_connected_loc_exp(CCx_loc_arr, feat=feat_x, thres_per=thres_per)
        else:
            data_mod = data.copy()
            data_fin, CCx_loc_arr = add_connected_loc(data_mod, CC[0], title='CC', feat_name=feat_x, return_splitted_loc=True,
                                                       thres_cc = True, thres_per=thres_per)

    if len(value_list) > 1: 
    # Calculate jaccard index
        if J_metric: 
            (Jmax, Jmean, Jmmx, Jmmy) = jaccard_and_connected_loc_(CCx_loc_arr=CCx_loc_arr, CCy_loc_arr=CCy_loc_arr, J_metric=True)
            result = (Jmax, Jmean, Jmmx, Jmmy) 
        else:
            J = jaccard_and_connected_loc_(CCx_loc_arr=CCx_loc_arr, CCy_loc_arr=CCy_loc_arr, J_metric=False)
            result = (J, CC[0], CC[1])
    else: result = CC[0]

    if return_mode=='all': return result, data_fin
    elif return_mode=='cc_loc': return data_fin
    else: return result



def topological_sim_multi_pairs_(data, feat_pairs, group_list=None, group_name='Layer_label',
                                 fwhm=2.5, min_size=5, thres_per=30):
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

    group_name: 
        the column name for the groups saved in metadata(.obs)
        spatial data is divided according to the group and calculate topological overlap separately in each group
    group_list: list of the name of groups

    fwhm: full width half maximum value for the gaussian smoothing kernal
    min_size: minimum size of a connected component
    thres_per: lower percentile value threshold to remove the connected components

    ### Output
    df_top_total: dataframe that contains spatial overlap measures represented by (Jmax, Jmean, Jmmx, Jmmy) for the feature pairs 
    and average value for the feature across the spatial spots (if group is provided, then calculate average for the spots in each group)
    data_mod: AnnData with summed location of all connected components in metadata(.obs) across all feature pairs
    '''
    start_time = time.time()
    
    # Check the format of the feature pairs
    if isinstance(feat_pairs, pd.DataFrame): df_feat = feat_pairs
    elif isinstance(feat_pairs, list): df_feat = pd.DataFrame(feat_pairs)
    else: raise ValueError("'feat_pairs' should be pandas dataframe or list")
    if df_feat.shape[1] != 2:
        raise ValueError("'feat_pairs' should be list format: [('A','B'),('C','D')] or equivalent pandas dataframe")

    # Add group name if no group name is provided
    if group_list is None:
        group_name, group_list = 'group', ['0']
        data.obs[group_name] = '0'
    else:
        if group_name not in data.obs.columns:
            raise ValueError("'group_name' not found in columns of 'data.obs'")
        if not (set(group_list) <= set(data.obs[group_name])):
            raise ValueError("Some elements in 'group_list' not found in 'data.obs["+str(group_name)+"]'")
        data = data[data.obs[group_name].isin(group_list)].copy()
    
    # Determine the type of the data
    if isinstance(data.X, np.ndarray): data_type = 'array'
    elif isinstance(data.X, sparse.spmatrix): data_type = 'sparse'
    else: raise ValueError("'data.X' should be either numpy ndarray or scipy sparse matrix")

    # Change the column names if nan is included
    if np.nan in data.obs.columns:
        data.obs = data.obs.rename(columns=str)
    
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
    if obs_tf_x and obs_tf_y:
        df_feat = df_feat[obs_data_x & obs_data_y].reset_index(drop=True)
    elif (not obs_tf_x) and obs_tf_y:
        df_feat = df_feat[var_data_x & obs_data_y].reset_index(drop=True)
    elif obs_tf_x and (not obs_tf_y):
        df_feat = df_feat[obs_data_x & var_data_y].reset_index(drop=True)
    else:
        df_feat = df_feat[var_data_x & var_data_y].reset_index(drop=True)

    # Repeat the process for the proivded group_list
    for num, i in enumerate(group_list):
        data_sub = data[data.obs[group_name]==i].copy()
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
                df_tmp['Avg_1'] = np.log1p(np.mean(np.expm1(data_sub[:,df_tmp.iloc[:,1].tolist()].X.toarray()), axis=0))
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
                df_tmp['Avg_2'] = np.log1p(np.mean(np.expm1(data_sub[:,df_tmp.iloc[:,2].tolist()].X.toarray()), axis=0))

        # Remove the features which have zero values only
        df_tmp = df_tmp[(df_tmp['Avg_1']!=0) & (df_tmp['Avg_2']!=0)].reset_index(drop=True)

        # Replace the absolute average values to the real average
        if data_x is not None: df_tmp['Avg_1'] = np.mean(data_x, axis=0)[df_tmp.index]
        if data_y is not None: df_tmp['Avg_2'] = np.mean(data_y, axis=0)[df_tmp.index]

        # Extract the non-overlapping feat list from group i and save index number corresponding to feature pairs
        if obs_tf_x != obs_tf_y:
            # In case type of feature x and y is different
            comb_feat_list_x = df_tmp.iloc[:,1].drop_duplicates().reset_index().set_index(0)
            comb_feat_list_y = df_tmp.iloc[:,2].drop_duplicates().reset_index().set_index(1)
            # Find the index for the combined feature
            df_tmp['Index_1'] = comb_feat_list_x.loc[df_tmp.iloc[:,1]].reset_index()['index'].astype(int)
            df_tmp['Index_2'] = comb_feat_list_y.loc[df_tmp.iloc[:,2]].reset_index()['index'].astype(int)

            if obs_tf_x:
                val_x = data_sub.obs[comb_feat_list_x.index].to_numpy()
                if data_type=='array': val_list.append(np.concatenate((val_x, data_sub[:,comb_feat_list_y.index].X), axis=1))
                else: val_list.append(np.concatenate((val_x, data_sub[:,comb_feat_list_y.index].X.toarray()), axis=1))
            else:                
                val_y = data_sub.obs[comb_feat_list_y.index].to_numpy()
                if data_type=='array': val_list.append(np.concatenate((data_sub[:,comb_feat_list_x.index].X, val_y), axis=1))
                else: val_list.append(np.concatenate((data_sub[:,comb_feat_list_x.index].X.toarray(), val_y), axis=1))
        else:
            # In case type of feature x and y is same
            # Define combined feature list for feat_1 and feat_2 and remove duplicates
            comb_feat_list = pd.concat([df_tmp.iloc[:,1], 
                                        df_tmp.iloc[:,2]], axis=0).drop_duplicates().to_frame().set_index(0)
            comb_feat_list['index'] = range(len(comb_feat_list))
            # Find the index of the feature in Feat_1 and Feat_2 among the comb_feat_list.index
            df_x = comb_feat_list.loc[df_tmp.iloc[:,1].drop_duplicates()]
            df_y = comb_feat_list.loc[df_tmp.iloc[:,2].drop_duplicates()]
            # Combine the dataframe df_x and df_y
            df_xy = pd.concat([df_x, df_y], axis=1)
            df_xy.columns = ['index_x','index_y']

            # Find the index for the Feature 1 and Feature 2: comb_feat_list as reference
            df_tmp['Index_1'] = df_xy.loc[df_tmp.iloc[:,1]].reset_index()['index_x'].astype(int)
            df_tmp['Index_2'] = df_xy.loc[df_tmp.iloc[:,2]].reset_index()['index_y'].astype(int)

            if obs_tf_x:
                val_list.append(data_sub.obs[comb_feat_list.index].to_numpy())
            else:
                if data_type=='array': val_list.append(data_sub[:,comb_feat_list.index].X)
                else: val_list.append(data_sub[:,comb_feat_list.index].X.toarray())
        # Append the dataframe
        df_top_total = df_top_total.append(df_tmp)

        # Add the location information of the spots
        try: loc_list.append(data_sub.obs.loc[:,['array_row','array_col']].to_numpy())
        except: raise ValueError("'data' should contain coordinates of spots in .obs as 'array_row' and 'array_col'")
        
    # Make dataframe for the list of feature 1 and 2 across the groups
    column_names = [group_name,'Feat_1','Feat_2','Avg_1','Avg_2','Index_1','Index_2']
    df_top_total.columns = column_names
    df_top_total.index = range(df_top_total.shape[0])

    print('End of data preparation')
    print("Elapsed time: %s seconds " % (time.time()-start_time))

    # Extract connected components for the features
    procs = []
    pool = mp.Pool(processes=os.cpu_count())
    for feat, loc in zip(val_list, loc_list):
        A, mask = extract_adjacency_spatial(loc, fwhm=fwhm)
        proc_grp = []
        for index in range(feat.shape[1]):
            proc = pool.apply_async(func=topological_sim_pair,
                                    args=(None, feat[:,index].reshape((-1,1)), None,
                                    fwhm, min_size, thres_per, False, 'cc_loc', A, mask))                                    
            proc_grp.append(proc)
        procs.append(proc_grp)
    pool.close()
    pool.join()
    # Get the output in case feature x and feature y has same data type
    output_cc = [[proc.get() for proc in proc_grp] for proc_grp in procs]

    print('End of computation for topological similarity')
    print("Elapsed time: %s seconds " % (time.time()-start_time))
    
    # Make dataframe for the similarity between feature 1 and 2 across the groups
    jaccard_total = []; output_cc_loc = []
    pool = mp.Pool(processes=os.cpu_count())
    for num, element in enumerate(group_list):
        df_subset = df_top_total[df_top_total[group_name]==element]
        
        # Find the subset of the given data
        data_sub = data[data.obs[group_name]==element].copy()
        # Add the connected component location
        df_cc_loc = pd.concat([pd.DataFrame(np.sum(arr, axis=1)) for arr in output_cc[num]], axis=1)

        # Make dataframe representing location of CC when the data type is different between feature x and y
        if obs_tf_x != obs_tf_y:
            # Reconstruct combined feature list x and y for each group
            comb_feat_list_x = df_subset['Feat_1'].drop_duplicates().tolist()
            comb_feat_list_y = df_subset['Feat_2'].drop_duplicates().tolist()
            # Assign column names and index
            df_cc_loc.columns = ['Comb_CC_'+str(i) for i in comb_feat_list_x+comb_feat_list_y]
            df_cc_loc.index = data[data.obs[group_name]==group_list[num]].obs.index
        else:
            # Reconstruct combined feature list for each group
            comb_feat_list = pd.concat([df_subset['Feat_1'], df_subset['Feat_2']],
                                            axis=0, ignore_index=True).drop_duplicates().tolist()
            # Assign column names and index
            df_cc_loc.columns = ['Comb_CC_'+str(i) for i in comb_feat_list]
            df_cc_loc.index = data[data.obs[group_name]==group_list[num]].obs.index
        output_cc_loc.append(df_cc_loc)

        for index in range(len(df_subset)):
            if obs_tf_x != obs_tf_y:
                CCx_loc_arr = output_cc[num][:len(comb_feat_list_x)][df_subset['Index_1'].iloc[index]]
                CCy_loc_arr = output_cc[num][len(comb_feat_list_x):][df_subset['Index_2'].iloc[index]]
            else:
                CCx_loc_arr = output_cc[num][df_subset['Index_1'].iloc[index]]
                CCy_loc_arr = output_cc[num][df_subset['Index_2'].iloc[index]]

            jaccard = pool.apply_async(func=jaccard_and_connected_loc_, 
                                       args=(None, None, None, CCx_loc_arr, CCy_loc_arr, 
                                       df_subset['Feat_1'].iloc[index], df_subset['Feat_2'].iloc[index], 
                                       True, 'jaccard', False))
            jaccard_total.append(jaccard)
    pool.close()
    pool.join()

    # Get the output for connected component location and save
    data_mod = data.copy()
    output_cc_loc = pd.concat(output_cc_loc, axis=0).fillna(0)
    # Remove the duplicate columns
    output_cc_loc = output_cc_loc.T.groupby(level=0).first().T
    data_mod.obs = pd.concat([data_mod.obs, output_cc_loc], axis=1)

    # Get the output for jaccard
    output_j = [jaccard.get() for jaccard in jaccard_total]
    
    # Create dataframe for J metrics
    output_j = pd.DataFrame(output_j, columns=['J_max','J_mean','J_mmx','J_mmy'])

    # Create dataframe with pairwise topological similarity measures
    df_top_total = pd.concat([df_top_total.iloc[:,:-2], output_j], axis=1)
    
    print("End of the whole process: %s seconds" % (time.time()-start_time))
    
    return df_top_total, data_mod