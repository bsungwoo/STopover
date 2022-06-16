import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform

from .topological_comp import split_connected_loc
from .topological_comp import extract_connected_loc_arr


def jaccard_and_connected_loc_(data=None, CCx=None, CCy=None, CCx_loc_arr=None, CCy_loc_arr=None,
                                feat_name_x="", feat_name_y="", J_metric=False, 
                                return_mode='jaccard', return_sep_loc=False):
    '''
    ## Calculate jaccard index and extract location for connected components
    ### Input
    data: spatial data (format: anndata) containing log-normalized gene expression
    CCx, CCy: list containing index of spots for each connected component x and y
    CCx_loc_arr, CCy_loc_arr: array representing the connected component location separately in each column
    feat_name_x, feat_name_y: name of the feature x and y
    J_metric: whether to calculate Jaccard index (Jmax, Jmean, Jmmx, Jmmy) between CCx and CCy pair 

    return_mode: mode of return
        'all': return jaccard index result along with dataframe for location of connected components and number of connected components of feature x
        'jaccard': return jaccard index result only
        'cc_loc_df': return numpy array for location of connected components only
    return_sep_loc:
        whether to return dataframe of connected component location separately for feature x and y 
        or return merged dataframe representing summed location of all connected components for feature x and y, respectively

    ### Output
    CCxy_df: pandas dataframe including the summed location of all connected components in feature x and y 
    J_result:
        if J_metric is True, then jaccard simliarity metrics calculated from jaccard similarity array between CCx and CCy (dim 0: CCx, dim 1: CCy)
        -> (Jmax, Jmean, Jmmx, Jmmy): maximum jaccard index, mean jaccard index and mean of maximum jaccard for CCx and CCy
        if J_metric is False, then return jaccard similarity array between CCx and CCy (dim 0: CCx, dim 1: CCy)
    '''
    # Check the feasibility of the dataset given
    if return_mode not in ['all','jaccard','cc_loc_arr']:
        raise ValueError("'return_mode' should be among 'all', 'cc_loc_arr', or 'jaccard'")
    
    if not (isinstance(feat_name_x, str) and isinstance(feat_name_y, str)):
        raise ValueError("'feat_x' and 'feat_y' should be both string")

    # Calculate CCxy_loc_arr: combined connected component location array for feature x and y
    if (CCx is None) or (CCy is None):
        if (data is None) and ((CCx_loc_arr is None) or (CCy_loc_arr is None)): 
            raise ValueError("'CCx_loc_arr' and 'CCy_loc_arr' should be given when 'data' is None and 'CCx' or 'CCy' is None")
        if data is None:
            CCxy_loc_arr = np.concatenate((CCx_loc_arr, CCy_loc_arr), axis=1)
            num_ccx = CCx_loc_arr.shape[1]
        else:
            data_mod = data.copy()
            if len([i for i in data_mod.obs.columns if str(i).startswith('CC_1')])<2:
                data_mod, CCxy_loc_arr, num_ccx = split_connected_loc(data_mod, feat_name_x=feat_name_x, feat_name_y=feat_name_y, return_loc_array=True)
            else:
                column_names_x = [i for i in data_mod.obs.columns if ('CC_' in str(i)) and (feat_name_x in str(i))]
                column_names_y = [i for i in data_mod.obs.columns if ('CC_' in str(i)) and (feat_name_y in str(i))]
                CCxy_loc_arr = data_mod.obs[:,(column_names_x+column_names_y)].to_numpy()
                num_ccx = len(column_names_x)
    else:
        if data is None: raise ValueError("'data' should be provided when 'CCx' and 'CCy' is given")
        data_mod = data.copy()
        # Extract the connected component location for feature x and y
        CCx_loc_arr = extract_connected_loc_arr(CCx, data_mod.shape[0])
        CCy_loc_arr = extract_connected_loc_arr(CCy, data_mod.shape[0])
        num_ccx = CCx_loc_arr.shape[1]

        # Concat connected component location array for feature x and y
        CCxy_loc_arr = np.concatenate((CCx_loc_arr, CCy_loc_arr), axis=1)

    # Make binary location array for feature x and y
    CCxy_loc_bool = (CCxy_loc_arr != 0)
    # Calculate jaccard array for feature x and y
    J_dist = pdist(CCxy_loc_bool.T, 'jaccard')
    J_result = (1-squareform(J_dist))[:num_ccx,num_ccx:]

    if J_metric:
        if np.count_nonzero(J_result.shape)==2: 
            # Calculate maximum, mean, and mean of maximum jaccard indices
            J_result = (np.max(J_result), np.mean(J_result), 
                        np.mean(np.max(J_result, axis = 1)), np.mean(np.max(J_result, axis = 0)))
        else: 
            J_result = (0, 0, 0, 0)
    
    if not return_sep_loc: 
        CCxy_loc_arr = np.concatenate((np.sum(CCxy_loc_arr[:,:num_ccx], axis=1).reshape((-1,1)),
                                        np.sum(CCxy_loc_arr[:,num_ccx:], axis=1).reshape((-1,1))), axis=1)

    if return_mode=='all': return J_result, CCxy_loc_arr, num_ccx
    elif return_mode=='jaccard': return J_result
    else: return CCxy_loc_arr



def jaccard_top_n_connected_loc_(data, CCx=None, CCy=None, feat_name_x='', feat_name_y='', top_n = 5):
    '''
    ## Calculate top n connected component locations for given feature pairs x and y
    ### Input
    data: 
        AnnData with summed location of all connected components in metadata(.obs) across feature pairs
        -> if the location is not provided in metadata, then calculate connected component location array with CCx and CCy
    CCx, CCy:
        list containing index of spots for each connected component x and y (the direct output of topological_sim_pair when J_metric is False)
    feat_name_x, feat_name_y: name of the feature x and y
    top_n: the number of the top connected components to be found

    ### Output
    data_mod: 
        AnnData with intersecting location of top n connected components between feature x and y saved in metadata(.obs)
        -> top 1, 2, 3, ... intersecting connected component locations are separately saved
    '''
    data_mod = data.copy()
    # Add location of connected components when not provided in metadata of data (.obs)
    # Check the feasibility of the given dataset
    if not (set(['Comb_CC_'+str(feat_name_x),'Comb_CC_'+str(feat_name_y)]) <= set(data.obs.columns)):
        if (CCx is None) or (CCy is None): 
            raise ValueError("No CC locations in 'data': 'CCx' and 'CCy' should be provided")

        J, CCxy_loc_arr, num_ccx = jaccard_and_connected_loc_(data=data_mod, CCx=CCx, CCy=CCy, 
                                                              feat_name_x=feat_name_x,feat_name_y=feat_name_y, 
                                                              J_metric=False, 
                                                              return_mode='all', return_sep_loc=True)
    else:
        J, CCxy_loc_arr, num_ccx = jaccard_and_connected_loc_(data=data_mod, feat_name_x=feat_name_x,feat_name_y=feat_name_y, 
                                                              J_metric=False, 
                                                              return_mode='all', return_sep_loc=True)

    column_names = ['_'.join(('CC',str(i+1),str(feat_name_x))) for i in range(num_ccx)] + \
                                ['_'.join(('CC',str(i+1),str(feat_name_y))) for i in range(CCxy_loc_arr.shape[1]-num_ccx)]
    CCxy_df = pd.DataFrame(CCxy_loc_arr, columns=column_names).astype(int)
    CCxy_df.index = data_mod.obs.index
    data_mod.obs = pd.concat([data_mod.obs, CCxy_df], axis=1)
    
    # Flatten int jaccard array and find the top n indexes
    top_n_flat_index = np.argsort(-J, axis=None)[:top_n]
    J_top_n_arg = [np.unravel_index(i, J.shape) for i in top_n_flat_index]

    # Raise error if top_n value is larger than total number of J
    if top_n > np.count_nonzero(J):
        raise ValueError("'top_n' is larger than the non-zero J number")
    
    for num, (i, j) in enumerate(J_top_n_arg):
        locx = data_mod.obs['_'.join(('CC',str(i+1),str(feat_name_x)))]
        locy = data_mod.obs['_'.join(('CC',str(j+1),str(feat_name_y)))]

        # Find intersecting location for the top n location and assign the number
        data_mod.obs['_'.join(('CCxy_top',str(num+1),feat_name_x,feat_name_y))] = \
            (num + 1) * ((locx != 0) & (locy != 0))

    return data_mod

