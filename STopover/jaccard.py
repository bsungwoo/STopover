import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy import sparse

from .topological_comp import split_connected_loc
from .topological_comp import extract_connected_loc_mat


def jaccard_composite(CCx_loc_sum, CCy_loc_sum, feat_x=None, feat_y=None):
    '''
    ## Calculate jaccard composite index from the connected component location array
    ### Input
    CCx_loc_sum, CCy_loc_sum: numpy ndarray or scipy sparse matrix (shape: (-1,1)) representing the summed connected component location (x or y) in one column
    feat_x, feat_y: feature values corresponding to the feature x and y (shape: (-1,1))
    
    ### Output
    return composite jaccard simliarity index calculated between all CCx and all CCy (jaccard between all connected components)
    '''
    if isinstance(CCx_loc_sum, np.ndarray) and isinstance(CCy_loc_sum, np.ndarray): 
        CCxy_loc_sum = np.concatenate((CCx_loc_sum, CCy_loc_sum), axis=1)
    elif isinstance(CCx_loc_sum, sparse.spmatrix) and isinstance(CCy_loc_sum, sparse.spmatrix): 
        CCxy_loc_sum = np.concatenate((CCx_loc_sum.toarray(), CCy_loc_sum.toarray()), axis=1)
    else: raise ValueError("'CCx_loc' and 'CCy_loc' should be both numpy ndarray or scipy sparse matrix")

    if np.count_nonzero(CCxy_loc_sum) == 0: J_comp = 0
    else:
        if (feat_x is None) and (feat_y is None):
            J_comp = 1 - pdist((CCxy_loc_sum != 0).T, 'jaccard')[0]
        else:
            if isinstance(feat_x, np.ndarray) and isinstance(feat_y, np.ndarray): 
                feat_val = np.concatenate((feat_x, feat_y), axis=1)
            elif isinstance(feat_x, sparse.spmatrix) and isinstance(feat_y, sparse.spmatrix): 
                feat_val = np.concatenate((feat_x.toarray(), feat_y.toarray()), axis=1)
            else: raise ValueError("'feat_x' and 'feat_y' should be both numpy ndarray or scipy sparse matrix")
            feat_val = np.array(CCxy_loc_sum != 0)*(feat_val-feat_val.min(axis=0))/(feat_val.max(axis=0)-feat_val.min(axis=0))
            J_comp = np.sum(feat_val.min(axis=1))/np.sum(feat_val.max(axis=1))
    
    return J_comp



def jaccard_and_connected_loc_(data, CCx=None, CCy=None, feat_name_x="", feat_name_y="", J_comp=False,
                               jaccard_type='default', return_mode='jaccard', return_sep_loc=False):
    '''
    ## Calculate jaccard index and extract location for connected components
    ### Input
    data: spatial data (format: anndata) containing log-normalized gene expression
    CCx, CCy: list containing index of spots for each connected component x and y
    feat_name_x, feat_name_y: name of the feature x and y
    J_comp: whether to calculate Jaccard composite index Jcomp between CCx and CCy pair 
    jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

    return_mode: mode of return
        'all': return jaccard index result along with dataframe for location of connected components and number of connected components of feature x
        'jaccard': return jaccard index result only
        'cc_loc_df': return pandas dataframe for location of connected components only
    return_sep_loc:
        whether to return dataframe of connected component location separately for feature x and y 
        or return merged dataframe representing summed location of all connected components for feature x and y, respectively

    ### Output
    CCxy_loc_arr: numpy ndarray including the location of all connected components in feature x and y 
    J_result:
        if J_comp is True, then composite jaccard simliarity index calculated between all CCx and all CCy (jaccard between all connected components)
        if J_comp is False, then return pairwise jaccard similarity array between CCx and CCy (dim 0: CCx, dim 1: CCy)
    num_ccx: number of connected components for the feature x
    '''
    # Check the feasibility of the dataset given
    if return_mode not in ['all','jaccard','cc_loc_arr']: raise ValueError("'return_mode' should be among 'all', 'cc_loc_arr', or 'jaccard'")
    if jaccard_type not in ['default', 'weighted']: raise ValueError("'jaccard_type' should be either 'default' or 'weighted'") 
    if not (isinstance(feat_name_x, str) and isinstance(feat_name_y, str)): raise ValueError("'feat_x' and 'feat_y' should be both string")
    # Extract the feature values when jaccard type is "weighted"
    if jaccard_type=="weighted":
        feat_val = []
        for feat_name in [feat_name_x, feat_name_y]:
            if feat_name in data.obs.columns: feat_val.append(data.obs[[feat_name]].to_numpy())
            elif feat_name in data.var.index: 
                # Determine the type of the data
                if isinstance(data.X, np.ndarray): feat_val.append(data[:,feat_name].X)
                elif isinstance(data.X, sparse.spmatrix): feat_val.append(data[:,feat_name].X.toarray())
                else: ValueError("'data.X' should be either numpy ndarray or scipy sparse matrix")
            else: raise ValueError("'"+feat_name+"' is not found among gene names and metadata")
        feat_val = np.concatenate(feat_val, axis=1)
        feat_val = (feat_val-feat_val.min(axis=0))/(feat_val.max(axis=0)-feat_val.min(axis=0))

    # Calculate CCxy_loc_arr: combined connected component location array for feature x and y
    data_mod = data.copy()
    if (CCx is None) or (CCy is None):
        if 'CC_1_'+feat_name_x not in data_mod.obs.columns or 'CC_1_'+feat_name_y not in data_mod.obs.columns:
            CCxy_loc_arr, num_ccx = split_connected_loc(data_mod, feat_name_x=feat_name_x, feat_name_y=feat_name_y, return_loc_arr=True)
        else:
            column_names_x = [i for i in data_mod.obs.columns if str(i).startswith('CC_') and str(i).endswith(feat_name_x)]
            column_names_y = [i for i in data_mod.obs.columns if str(i).startswith('CC_') and str(i).endswith(feat_name_y)]
            CCxy_loc_arr = data_mod.obs.loc[:,(column_names_x+column_names_y)].to_numpy()
            num_ccx = len(column_names_x)   
    else:
        # Extract the connected component location for feature x and y
        CCx_loc_arr = extract_connected_loc_mat(CCx, data_mod.shape[0], format='array')
        CCy_loc_arr = extract_connected_loc_mat(CCy, data_mod.shape[0], format='array')
        num_ccx = CCx_loc_arr.shape[1]
        # Concat connected component location sparse matrix for feature x and y
        CCxy_loc_arr = np.concatenate((CCx_loc_arr, CCy_loc_arr), axis=1)

    if np.count_nonzero(CCxy_loc_arr) == 0:
        if J_comp: J_result = 0
        else: J_result = np.array([])
    else:
        if J_comp:
            CCxy_loc_sum = np.concatenate((CCxy_loc_arr[:,:num_ccx].sum(axis=1).reshape((-1,1)), 
                                           CCxy_loc_arr[:,num_ccx:].sum(axis=1).reshape((-1,1))), axis=1)
            if jaccard_type == "default":  
                J_result = 1 - pdist((CCxy_loc_sum != 0).T, 'jaccard')[0]
            else:
                feat_val = (CCxy_loc_sum != 0) * feat_val
                J_result = np.sum(feat_val.min(axis=1))/np.sum(feat_val.max(axis=1))
        else:
            if jaccard_type == "default":  
                # Calculate jaccard matrix for feature x and y
                J_dist = pdist((CCxy_loc_arr != 0).T, 'jaccard')
                J_result = (1-squareform(J_dist))[:num_ccx,num_ccx:]
            else:
                J_result = np.zeros((num_ccx, CCxy_loc_arr.shape[1]-num_ccx))
                for idx in range(J_result.shape[0]):
                    for idy in range(J_result.shape[1]):
                        feat_val_ = (CCxy_loc_arr[:,[idx,num_ccx+idy]]!=0) * feat_val
                        J_result[idx][idy] = np.sum(feat_val_.min(axis=1))/np.sum(feat_val_.max(axis=1))
    
    if (not return_sep_loc) and (return_mode != 'jaccard'):
        CCxy_loc_arr = np.concatenate((CCxy_loc_arr[:,:num_ccx].sum(axis=1).reshape((-1,1)), 
                                        CCxy_loc_arr[:,num_ccx:].sum(axis=1).reshape((-1,1))), axis=1)
    
    if return_mode=='all': return J_result, CCxy_loc_arr, num_ccx
    elif return_mode=='jaccard': return J_result
    else: return CCxy_loc_arr



def jaccard_top_n_connected_loc_(data, CCx=None, CCy=None, feat_name_x='', feat_name_y='', top_n = 2, jaccard_type='default'):
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
    jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

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
                                                              J_comp=False, jaccard_type=jaccard_type,
                                                              return_mode='all', return_sep_loc=True)
    else:
        J, CCxy_loc_arr, num_ccx = jaccard_and_connected_loc_(data=data_mod, feat_name_x=feat_name_x,feat_name_y=feat_name_y, 
                                                              J_comp=False, jaccard_type=jaccard_type,
                                                              return_mode='all', return_sep_loc=True)

    column_names = ['_'.join(('CC',str(i+1),str(feat_name_x))) for i in range(num_ccx)] + \
                                ['_'.join(('CC',str(i+1),str(feat_name_y))) for i in range(CCxy_loc_arr.shape[1]-num_ccx)]
    CCxy_df = pd.DataFrame(CCxy_loc_arr, columns=column_names).astype(int)
    CCxy_df.index = data_mod.obs.index
    # data_mod.obs = pd.concat([data_mod.obs, CCxy_df], axis=1)
    data_mod.obs = data_mod.obs.join(CCxy_df, lsuffix='_prev')
    
    # Flatten int jaccard array and find the top n indexes and corresponding jaccard indices
    top_n_flat_index = np.argsort(-J, axis=None)[:top_n]
    J_top_n = J.flatten()[top_n_flat_index]
    J_top_n_arg = [np.unravel_index(i, J.shape) for i in top_n_flat_index]

    # Raise error if top_n value is larger than total number of J
    if top_n > np.count_nonzero(J):
        raise ValueError("'top_n' is larger than the non-zero J number")

    for num, (i, j) in enumerate(J_top_n_arg):
        locx = data_mod.obs['_'.join(('CC',str(i+1),str(feat_name_x)))]
        locy = data_mod.obs['_'.join(('CC',str(j+1),str(feat_name_y)))]

        # Find intersecting location for the top n location and assign the number
        data_mod.obs['_'.join(('CCxy_top',str(num+1),feat_name_x,feat_name_y))] = \
            ((1 * ((locx != 0) & (locy == 0))) + (2 * ((locx == 0) & (locy != 0))) + (3 * ((locx != 0) & (locy != 0)))).astype('category')

    return data_mod, J_top_n

