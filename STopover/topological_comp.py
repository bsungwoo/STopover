"""
Created on Sat Apr 3 18:39:34 2021
Last modified on Thurs March 31 09:16:00 2022
Numba optimization added for performance improvement

@author: 
Original matlab code for graph filtration: Hyekyoung Lee
Translation to python and addition of visualization code: Sungwoo Bae with the help of matlab2python
"""
import os
import numpy as np
from numpy.matlib import repmat
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from anndata import AnnData
from scipy.ndimage import gaussian_filter

from .make_original_dendrogram_cc import make_original_dendrogram_cc
from .make_smoothed_dendrogram import make_smoothed_dendrogram
from .make_dendrogram_bar import make_dendrogram_bar

# Check if numba is available
try:
    import numba
    from numba import jit, prange, int32, float64, boolean
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    print("Numba not found. Using standard Python implementation.")

def extract_adjacency_spatial(loc, spatial_type='visium', fwhm=2.5):
    '''
    ## Extract adjacency matrix from spatial coordinates
    
    ### Input
    loc: spatial coordinates (n_spots x 2)
    spatial_type: type of spatial data ('visium', 'ST', 'imageST', 'visiumHD')
    fwhm: full width at half maximum for Gaussian kernel (default: 2.5)
    
    ### Output
    A: adjacency matrix (sparse matrix)
    arr_mod: modified array (for visualization)
    '''
    # Convert to numpy array if not already
    if isinstance(loc, pd.DataFrame):
        loc = loc.values
    
    # Calculate sigma from FWHM
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    if spatial_type in ['visium', 'ST']:
        # For Visium and ST, use hexagonal grid
        n = loc.shape[0]
        
        # Calculate pairwise distances
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((loc[i] - loc[j])**2))
                A[i, j] = dist
                A[j, i] = dist
        
        # Set diagonal to infinity
        np.fill_diagonal(A, np.inf)
        
        # Find minimum distance
        min_distance = np.min(A)
        
        # Set threshold for adjacency (1.5 times minimum distance)
        A[A > 1.5 * min_distance] = np.inf
        
        # Apply Gaussian filter
        A_inf = A.copy()
        A_inf[A_inf == np.inf] = 0
        arr_mod = gaussian_filter(A_inf, sigma)
        
        # Create adjacency matrix (0/1)
        A_adj = ((A > 0) & (A <= 1.5 * min_distance)).astype(int)
        
        return sparse.csr_matrix(A_adj), arr_mod
    
    elif spatial_type in ['imageST', 'visiumHD']:
        # For imageST and visiumHD, use grid
        n = loc.shape[0]
        A = np.zeros((n, n), dtype=int)
        
        # Connect adjacent pixels/spots
        for i in range(n):
            for j in range(i+1, n):
                # Check if spots are adjacent (Manhattan distance = 1)
                if abs(loc[i, 0] - loc[j, 0]) + abs(loc[i, 1] - loc[j, 1]) == 1:
                    A[i, j] = 1
                    A[j, i] = 1
        
        # No Gaussian mask for these types
        arr_mod = None
        
        return sparse.csr_matrix(A), arr_mod
    
    return None, None

@jit(nopython=True, parallel=True, cache=True)
def _extract_connected_loc_mat_numba(CC, loc, feat_val):
    """
    Numba-optimized implementation of connected component location matrix extraction
    
    Parameters:
    -----------
    CC : list of arrays
        Connected components
    loc : numpy.ndarray
        Spatial coordinates
    feat_val : numpy.ndarray
        Feature values
    
    Returns:
    --------
    CC_loc_mat : numpy.ndarray
        Location matrix for connected components
    """
    n_cc = len(CC)
    n_spots = loc.shape[0]
    n_dims = loc.shape[1]
    
    # Initialize output
    CC_loc_mat = np.zeros((n_spots, n_dims + 1))
    
    # Fill output
    for i in prange(n_cc):
        cc = CC[i]
        for j in range(len(cc)):
            node = cc[j]
            CC_loc_mat[node, :n_dims] = loc[node]
            CC_loc_mat[node, n_dims] = feat_val[node]
    
    return CC_loc_mat

def extract_connected_loc_mat(CC, loc, feat_val, use_numba=True):
    '''
    ## Extract location matrix for connected components
    
    ### Input
    CC: list of connected components
    loc: spatial coordinates (n_spots x 2)
    feat_val: feature values (n_spots)
    use_numba: whether to use numba optimization (default: True)
    
    ### Output
    CC_loc_mat: location matrix for connected components
    '''
    # Convert to numpy arrays if not already
    if isinstance(loc, pd.DataFrame):
        loc = loc.values
    if isinstance(feat_val, pd.Series):
        feat_val = feat_val.values
    
    # Use Numba optimization if available
    if use_numba and _HAS_NUMBA:
        # Convert CC to format compatible with Numba
        CC_numba = [np.array(cc, dtype=np.int32) for cc in CC]
        
        # Call Numba-optimized function
        CC_loc_mat = _extract_connected_loc_mat_numba(CC_numba, loc, feat_val)
    else:
        # Standard Python implementation
        n_spots = loc.shape[0]
        n_dims = loc.shape[1]
        
        # Initialize output
        CC_loc_mat = np.zeros((n_spots, n_dims + 1))
        
        # Fill output
        for cc in CC:
            for node in cc:
                CC_loc_mat[node, :n_dims] = loc[node]
                CC_loc_mat[node, n_dims] = feat_val[node]
    
    return CC_loc_mat

@jit(nopython=True, cache=True)
def _split_connected_loc_numba(CC_loc_mat):
    """
    Numba-optimized implementation of connected location matrix splitting
    
    Parameters:
    -----------
    CC_loc_mat : numpy.ndarray
        Location matrix for connected components
    
    Returns:
    --------
    loc_mat : numpy.ndarray
        Location matrix
    feat_val : numpy.ndarray
        Feature values
    """
    n_spots = CC_loc_mat.shape[0]
    n_dims = CC_loc_mat.shape[1] - 1
    
    # Initialize outputs
    loc_mat = np.zeros((n_spots, n_dims))
    feat_val = np.zeros(n_spots)
    
    # Fill outputs
    for i in range(n_spots):
        loc_mat[i] = CC_loc_mat[i, :n_dims]
        feat_val[i] = CC_loc_mat[i, n_dims]
    
    return loc_mat, feat_val

def split_connected_loc(CC_loc_mat, use_numba=True):
    '''
    ## Split connected location matrix into location matrix and feature values
    
    ### Input
    CC_loc_mat: location matrix for connected components
    use_numba: whether to use numba optimization (default: True)
    
    ### Output
    loc_mat: location matrix
    feat_val: feature values
    '''
    # Use Numba optimization if available
    if use_numba and _HAS_NUMBA:
        # Call Numba-optimized function
        loc_mat, feat_val = _split_connected_loc_numba(CC_loc_mat)
    else:
        # Standard Python implementation
        n_dims = CC_loc_mat.shape[1] - 1
        loc_mat = CC_loc_mat[:, :n_dims]
        feat_val = CC_loc_mat[:, n_dims]
    
    return loc_mat, feat_val

def topological_comp_res(feat_val, A, threshold=None, min_size=5, use_numba=True):
    '''
    ## Compute topological components from feature values and adjacency matrix
    
    ### Input
    feat_val: feature values (n_spots)
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
    # Convert to numpy array if not already
    if isinstance(feat_val, pd.Series):
        feat_val = feat_val.values
    
    # Make original dendrogram
    CC, E, duration, history = make_original_dendrogram_cc(
        feat_val, A, threshold, min_size, use_numba=use_numba
    )
    
    # Make smoothed dendrogram
    CC, E, duration, history = make_smoothed_dendrogram(
        CC, E, duration, history, min_size, use_numba=use_numba
    )
    
    return CC, E, duration, history

def extract_connected_comp(tx, A_sparse, threshold_x, num_spots, min_size=5):
    '''
    ## Compute commnected components
    ### Input
    tx: gene expression profiles of a feature across p spots/grids (p * 1 array)
    A_sparse: sparse matrix for spatial adjacency matrix across spots/grids (0 and 1)
    threshold_x: threshold value for tx
    num_spots: number of spots/grids in the spatial dataset
    min_size: minimum size of a connected component

    ### Output:
    CCx: list containing index of spots/grids indicating location of connected components for feature x
    '''
    cCC_x,cE_x,cduration_x,chistory_x = make_original_dendrogram_cc(tx,A_sparse,threshold_x)

    ## Estimated smoothed dendrogram for feat_x
    nCC_x,_,nduration_x,nhistory_x = make_smoothed_dendrogram(cCC_x,cE_x,cduration_x,chistory_x,np.array([min_size, num_spots]))
    
    ## Estimate bars for plot for feat_x
    cvertical_x_x,cvertical_y_x,chorizontal_x_x,chorizontal_y_x,cdots_x,clayer_x = make_dendrogram_bar(chistory_x,cduration_x)
    _,_,_,_,_,nlayer_x = make_dendrogram_bar(nhistory_x,nduration_x,cvertical_x_x,cvertical_y_x,chorizontal_x_x,chorizontal_y_x,cdots_x)
    
    ## Remove unnecessary variables
    del cvertical_x_x,cvertical_y_x,chorizontal_x_x,chorizontal_y_x,cdots_x,clayer_x

    ## Extract connected components for feat_x
    sind = nlayer_x[0]
    CCx = [nCC_x[i] for i in sind]
    return CCx

def extract_connected_loc_mat(CC, num_spots, format='sparse'):
    '''
    ## Calculate the integer array which explains the location of each connected component
    ### Input
    CC: list containing index of spots/grids for each connected component
    num_spots: total number of spots/grids in the spatial data used for analysis
    format: format of the connected location data

    ### Output
    Returns connected component location sparse matrix: positive integers are assigned to the corresponding spots/grids composing each connected components
    For example 1 was given to the spots/grids composing the first connected component, 2 to the spots/grids composing the second connected components, and so on.

    Different connected components of a feature are separated along the axis=1 of numpy array
    Therefore, when the number of spot is p and the number of connected component is m then the shape of array is p*m
    If two conneceted components(CCs) are found in a total of 5 spots/grids and CC1 is composed of 4th-5th spots/grids and CC2 of 2nd-3rd spots/grids,
    then the array will be np.array([[0, 0], [0, 2], [0, 2], [1, 0], [1, 0]])
    '''
    if format not in ['sparse', 'array']: raise ValueError("'format' should be either 'sparse' or 'array'")

    if len(CC) > 0:
        for num, element in enumerate(CC):
            # if len(element) == 0: continue
            # Convert CC location index list to array
            CC_loc_index = np.array(element, dtype=int)   
            # Assign the same number (num+1) to the location of each connected component
            CC_zero_arr = np.zeros(num_spots)
            np.put(CC_zero_arr, CC_loc_index, (num+1))
            # Concatenate the location array of connected components
            if num == 0: CC_loc_arr = CC_zero_arr.reshape((-1,1))
            else: CC_loc_arr = np.concatenate((CC_loc_arr, CC_zero_arr.reshape((-1,1))), axis=1)
    else:
        CC_loc_arr = np.zeros((num_spots, 1))
    
    if format == 'sparse': return sparse.csr_matrix(CC_loc_arr.astype(int))
    else: return CC_loc_arr.astype(int)

def filter_connected_loc_exp(CC_loc_mat, data=None, feat=None, thres_per=30, return_sep_loc=False):
    '''
    ## Filter connected component location according to the expression value or metadata value
    ### Input
    data: spatial data (format: anndata) containing log-normalized gene expression
    CC_loc_mat: connected component matrix representing all connected component location separately for one feature
        -> when the number of spot is p and the number of connected component is m then the shape of matrix is p*m
    feat: name of the feature to calculate CC or values
    thres_per: lower percentile value threshold to remove the connected components
    return_sep_loc:
        whether to return dataframe of connected component location separately for feature x and y 
        or return merged dataframe representing summed location of all connected components for feature x and y, respectively

    ### Output
    CC_loc_mat_fin: connected component location sparse matrix filtered according to the expression values in each connected component cluster
        -> the matrix represents all connected component location separately for one feature
        -> when the number of spot is p and the number of connected component is m then the shape of matrix is p*m
    '''
    # Extract expression information
    if isinstance(feat, str):
        if data is None: 
            raise ValueError("Anndata object with log-normalized count matrix should be provided in 'data'")
        if feat in data.obs.columns: feat_data = data.obs[[feat]].to_numpy()
        elif feat in data.var.index: 
            # Determine the type of the data
            if isinstance(data.X, np.ndarray): feat_data = sparse.csr_matrix(data[:,feat].X)
            elif isinstance(data.X, sparse.spmatrix): feat_data = data[:,feat].X
            else: ValueError("'data.X' should be either numpy ndarray or scipy sparse matrix")
        else: raise ValueError("'feat' is not found among gene names and metadata")
    elif isinstance(feat, np.ndarray): feat_data = feat
    else: raise ValueError("'feat' should be either string or numpy ndarray")

    # Calculate the sum of the sparse matrix which summarizes the location of all connected components
    CC_mat_sum = CC_loc_mat.sum(axis=1)

    ## Remove the connected components with lower values (below the threshold percentile)    
    df_CC_loc_exp = pd.DataFrame(np.concatenate((CC_mat_sum, feat_data), axis=1))
    # Calculate the mean value for each connected components (expression or metadata values)
    CC_mean = df_CC_loc_exp.groupby([0]).mean().sort_values(by=[1], ascending=False)
    # Filter the data for the percentile threshold for the values (expression or metadata values)
    CC_mean = CC_mean.iloc[:int(len(CC_mean)*(1-(thres_per/100))),:]
    # Save the location of connected component only for the high values (expression or metadata values)
    CC_loc_mat_fin = CC_loc_mat[:, (np.sort(CC_mean.index[CC_mean.index != 0]).astype(int) - 1)]
    
    # Return sparse csr matrix format connected component location separately in each column
    if return_sep_loc: return CC_loc_mat_fin
    else: return CC_loc_mat_fin.sum(axis=1)

def split_connected_loc(data, feat_name_x='', feat_name_y='', return_loc_arr=True):
    '''
    ## Return anndata with location for each connected component separately to .obs
    ### Input
    data: spatial data (format: anndata) containing log-normalized gene expression
    feat_name_x, feat_name_y: name of the feature x and y
    return_loc_mat: whether to return numpy ndarray representing each connected component location separately (sequence of CCx and CCy)

    ### Output
    data_mod: AnnData with location of each connected components separately in metadata(.obs)
    CCxy_fin: numpy ndarray representing each connected component location separately (sequence of CCx and CCy)
    num_ccx: total number of connected component for feature x
    '''
    data_mod = data.copy()
    # Check the feasibility of the given dataset
    if not (set(['Comb_CC_'+feat_name_x,'Comb_CC_'+feat_name_y]) <= set(data_mod.obs.columns)):
        raise ValueError("No CC location data for the given 'feat_x' and 'feat_y'")

    # Extract metadata (.obs) with 'Comb_CC_'+feat_name_x or 'Comb_CC_'+feat_name_y
    CCxy = data_mod.obs.loc[:,['Comb_CC_'+feat_name_x,'Comb_CC_'+feat_name_y]].to_numpy()

    # Number of total connected components for CCx and CCy
    CCx_unique, CCy_unique = np.unique(CCxy[:,0])[1:], np.unique(CCxy[:,1])[1:]
    num_ccx, num_ccy = len(CCx_unique), len(CCy_unique)

    # Define numpy replicative array with the CCx and CCy index in each column
    CCx = repmat(CCx_unique, len(CCxy), 1)
    CCy = repmat(CCy_unique, len(CCxy), 1)
    CCxy_index = np.concatenate((CCx, CCy), axis=1)
    # Define numpy replicate array for CCxy across the rows
    CCxy_rep = np.concatenate((repmat(CCxy[:,0].reshape((-1,1)), 1, num_ccx), 
                                repmat(CCxy[:,1].reshape((-1,1)), 1, num_ccy)), axis=1)
    
    # Create array for the intersecting elements between CCxy_index and CCxy_rep
    CCxy_fin = CCxy_index*(CCxy_index == CCxy_rep)
    # Return anndata with location for each CC separately
    if not return_loc_arr:
        column_names = ['_'.join(('CC',str(i+1),feat_name_x)) for i in CCx_unique] + \
                        ['_'.join(('CC',str(i+1),feat_name_y)) for i in CCy_unique]
        CCxy_df = pd.DataFrame(CCxy_fin, columns=column_names).astype('category')
        CCxy_df.index = data_mod.obs.index
        data_mod.obs = pd.concat([data_mod.obs, CCxy_df], axis=1)

    if return_loc_arr: return CCxy_fin, num_ccx
    else: return data_mod

def save_connected_loc_data_(data, save_format='h5ad', path = os.getcwd(), filename = 'cc_location'):
    '''
    ## Save the anndata or metadata file to the certain location
    ### Input
    data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
    save_format: format to save the location of connected components; either 'h5ad' or 'csv'
    file_name: file name to save (default: cc_location)

    ### Output: None
    '''
    if len([i for i in data.obs.columns if str(i).startswith('Comb_CC')])<2:
        print("'data' does not contain location of connected components")

    if save_format=="h5ad":
        data_mod = data.copy()
        # Save the anndata after removing all anndata saved in .uns
        for key in data.uns.keys():
            if isinstance(data_mod.uns[key], AnnData):
                print("Saving anndata in .uns separately as .h5ad:", key)
                data_mod.uns[key].write_h5ad(os.path.join(path,'_'.join((filename,key,'uns.h5ad'))), compression='gzip')
                del data_mod.uns[key]
        data_mod.write_h5ad(os.path.join(path,'_'.join((filename,'adata.h5ad'))), compression='gzip')
    elif save_format=="csv":
        df_adata = data.obs[[i for i in data.obs.columns if str(i).startswith('Comb_CC')]]
        df_adata.to_csv(os.path.join(path,'_'.join((filename,'df.csv'))),
                        sep = ',', header=True, index=True)
    else:
        raise ValueError("'save_format' should be either h5ad or csv")