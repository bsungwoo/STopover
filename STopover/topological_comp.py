"""
Created on Sat Apr 3 18:39:34 2021
Last modified on Thurs March 31 09:16:00 2022

@author: 
Original matlab code for graph filtration: Hyekyoung Lee
Translation to python and addition of visualization code: Sungwoo Bae with the help of matlab2python

"""
import numpy as np
import numpy.matlib
import pandas as pd
from scipy import sparse
from math import pi

import os
            
from .make_original_dendrogram_cc import make_original_dendrogram_cc
from .make_smoothed_dendrogram import make_smoothed_dendrogram
from .make_dendrogram_bar import make_dendrogram_bar


def extract_adjacency_spatial(loc, fwhm=2.5):
    '''
    ## Compute adjacency matrix and gaussian mask based on spatial locations of spots
    ### Input
    loc: p*2 array for x, y coordinates of p spots
    fwhm: full width half maximum for Gaussian smoothing

    ### Output
    A: Spatial adjacency matrix for spots based on the given x,y coordiantes
    mask: Gaussian smoothing mask for the features based on x,y coordinates of spots
    '''
    p = loc.shape[0]
    sigma = fwhm / 2.355
    # adjacency matrix between spots
    A = np.zeros((p,p))
    for i in range(p):
        for j in range(i, p):
            A[i,j] = np.sqrt(sum((loc[i,:] - loc[j,:])**2))
            A[j,i] = A[i,j]
        
    A[np.where(A > fwhm)] = np.inf

    # Smoothing x and y
    # Gaussian smoothing with zero padding
    # adjacency matrix was actually distance matrix
    mask = 1/(2*pi*sigma**2)*np.exp(-(A**2)/(2*sigma**2))
        
    # really estimate adjacency matrix (convert to 1, 0)
    min_distance = np.min(A[np.nonzero(A)])
    A = ((A > 0) & (A <= min_distance)).astype(int)
    return A, mask



def extract_connected_comp(tx, A_sparse, threshold_x, num_spots, min_size=5):
    '''
    ## Compute commnected components
    ### Input
    tx: gene expression profiles of a feature across p spots (p * 1 array)
    A_sparse: sparse matrix for spatial adjacency matrix across spots (0 and 1)
    threshold_x: threshold value for tx
    num_spots: number of spots in the spatial dataset
    min_size: minimum size of a connected component

    ### Output:
    CCx: list containing index of spots indicating location of connected components for feature x
    '''
    cCC_x,cE_x,cduration_x,chistory_x = make_original_dendrogram_cc(tx,A_sparse,threshold_x)

    ## Estimated smoothed dendrogram for feat_x
    nCC_x,nE_x,nduration_x,nhistory_x = make_smoothed_dendrogram(cCC_x,cE_x,cduration_x,chistory_x,np.array([min_size, num_spots]))
    
    ## Estimate bars for plot for feat_x
    cvertical_x_x,cvertical_y_x,chorizontal_x_x,chorizontal_y_x,cdots_x,clayer_x = make_dendrogram_bar(chistory_x,cduration_x)
    nvertical_x_x,nvertical_y_x,nhorizontal_x_x,nhorizontal_y_x,ndots_x,nlayer_x = make_dendrogram_bar(nhistory_x,nduration_x,cvertical_x_x,cvertical_y_x,chorizontal_x_x,chorizontal_y_x,cdots_x)
    
    ## Extract connected components for feat_x
    sind = nlayer_x[0]
    CCx = [nCC_x[i] for i in sind]
    return CCx



def extract_connected_loc_arr(CC, num_spots):
    '''
    ## Calculate the integer array which explains the location of each connected component
    ### Input
    CC: list containing index of spots for each connected component
    num_spots: total number of spots in the spatial data used for analysis

    ### Output
    CC_loc_arr: positive integers were assigned to the corresponding spots composing each connected components
    For example 1 was given to the spots composing the first connected component, 2 to the spots composing the second connected components, and so on.

    Different connected components of a feature are separated along the axis=1 of numpy array
    Therefore, when the number of spot is p and the number of connected component is m then the shape of array is p*m
    If two conneceted components(CCs) are found in a total of 5 spots and CC1 is composed of 4th-5th spots and CC2 of 2nd-3rd spots,
    then the array will be np.array([[0, 0], [0, 2], [0, 2], [1, 0], [1, 0]])
    '''
    if len(CC) > 0:
        for num, element in enumerate(CC):
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
    
    return CC_loc_arr.astype(int)



def filter_connected_loc_exp(CC_loc_arr, data=None, feat=None, thres_per=30):
    '''
    ## Filter connected component location according to the expression value or metadata value
    ### Input
    data: spatial data (format: anndata) containing log-normalized gene expression
    CC_loc_arr: whether to return connected component array representing all connected component location separately for one feature
        -> when the number of spot is p and the number of connected component is m then the shape of array is p*m
    feat_value: the 
    feat_name: name of the feature to calculate CC or values
    thres_per: lower percentile value threshold to remove the connected components

    ### Output
    CC_loc_arr_fin: connected component location array filtered according to the expression values in each connected component cluster
        -> the array represents all connected component location separately for one feature
        -> when the number of spot is p and the number of connected component is m then the shape of array is p*m
    '''
    # Extract expression information
    if isinstance(feat, str):
        if data is None: 
            raise ValueError("Anndata object with log-normalized count matrix should be provided in 'data'")
        if feat in data.obs.columns: feat_data = data.obs[[feat]].to_numpy()
        elif feat in data.var.index: 
            # Determine the type of the data
            if isinstance(data.X, np.ndarray): feat_data = data[:,feat].X
            elif isinstance(data.X, sparse.spmatrix): feat_data = data[:,feat].X.toarray()
            else: ValueError("'data.X' should be either numpy ndarray or scipy sparse matrix")
        else: raise ValueError("'feat_exp' is not found among gene names and metadata")
    elif isinstance(feat, np.ndarray): feat_data = feat
    else: raise ValueError("'feat' should be either string or numpy ndarray")

    # Calculate the sum of the array which summarizes the location of all connected components
    CC_array_sum = np.sum(CC_loc_arr, axis=1).reshape((-1, 1))

    ## Remove the connected components with lower values (below the threshold percentile)    
    df_CC_loc_exp = pd.DataFrame(np.concatenate((CC_array_sum, feat_data), axis=1))
    # Calculate the mean value for each connected components (expression or metadata values)
    CC_mean = df_CC_loc_exp.groupby([0]).mean().sort_values(by=[1], ascending=False)
    # Filter the data for the percentile threshold for the values (expression or metadata values)
    CC_mean = CC_mean.iloc[:int(len(CC_mean)*(1-(thres_per/100))),:]
    # Save the location of connected component only for the high values (expression or metadata values)
    CC_loc_arr_fin = CC_loc_arr[:, (np.array(CC_mean.index[CC_mean.index != 0], dtype=int) - 1)]

    return CC_loc_arr_fin



def add_connected_loc(data, CC, title='CC', feat_name='', return_splitted_loc=False, thres_cc = False, thres_per=30):
    '''
    ## Return anndata with summed location of each connected components to .obs
    ### Input
    data: spatial data (format: anndata) containing log-normalized gene expression
    CC: list containing index of spots for each connected component
    title: column name of the CC data to be added in .obs
    feat_name: name of the feature to calculate CC
    return_splitted_loc: 
        whether to return the splitted array for the location of connected components separately
        Different connected components of a feature are separated along the axis=1 of numpy array
        If two conneceted components(CCs) are found in a total of 5 spots and CC1 is composed of 4th-5th spots and CC2 of 2nd-3rd spots,
        then the array will be np.array([0, 0], [0, 2], [0, 2], [1, 0], [1, 0])
    thres_cc: whether to threshold the connected component location by value
    thres_per: lower percentile value threshold to remove the connected components

    ### Output
    data_mod: AnnData with summed location of each connected components saved in metadata(.obs)
    CC_loc_arr_fin: 
        array representing all connected component location separately for one feature
        -> when the number of spot is p and the number of connected component is m then the shape of array is p*m
    '''
    # Calculate number of spots in the dataset
    num_spots = data.shape[0]

    # Calculate the integer array which explains the location of each connected component
    # Positive integers were assigned to the corresponding spots componsing connected components
    CC_loc_arr = extract_connected_loc_arr(CC, num_spots)

    # Extract the location of connected component only for the high values (expression or metadata values)
    if thres_cc: CC_loc_arr_fin = filter_connected_loc_exp(data, CC_loc_arr, feat_name, thres_per)
    else: CC_loc_arr_fin  = CC_loc_arr

    # Calculate the sum of the array which summarizes the location of all connected components
    CC_array_sum_fin = np.sum(CC_loc_arr_fin, axis=1)
    
    # Save the location information of all connected components in data_mod.obs
    data_mod = data.copy()
    data_mod.obs['_'.join(('Comb',title,feat_name))] = CC_array_sum_fin
    
    if return_splitted_loc: return data_mod, CC_loc_arr_fin
    else: return data_mod



def split_connected_loc(data, feat_name_x='', feat_name_y='', return_loc_array=True):
    '''
    ## Return anndata with location for each connected component separately to .obs
    ### Input
    data: spatial data (format: anndata) containing log-normalized gene expression
    feat_name_x, feat_name_y: name of the feature x and y
    return_loc_array: whether to return array representing each connected component location separately (sequence of CCx and CCy)

    ### Output
    (CCxy_index == CCxy_rep): boolearn array representing each connected component location separately (sequence of CCx and CCy)
    num_ccx: total number of connected component for feature x
    data: AnnData with location of each connected components separately in metadata(.obs)
    '''
    data_mod = data.copy()
    # Check the feasibility of the given dataset
    if not (set(['Comb_CC_'+feat_name_x,'Comb_CC_'+feat_name_y]) <= set(data_mod.obs.columns)):
        raise ValueError("No CC location data for the given 'feat_x' and 'feat_y'")

    # Extract metadata (.obs) with 'Comb_CC_'+feat_name_x or 'Comb_CC_'+feat_name_y
    CCxy = data_mod.obs.loc[:,['Comb_CC_'+feat_name_x,'Comb_CC_'+feat_name_y]].to_numpy()

    # Number of total connected components for CCx and CCy
    num_ccx = int(np.max(CCxy, axis=0)[0])
    num_ccy = int(np.max(CCxy, axis=0)[1])

    # Define numpy replicative array with the CCx and CCy index in each column
    CCx = np.matlib.repmat(np.arange(1,(num_ccx+1)), len(CCxy), 1)
    CCy = np.matlib.repmat(np.arange(1,(num_ccy+1)), len(CCxy), 1)
    CCxy_index = np.concatenate((CCx, CCy), axis=1)
    # Define numpy replicate array for CCxy across the rows
    CCxy_rep = np.concatenate((np.matlib.repmat(CCxy[:,0].reshape((-1,1)), 1, num_ccx), 
                                np.matlib.repmat(CCxy[:,1].reshape((-1,1)), 1, num_ccy)), axis=1)
    
    # Return anndata with location for each CC separately
    # Create array for the intersecting elements between CCxy_index and CCxy_rep
    CCxy_fin = CCxy_index*(CCxy_index == CCxy_rep)
    column_names = ['_'.join(('CC',str(i+1),feat_name_x)) for i in range(num_ccx)] + \
                    ['_'.join(('CC',str(i+1),feat_name_y)) for i in range(num_ccy)]
    CCxy_df = pd.DataFrame(CCxy_fin, columns=column_names).astype(int)
    CCxy_df.index = data_mod.obs.index
    data_mod.obs = pd.concat([data_mod.obs, CCxy_df], axis=1)

    if return_loc_array: return data_mod, CCxy_fin, num_ccx
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
        raise ValueError("'data' does not contain location of connected components")
    
    if save_format=="h5ad":
        data.obs[[i for i in data.obs.columns if str(i).startswith('Comb_CC')]] = \
        data.obs[[i for i in data.obs.columns if str(i).startswith('Comb_CC')]].astype('category')
        data.write_h5ad(os.path.join(path,'_'.join((filename,'adata.h5ad'))), compression='gzip')
    elif save_format=="csv":
        df_adata = data.obs[[i for i in data.obs.columns if str(i).startswith('Comb_CC')]].astype('category')
        df_adata.to_csv(os.path.join(path,'_'.join((filename,'df.csv'))),
                        sep = ',', header=True, index=True)
    else:
        raise ValueError("'save_format' should be either h5ad or csv")