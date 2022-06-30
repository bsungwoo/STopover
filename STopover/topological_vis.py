from multiprocessing.sharedctypes import Value
import os
import scanpy as sc
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from .jaccard import jaccard_top_n_connected_loc_

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



def vis_jaccard_top_n_pair_(data, top_n = 5, cmap='tab20', spot_size=1,
                            alpha_img=0.8, alpha = 0.8, feat_name_x='', feat_name_y='',
                            fig_size = (10,10), batch_colname='batch', batch_name='0', batch_library_dict=None,
                            image_res = 'hires', adjust_image = True, border = 50, 
                            fontsize = 30, title = 'J', return_axis=False,
                            save = False, path = os.getcwd(), save_name_add = '', dpi=300):
    '''
    ## Visualizing top n connected component x and y showing maximum Jaccard index
    ### Input
    data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
    top_n: the number of the top connected component pairs withthe  highest Jaccard similarity index
    cmap: colormap for the visualization of CC identity
    spot_size: size of the spot visualized on the tissue
    alpha_img: transparency of the tissue, alpha: transparency of the colored spot
    feat_name_x, feat_name_y: name of the feature x and y

    fig_size: size of the drawn figure
    batch_colname: column name to categorize the batch in .obs
    batch_name: the name of the batch slide to visualize (should be one of the elements of batch in .obs)
    batch_library_dict: dictionary that matches batch name with library keys in adata.uns["spatial"]
        -> can be utilized When the multiple Visium slides are merged.
        -> if not provided, then categories for batch_colname in .obs will be matched with library keys in adata.uns["spatial"]

    image_res: resolution of the tissue image to be used in visualization ('hires' or 'lowres')
    adjust_image: whether to adjust the image to show the whole tissue image, if False then crop and show the location of connected component spots only
    border: border of the spots around the spots; this information is used to adjust the image
    fontsize: size of the figure title, title: title of the figure
    return_axis: whether to return the plot axis

    save: whether to save of figure, path: saving path
    save_name_add: additional name to be added in the end of the filename
    dpi: dpi for image

    ### Outut
    axs: matplotlib axis for the plot
    '''
    xsize = ((top_n-1)//4)+1
    if xsize > 1: ysize = 4
    else: ysize = ((top_n-1)%4)+1
    
    # Set figure parameters
    sc.set_figure_params(figsize=(fig_size[0]*ysize, fig_size[1]*xsize), facecolor='white', frameon=False)
    fig, axs = plt.subplots(xsize, ysize, sharey=True, tight_layout=True, squeeze=False)

    # Check the feasibility of the dataset
    if batch_name not in data.obs[batch_colname].values:
        raise ValueError("'batch_name' not among the elements of 'batch_colname' in .obs")
    batch_keys = pd.unique(data.obs[batch_colname]).tolist()
    # Generate dictionary between batch keys and library keys if not given
    if batch_library_dict is None:
        library_keys = list(data.uns['spatial'].keys())   
        if len(library_keys) == len(batch_keys): batch_library_dict = dict(zip(batch_keys, library_keys))
        else: raise ValueError("Number of library keys and batches are different")
    # Subset the spatial data to contain only the batch_name slide
    if len(batch_keys) > 1:
        # Subset the dataset to contain only the batch_name slide
        data_mod = data[data.obs[batch_colname]==batch_name].copy()

    # Calculate top n connected component location
    data_mod = jaccard_top_n_connected_loc_(data_mod, CCx=None, CCy=None, 
                                            feat_name_x=feat_name_x, feat_name_y=feat_name_y, top_n = top_n)

    # Adjust the image to contain the whole slide image
    if adjust_image:
        # Crop the image with certain borders
        height = data.uns['spatial'][batch_library_dict[batch_name]]['images'][image_res].shape[1]
        spot_coord_arr = data.obsm['spatial'] * data.uns['spatial'][batch_library_dict[batch_name]]['scalefactors']['tissue_'+image_res+'_scalef']
        crop_max = np.max(spot_coord_arr, axis=0)
        crop_min = np.min(spot_coord_arr, axis=0)
        crop_coord_list = [crop_min[0]-border, crop_max[0]+border, height-crop_min[1]+border, height-crop_max[1]-border]
    else:
        crop_coord_list = None

    for i in range(top_n):
        # Remove the spots not included in the top connected components
        data_mod_xy = data_mod[data_mod.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))] != 0, :].copy()
    
        # Make categorical variables
        data_mod_xy.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))] = \
            data_mod_xy.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))].astype('category')
    
        sc.pl.spatial(data_mod_xy, img_key=image_res,
                      color='_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y)),
                      library_id=batch_library_dict[batch_name],
                      cmap = cmap, size=spot_size, alpha = alpha,
                      alpha_img = alpha_img,
                      legend_loc = None, ax = axs[i//4][i%4], show = False, crop_coord = crop_coord_list)
        axs[i//4][i%4].set_title(feat_name_x+' & '+feat_name_y+'\n'+title+" top "+str(i+1)+" CCxy", fontsize = fontsize)
        
    if save: plt.savefig(os.path.join(path,'_'.join(('J_top',str(top_n),
                                                    feat_name_x,feat_name_y+save_name_add+'.png'))), dpi=dpi)

    if return_axis: return axs
    else: plt.show()



def vis_all_connected_(data, vis_intersect_only = False, cmap='tab20', spot_size=1, 
                       alpha_img=0.8, alpha = 0.8, feat_name_x='', feat_name_y='',
                       fig_size=(20,10), batch_colname='batch', batch_name='0', batch_library_dict=None,
                       image_res = 'hires', adjust_image = True, border = 50, 
                       fontsize=30, title = 'Locations of', return_axis=False,
                       save = False, path = os.getcwd(), save_name_add = '', dpi = 300):
    '''
    ## Visualizing all connected components x and y on tissue  
    ### Input  
    data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
    vis_intersect_only: 
        visualize only the intersecting spots for connected components of featrure x and y
        -> spots are color-coded by connected component in x
    cmap: colormap for the visualization of CC identity
    spot_size: size of the spot visualized on the tissue
    alpha_img: transparency of the tissue, alpha: transparency of the colored spot
    feat_name_x, feat_name_y: name of the feature x and y

    fig_size: size of the drawn figure
    batch_colname: column name to categorize the batch in .obs
    batch_name: the name of the batch slide to visualize (should be one of the elements of batch in .obs)
    batch_library_dict: dictionary that matches batch name with library keys in adata.uns["spatial"]
        -> can be utilized When the multiple Visium slides are merged.
        -> if not provided, then categories for batch_colname in .obs will be matched with library keys in adata.uns["spatial"]

    image_res: resolution of the tissue image to be used in visualization ('hires' or 'lowres')
    adjust_image: whether to adjust the image to show the whole tissue image, if False then crop and show the location of connected component spots only
    border: border of the spots around the spots; this information is used to adjust the image
    fontsize: size of the figure title, title: title of the figure
    return_axis: whether to return the plot axis

    save: whether to save of figure, path: saving path
    save_name_add: additional name to be added in the end of the filename
    dpi: dpi for image

    ### Outut
    axs: matplotlib axis for the plot
    '''
    # Check the feasibility of the dataset
    if set(['Comb_CC_'+feat_name_x,'Comb_CC_'+feat_name_y]) <= set(data.obs.columns):
        data_mod_x = data.copy()
        data_mod_y = data.copy()
    else:
        raise ValueError("No CC location data for the given 'feat_x' and 'feat_y'")

    # Check the feasibility of the dataset
    if batch_name not in data.obs[batch_colname].values:
        raise ValueError("'batch_name' not among the elements of 'batch_colname' in .obs")
    batch_keys = pd.unique(data.obs[batch_colname]).tolist()
    # Generate dictionary between batch keys and library keys if not given
    if batch_library_dict is None:
        library_keys = list(data.uns['spatial'].keys())
        if len(library_keys) == len(batch_keys): batch_library_dict = dict(zip(batch_keys, library_keys))
        else: raise ValueError("Number of library keys and batches are different")
    # Subset the spatial data to contain only the batch_name slide
    if len(batch_keys) > 1:
        # Subset the dataset to contain only the batch_name slide
        data_mod_x = data_mod_x[data_mod_x.obs[batch_colname]==batch_name].copy()

    # Calculate intersecting spots between connected component x and y to be visualized
    if vis_intersect_only:
        cc_loc_x_df = data_mod_x.obs['Comb_CC_'+feat_name_x].astype(int)
        cc_loc_y_df = data_mod_y.obs['Comb_CC_'+feat_name_y].astype(int)
        data_mod_x.obs['_'.join(('Comb_CCxy_int',feat_name_x,feat_name_y))] = \
            (cc_loc_x_df * ((cc_loc_x_df != 0) & (cc_loc_y_df != 0))).astype('category')
        
        # Remove the spots not included in the top connected components
        data_mod_x = data_mod_x[data_mod_x.obs['_'.join(('Comb_CCxy_int',feat_name_x,feat_name_y))] != 0, :].copy()
    else:
        # Subest the dataset to contain only the batch_name slide
        data_mod_y = data_mod_y[data_mod_y.obs[batch_colname]==batch_name].copy()

        # Remove the spots not included in the top connected components
        data_mod_x = data_mod_x[data_mod_x.obs['Comb_CC_'+feat_name_x] != 0, :].copy()
        data_mod_y = data_mod_y[data_mod_y.obs['Comb_CC_'+feat_name_y] != 0, :].copy()

        # Make categorical variables
        data_mod_x.obs['Comb_CC_'+feat_name_x] = data_mod_x.obs['Comb_CC_'+feat_name_x].astype('category')
        data_mod_y.obs['Comb_CC_'+feat_name_y] = data_mod_y.obs['Comb_CC_'+feat_name_y].astype('category')
            
    # Set figure parameters
    sc.set_figure_params(figsize=fig_size, facecolor='white', frameon=False)
    if vis_intersect_only:
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    else:
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    
    # Adjust the image to contain the whole slide image
    if adjust_image:
        # Crop the image with certain borders
        height = data.uns['spatial'][batch_library_dict[batch_name]]['images'][image_res].shape[1]
        spot_coord_arr = data.obsm['spatial'] * data.uns['spatial'][batch_library_dict[batch_name]]['scalefactors']['tissue_'+image_res+'_scalef']
        crop_max = np.max(spot_coord_arr, axis=0)
        crop_min = np.min(spot_coord_arr, axis=0)
        crop_coord_list = [crop_min[0]-border, crop_max[0]+border, height-crop_min[1]+border, height-crop_max[1]-border]
    else:
        crop_coord_list = None

    if vis_intersect_only:
        sc.pl.spatial(data_mod_x, img_key=image_res, color='_'.join(('Comb_CCxy_int',feat_name_x,feat_name_y)),
                      library_id=batch_library_dict[batch_name],
                      cmap = cmap, size=spot_size, alpha_img = alpha_img,
                      alpha = alpha, legend_loc = None, ax = axs, show = False, crop_coord = crop_coord_list)
        axs.set_title(feat_name_x+' & '+feat_name_y+'\n'+title+" CC", fontsize = fontsize)
    else:
        # Plot the top genes
        sc.pl.spatial(data_mod_x, img_key=image_res, color='_'.join(('Comb_CC',feat_name_x)),
                      library_id=batch_library_dict[batch_name],
                      cmap = cmap, size=spot_size, alpha_img = alpha_img,
                      alpha = alpha, legend_loc = None, ax = axs[0], show = False, crop_coord = crop_coord_list)
        sc.pl.spatial(data_mod_y, img_key=image_res, color='_'.join(('Comb_CC',feat_name_y)),
                      library_id=batch_library_dict[batch_name],
                      cmap = cmap, size=spot_size, alpha_img = alpha_img,
                      alpha = alpha, legend_loc = None, ax = axs[1], show = False, crop_coord = crop_coord_list)
        axs[0].set_title(feat_name_x+'\n'+title+" CC", fontsize = fontsize)
        axs[1].set_title(feat_name_y+'\n'+title+" CC", fontsize = fontsize)
    
    if save: plt.savefig(os.path.join(path,
                                    '_'.join(('Loc_CCxy',feat_name_x,feat_name_y,save_name_add+'.png'))), dpi=dpi)

    if return_axis: 
        if vis_intersect_only: return axs
        else: return (axs[0], axs[1])
    else:
        plt.show()