import os
from unicodedata import category
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
from .jaccard import jaccard_top_n_connected_loc_

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import offsetbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



def vis_jaccard_top_n_pair_visium(data, feat_name_x='', feat_name_y='',
                                  top_n = 5, spot_size=1, alpha_img=0.8, alpha = 0.8, 
                                  fig_size = (10,10), batch_colname='batch', batch_name='0', batch_library_dict=None,
                                  image_res = 'hires', adjust_image = True, border = 50, 
                                  title_fontsize=30, legend_fontsize=None, title = 'J', return_axis=False,
                                  save = False, path = os.getcwd(), save_name_add = '', dpi=150):
    '''
    ## Visualizing top n connected component x and y showing maximum Jaccard index in Visium dataset
    ### Input
    data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
    feat_name_x, feat_name_y: name of the feature x and y
    top_n: the number of the top connected component pairs withthe  highest Jaccard similarity index
    spot_size: size of the spot visualized on the tissue
    alpha_img: transparency of the tissue, alpha: transparency of the colored spot

    fig_size: size of the drawn figure
    batch_colname: column name to categorize the batch in .obs
    batch_name: the name of the batch slide to visualize (should be one of the elements of batch in .obs)
    batch_library_dict: dictionary that matches batch name with library keys in adata.uns["spatial"]
        -> can be utilized When the multiple Visium slides are merged.
        -> if not provided, then categories for batch_colname in .obs will be matched with library keys in adata.uns["spatial"]

    image_res: resolution of the tissue image to be used in visualization ('hires' or 'lowres')
    adjust_image: whether to adjust the image to show the whole tissue image, if False then crop and show the location of connected component spots only
    border: border of the spots around the spots; this information is used to adjust the image
    title_fontsize: size of the figure title, legend_fontsize: size of the legend text, title: title of the figure
    return_axis: whether to return the plot axis

    save: whether to save of figure, path: saving path
    save_name_add: additional name to be added in the end of the filename
    dpi: dpi for image

    ### Outut
    axs: matplotlib axis for the plot
    '''
    # Set figure parameters
    xsize = ((top_n-1)//4)+1
    if xsize > 1: ysize = 4
    else: ysize = ((top_n-1)%4)+1
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
    data_mod, J_top_n = jaccard_top_n_connected_loc_(data_mod, CCx=None, CCy=None, 
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

    # Define the colormap with three different colors: for CC locations of feature x, feature_y and intersecting regions
    colormap = ["#FBBC05","#4285F4","#34A853"]
    cmap = colors.ListedColormap(colormap)

    for i in range(top_n):
        # Remove the spots not included in the top connected components
        data_mod_xy = data_mod[(data_mod.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))] != 0), :].copy()
        
        # Make categorical variables
        data_mod_xy.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))] = \
            data_mod_xy.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))].astype('category')

        sc.pl.spatial(data_mod_xy, img_key=image_res,
                      color='_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y)),
                      library_id=batch_library_dict[batch_name],
                      cmap = cmap, size=spot_size, alpha = alpha,
                      alpha_img = alpha_img,
                      legend_loc = None, ax = axs[i//4][i%4], show = False, crop_coord = crop_coord_list)
        axs[i//4][i%4].set_title(feat_name_x+' & '+feat_name_y+'\n'+title+" top "+str(i+1)+" CCxy", fontsize = title_fontsize)
        axs[i//4][i%4].add_artist(offsetbox.AnchoredText(f'J = {J_top_n[i]:.3}', loc='upper right', frameon=False, prop=dict(size = fig_size[0]*2)))
    
    # Add legend to the figure
    category_label = [feat_name_x,feat_name_y,"Overlap"]
    for index, label in enumerate(category_label):
        plt.scatter([], [], c=colormap[index], label=label)
    if legend_fontsize is None: legend_fontsize=fig_size[1]*xsize*2
    plt.legend(frameon=False, loc='center left',  bbox_to_anchor=(1, 0.5), ncol=1, fontsize=legend_fontsize)#, bbox_transform = plt.gcf().transFigure)

    if save: fig.savefig(os.path.join(path,'_'.join(('Visium_J_top',str(top_n),
                                                    feat_name_x,feat_name_y+save_name_add+'.png'))), dpi=dpi)

    if return_axis: return axs
    else: plt.show()



def vis_all_connected_visium(data, feat_name_x='', feat_name_y='',
                             spot_size=1, alpha_img=0.8, alpha = 0.8, 
                             fig_size=(10,10), batch_colname='batch', batch_name='0', batch_library_dict=None,
                             image_res = 'hires', adjust_image = True, border = 50, 
                             title_fontsize=30, legend_fontsize=None, title = 'Locations of', return_axis=False,
                             save = False, path = os.getcwd(), save_name_add = '', dpi = 150):
    '''
    ## Visualizing all connected components x and y on tissue in Visium dataset
    -> Overlapping conected component locations in green, exclusive locations for x and y in red and blue, respectively
    ### Input  
    data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
    feat_name_x, feat_name_y: name of the feature x and y
    vis_intersect_only: 
        visualize only the intersecting spots for connected components of featrure x and y
        -> spots are color-coded by connected component in x
    spot_size: size of the spot visualized on the tissue
    alpha_img: transparency of the tissue, alpha: transparency of the colored spot

    fig_size: size of the drawn figure
    batch_colname: column name to categorize the batch in .obs
    batch_name: the name of the batch slide to visualize (should be one of the elements of batch in .obs)
    batch_library_dict: dictionary that matches batch name with library keys in adata.uns["spatial"]
        -> can be utilized When the multiple Visium slides are merged.
        -> if not provided, then categories for batch_colname in .obs will be matched with library keys in adata.uns["spatial"]

    image_res: resolution of the tissue image to be used in visualization ('hires' or 'lowres')
    adjust_image: whether to adjust the image to show the whole tissue image, if False then crop and show the location of connected component spots only
    border: border of the spots around the spots; this information is used to adjust the image
    title_fontsize: size of the figure title, legend_fontsize: size of the legend text, title: title of the figure
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
    cc_loc_x_df = data_mod_x.obs['Comb_CC_'+feat_name_x].astype(int)
    cc_loc_y_df = data_mod_y.obs['Comb_CC_'+feat_name_y].astype(int)
    data_mod_x.obs['_'.join(('Comb_CCxy_int',feat_name_x,feat_name_y))] = \
            ((1 * ((cc_loc_x_df != 0) & (cc_loc_y_df == 0))) + \
            (2 * ((cc_loc_x_df == 0) & (cc_loc_y_df != 0))) + \
            (3 * ((cc_loc_x_df != 0) & (cc_loc_y_df != 0)))).astype('category')
        
    # Remove the spots not included in the top connected components
    data_mod_x = data_mod_x[data_mod_x.obs['_'.join(('Comb_CCxy_int',feat_name_x,feat_name_y))] != 0, :].copy()
            
    # Set figure parameters
    sc.set_figure_params(figsize=fig_size, facecolor='white', frameon=False)
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    
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
    
    # Define the colormap with three different colors: for CC locations of feature x, feature_y and intersecting regions
    colormap = ["#FBBC05","#4285F4","#34A853"]
    cmap = colors.ListedColormap(colormap)

    sc.pl.spatial(data_mod_x, img_key=image_res, color='_'.join(('Comb_CCxy_int',feat_name_x,feat_name_y)),
                  library_id=batch_library_dict[batch_name],
                  cmap = cmap, size=spot_size, alpha_img = alpha_img,
                  alpha = alpha, legend_loc = None, ax = axs, show = False, crop_coord = crop_coord_list)
    axs.set_title(feat_name_x+' & '+feat_name_y+'\n'+title+" CC", fontsize = title_fontsize)
    
    # Add legend to the figure
    category_label = [feat_name_x,feat_name_y,"Overlap"]
    for index, label in enumerate(category_label):
        axs.scatter([], [], c=colormap[index], label=label)
    if legend_fontsize is None: legend_fontsize=fig_size[1]
    plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=legend_fontsize)
    
    if save: fig.savefig(os.path.join(path,
                                    '_'.join(('Visium_loc_CCxy',feat_name_x,feat_name_y,save_name_add+'.png'))), dpi=dpi)

    if return_axis: return axs
    else: plt.show()



def vis_spatial_cosmx_(data, feat_name='', cmap = None, dot_size=None, alpha = 0.8, 
                       fig_size = (10,10), title_fontsize = 30, legend_fontsize = None, title = None, 
                       return_axis=False, save = False, path = os.getcwd(), save_name_add = '', dpi=150):
    '''
    ## Visualizing spatial distribution of features in CosMx dataset
    ### Input
    data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
    feat_name_x, feat_name_y: name of the feature x and y
    cmap: colormap for the visualization of CC identity
    dot_size: size of the spot visualized on the tissue
    alpha: transparency of the colored spot

    fig_size: size of the drawn figure
    title_fontsize: size of the figure title, legend_fontsize: size of the legend text, title: title of the figure
    return_axis: whether to return the plot axis

    save: whether to save of figure, path: saving path
    save_name_add: additional name to be added in the end of the filename
    dpi: dpi for image

    ### Outut
    axs: matplotlib axis for the plot
    '''
    try: tsimg = data.obs.loc[:,['array_col','array_row']].to_numpy()
    except: raise ValueError("'data' should contain coordinates of spots in .obs as 'array_col' and 'array_row'")
    tsimg_row = tsimg[:,1]
    tsimg_col = tsimg[:,0]

    # Set figure parameters
    plt.rcParams.update(plt.rcParamsDefault)
    fig, axs = plt.subplots(1,1, sharey=True, tight_layout=True, figsize=fig_size)
    if dot_size is None: dot_size = fig_size[1]/1.5

    if feat_name in data.var_names:
        if isinstance(data.X, np.ndarray): feat_data = data[:,feat_name].X.reshape(-1)
        elif isinstance(data.X, sparse.spmatrix): feat_data = data[:,feat_name].X.toarray().reshape(-1)
        else: ValueError("'data.X' should be either numpy ndarray or scipy sparse matrix")
        data_type='others'
    elif feat_name in data.obs.columns:
        if data.obs[feat_name].dtype.name == 'category': data_type='category'
        else: data_type='others'
        feat_data = data.obs[feat_name].to_numpy()
    else:
        raise ValueError("'feat_name' not found in .var_names and .obs.columns")

    if data_type == 'others':
        if cmap is None: cmap = 'viridis'
        im = axs.scatter(tsimg_col, tsimg_row, s = dot_size**2, 
                        c = feat_data, cmap = cmap, linewidth = 0, alpha=alpha, marker="s")
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax = cax)
        if legend_fontsize is None: legend_fontsize=fig_size[1]*2
        cbar.ax.tick_params(labelsize=legend_fontsize)
    else:
        # Factorize the data and draw sccater plot
        factorize_data = pd.factorize(feat_data, sort=True)
        cats = factorize_data[1].tolist()
        if cmap is None:
            colormap = ["#a2e1ca", "#110f1f", "#f09bf1", "#02531d", "#3ba7e5", 
                        "#730f44", "#15974d", "#f75ef0", "#1357ca", "#c0e15c", "#fb2076", "#859947", 
                        "#214a65", "#e7ad79", "#5a3100", "#fd8992", "#900e08", "#fbd127", "#270fe2", 
                        "#fb7810", "#922eb1", "#9f6c3b", "#fe2b27","#8adc30", "#2e0d93", "#8de6c0", 
                        "#370e01", "#e8ced5", "#113630", "#1cf1a3", "#1e1e58", "#f09ede", "#48950f", 
                        "#a93aae", "#20f53d", "#8c1132", "#38b5fc", "#805f84", "#577cf5", "#e2d923", "#69ef7b","#1e0e76"]
            cmap = colors.ListedColormap(colormap)
        axs.scatter(tsimg_col, tsimg_row, s = dot_size**2, 
                    c = factorize_data[0], cmap = cmap, linewidth = 0, alpha=alpha, marker="s")

        for index, label in enumerate(cats):
            axs.scatter([], [], c=colormap[index], label=label)
        if legend_fontsize is None: legend_fontsize=fig_size[1]
        axs.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), ncol=((len(cats)-1)//20)+1, fontsize=legend_fontsize)

    axs.axis('off')
    if title is None: axs.set_title(str(feat_name), fontsize=title_fontsize)
    else: axs.set_title(title + ' '+ str(feat_name), fontsize=title_fontsize)

    if save: fig.savefig(os.path.join(path, '_'.join(('CosMx_spatial',feat_name,save_name_add+'.png'))), dpi=dpi)
    
    if return_axis: return axs
    else: plt.show()



def vis_jaccard_top_n_pair_cosmx(data, feat_name_x='', feat_name_y='',
                                 top_n = 5, dot_size=None, alpha = 0.8, 
                                 fig_size = (10,10), title_fontsize = 30, legend_fontsize = None,
                                 title = 'J', return_axis=False,
                                 save = False, path = os.getcwd(), save_name_add = '', dpi=150):
    '''
    ## Visualizing top n connected component x and y showing maximum Jaccard index in CosMx dataset
    -> Overlapping conected component locations in green, exclusive locations for x and y in red and blue, respectively
    ### Input
    data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
    feat_name_x, feat_name_y: name of the feature x and y
    top_n: the number of the top connected component pairs withthe  highest Jaccard similarity index
    dot_size: size of the spot visualized on the tissue
    alpha: transparency of the colored spot

    fig_size: size of the drawn figure
    title_fontsize: size of the figure title, legend_fontsize: size of the legend text, title: title of the figure
    return_axis: whether to return the plot axis

    save: whether to save of figure, path: saving path
    save_name_add: additional name to be added in the end of the filename
    dpi: dpi for image

    ### Outut
    axs: matplotlib axis for the plot
    '''
    try: tsimg = data.obs.loc[:,['array_col','array_row']].to_numpy()
    except: raise ValueError("'data' should contain coordinates of spots in .obs as 'array_col' and 'array_row'")
    tsimg_row = tsimg[:,1]
    tsimg_col = tsimg[:,0]

    # Set figure parameters
    xsize = ((top_n-1)//4)+1
    if xsize > 1: ysize = 4
    else: ysize = ((top_n-1)%4)+1
    sc.set_figure_params(figsize=(fig_size[0]*ysize, fig_size[1]*xsize), facecolor='white', frameon=False)
    fig, axs = plt.subplots(xsize, ysize, sharey=True, tight_layout=True, squeeze=False)

    # Calculate top n connected component location
    data_mod, J_top_n = jaccard_top_n_connected_loc_(data, CCx=None, CCy=None, 
                                                     feat_name_x=feat_name_x, feat_name_y=feat_name_y, top_n = top_n)

    # Define the colormap with three different colors: for CC locations of feature x, feature_y and intersecting regions
    colormap = ["#A2E1CA","#FBBC05","#4285F4","#34A853"]
    cmap = colors.ListedColormap(colormap)
    if dot_size is None: dot_size = fig_size[0]/1.5

    for i in range(top_n):
        # Make categorical variables
        data_mod.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))] = \
            data_mod.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))].astype('category')

        # Factorize the data and draw sccater plot
        factorize_data = pd.factorize(data_mod.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))], sort=True)
        axs[i//4][i%4].scatter(tsimg_col, tsimg_row, s = dot_size**2, c = factorize_data[0], 
                                cmap = cmap, alpha=alpha, marker="s")

        axs[i//4][i%4].axis('off')
        axs[i//4][i%4].set_title(feat_name_x+' & '+feat_name_y+'\n'+title+" top "+str(i+1)+" CCxy", fontsize = title_fontsize)
        axs[i//4][i%4].add_artist(offsetbox.AnchoredText(f'J = {J_top_n[i]:.3}', loc='upper right', frameon=False, prop=dict(size = fig_size[0]*2)))
        
    # Add legend to the figure
    category_label = ["Others",feat_name_x,feat_name_y,"Overlap"]
    for index, label in enumerate(category_label):
        plt.scatter([], [], c=colormap[index], label=label)        
    if legend_fontsize is None: legend_fontsize=fig_size[1]*xsize*2
    plt.legend(frameon=False, loc='center left',  bbox_to_anchor=(1, 0.5), ncol=1, fontsize=legend_fontsize)

    if save: fig.savefig(os.path.join(path,'_'.join(('Visium_J_top',str(top_n),
                                                    feat_name_x,feat_name_y+save_name_add+'.png'))), dpi=dpi)

    if return_axis: return axs
    else: plt.show()



def vis_all_connected_cosmx(data, feat_name_x='', feat_name_y='',
                            dot_size=None, alpha = 0.8, 
                            fig_size=(10,10), title_fontsize = 30, legend_fontsize = None, 
                            title = 'Locations of', return_axis=False,
                            save = False, path = os.getcwd(), save_name_add = '', dpi = 150):
    '''
    ## Visualizing all connected components x and y on tissue in CosMx dataset
    -> Overlapping conected component locations in green, exclusive locations for x and y in red and blue, respectively
    ### Input  
    data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
    feat_name_x, feat_name_y: name of the feature x and y
    dot_size: size of the spot visualized on the tissue
    alpha: transparency of the colored spot

    fig_size: size of the drawn figure
    title_fontsize: size of the figure title, legend_fontsize: size of the legend text, title: title of the figure
    return_axis: whether to return the plot axis

    save: whether to save of figure, path: saving path
    save_name_add: additional name to be added in the end of the filename
    dpi: dpi for image

    ### Outut
    axs: matplotlib axis for the plot
    '''
    try: tsimg = data.obs.loc[:,['array_col','array_row']].to_numpy()
    except: raise ValueError("'data' should contain coordinates of spots in .obs as 'array_col' and 'array_row'")
    tsimg_row = tsimg[:,1]
    tsimg_col = tsimg[:,0]

    plt.rcParams.update(plt.rcParamsDefault)
    fig, axs = plt.subplots(1,1, sharey=True, tight_layout=True, figsize=fig_size)

    # Calculate overlapping locations
    cc_loc_x_df = data.obs['Comb_CC_'+feat_name_x].astype(int)
    cc_loc_y_df = data.obs['Comb_CC_'+feat_name_y].astype(int)
    data_mod = data.copy()
    data_mod.obs['Over'] = ((1 * ((cc_loc_x_df != 0) & (cc_loc_y_df == 0))) + \
                            (2 * ((cc_loc_x_df == 0) & (cc_loc_y_df != 0))) + \
                            (3 * ((cc_loc_x_df != 0) & (cc_loc_y_df != 0)))).astype('category')

    # Define the colormap with three different colors: for CC locations of feature x, feature_y and intersecting regions
    colormap = ["#A2E1CA","#FBBC05","#4285F4","#34A853"]
    cmap = colors.ListedColormap(colormap)
    if dot_size is None: dot_size = fig_size[0]/1.5
    
    # Factorize the data and draw sccater plot
    factorize_data = pd.factorize(data_mod.obs['Over'], sort=True)
    axs.scatter(tsimg_col, tsimg_row, s = dot_size**2, c = factorize_data[0], 
                cmap = cmap, linewidth = 0, alpha=alpha, marker="s")

    category_label = ["Others",feat_name_x,feat_name_y,"Overlap"]
    for index, label in enumerate(category_label):
        axs.scatter([], [], c=colormap[index], label=label)        
    if legend_fontsize is None: legend_fontsize=fig_size[1]
    axs.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=legend_fontsize)

    axs.axis('off')
    axs.set_title(feat_name_x+' & '+feat_name_y+'\n'+title+" CC", fontsize = title_fontsize)

    if save:
        fig.savefig(os.path.join(path,'_'.join(('CosMx_loc_CCxy',feat_name_x,feat_name_y,save_name_add+'.png'))), dpi=dpi)
    
    if return_axis: return axs
    else: plt.show()