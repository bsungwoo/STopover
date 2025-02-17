import os
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
from .jaccard import jaccard_top_n_connected_loc_
from .jaccard import jaccard_and_connected_loc_

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import offsetbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



def vis_jaccard_top_n_pair_visium(data, feat_name_x='', feat_name_y='',
                                  top_n = 5, jaccard_type='default', ncol = 4, spot_size=1, alpha_img=0.8, alpha = 0.8, 
                                  fig_size = (5,5), batch_colname='batch', batch_name='0', batch_library_dict=None,
                                  image_res = 'hires', adjust_image = True, border = 500, 
                                  title_fontsize=20, legend_fontsize=None, title = '', return_axis=False,
                                  save = False, path = os.getcwd(), save_name_add = '', dpi=150):
    '''
    ## Visualizing top n connected component x and y showing maximum Jaccard index
    ### Input
    * feat_name_x, feat_name_y: name of the feature x and y
    * top_n: the number of the top connected component pairs withthe  highest Jaccard similarity index
    * jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)
    * ncol: number of columns to visualize top n CCs
    * spot_size: size of the spot visualized on the tissue
    * alpha_img: transparency of the tissue, alpha: transparency of the colored spot

    * fig_size: size of the drawn figure
    * batch_colname: column name to categorize the batch in .obs
    * batch_name: the name of the batch slide to visualize (should be one of the elements of batch in .obs)
    * batch_library_dict: dictionary that matches batch name with library keys in adata.uns["spatial"]
        -> can be utilized When the multiple Visium slides are merged.
        -> if not provided, then categories for batch_colname in .obs will be matched with library keys in adata.uns["spatial"]

    * image_res: resolution of the tissue image to be used in visualization ('hires' or 'lowres')
    * adjust_image: whether to adjust the image to show the whole tissue image, if False then crop and show the location of connected component spots only
    * border: border of the spots around the spots; this information is used to adjust the image
    * title_fontsize: size of the figure title, legend_fontsize: size of the legend text, title: title of the figure  
     * return_axis: whether to return the plot axis

    * save: whether to save of figure, path: saving path
    * save_name_add: additional name to be added in the end of the filename
    * dpi: dpi for image

    ### Outut
    * axs: matplotlib axis for the plot
    '''
    # Set figure parameters
    xsize = ((top_n-1)//ncol)+1
    if xsize > 1: ysize = ncol
    else: ysize = ((top_n-1)%ncol)+1
    sc.set_figure_params(figsize=(fig_size[0]*ysize, fig_size[1]*xsize), facecolor='white', frameon=False)
    fig, axs = plt.subplots(xsize, ysize, tight_layout=True, squeeze=False)

    library_keys = list(data.uns['spatial'].keys())
    if len(library_keys) > 1:
        # Check the feasibility of the dataset
        if batch_name not in data.obs[batch_colname].values:
            raise ValueError("'batch_name' not among the elements of 'batch_colname' in .obs")

        batch_keys = pd.unique(data.obs[batch_colname]).tolist()
        # Generate dictionary between batch keys and library keys if not given
        if batch_library_dict is None:     
            if len(library_keys) == len(batch_keys): batch_library_dict = dict(zip(batch_keys, library_keys))
            else: raise ValueError("Number of library keys and batches are different")
        # Subset the dataset to contain only the batch_name slide
        data_mod = data[data.obs[batch_colname]==batch_name].copy()
    else:
        data_mod = data.copy()
        batch_library_dict = dict({'0': library_keys[0]})

    # Calculate top n connected component location
    data_mod, J_top_n = jaccard_top_n_connected_loc_(data_mod, CCx=None, CCy=None, feat_name_x=feat_name_x, feat_name_y=feat_name_y, 
                                                     top_n = top_n, jaccard_type=jaccard_type)

    # Adjust the image to contain the whole slide image
    if adjust_image:
        # Crop the image with certain borders
        spot_coord_arr = data.obsm['spatial']
        crop_max = np.max(spot_coord_arr, axis=0)
        crop_min = np.min(spot_coord_arr, axis=0)
        crop_coord_list = [crop_min[0]-border, crop_max[0]+border, crop_min[1]-border, crop_max[1]+border]
    else:
        crop_coord_list = None

    for i in range(top_n):
        # Remove the spots not included in the top connected components
        data_mod_xy = data_mod[(data_mod.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))] != 0), :].copy()
        
        # Define the colormap with three different colors: for CC locations of feature x, feature_y and intersecting regions
        colormap_mod = [["#A2E1CA","#FBBC05","#4285F4","#34A853"][index] for index in data_mod_xy.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))].cat.categories]
        sc.pl.spatial(data_mod_xy, img_key=image_res,
                      color='_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y)),
                      library_id=batch_library_dict[batch_name],
                      palette = colormap_mod, size=spot_size, alpha = alpha,
                      alpha_img = alpha_img,
                      legend_loc = None, ax = axs[i//ncol][i%ncol], show = False, crop_coord = crop_coord_list)
        axs[i//ncol][i%ncol].set_title(feat_name_x+' & '+feat_name_y+title+'\n'+"top "+str(i+1)+" CCxy", fontsize = title_fontsize)
        axs[i//ncol][i%ncol].add_artist(offsetbox.AnchoredText(f'J_local = {J_top_n[i]:.3f}', loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=axs[i//ncol][i%ncol].transAxes,
                                                         frameon=False, prop=dict(size = fig_size[0]*2.5)))

    if top_n < xsize*ysize: 
        for i in range(top_n, xsize*ysize): axs[i//ncol][i%ncol].axis('off')
    
    # Add legend to the figure
    colormap = ["#FBBC05","#4285F4","#34A853"]
    category_label = [feat_name_x,feat_name_y,"Overlap"]
    for index, label in enumerate(category_label):
        plt.scatter([], [], c=colormap[index], label=label)
    if legend_fontsize is None: legend_fontsize=fig_size[1]*xsize*2
    plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=legend_fontsize)#, bbox_transform = plt.gcf().transFigure)

    if save: fig.savefig(os.path.join(path,'_'.join(('Visium_J_top',str(top_n),
                                                    feat_name_x,feat_name_y))+save_name_add+'.png'), dpi=dpi)

    if return_axis: return axs
    else: plt.show()



def vis_all_connected_visium(data, feat_name_x='', feat_name_y='',
                             spot_size=1, alpha_img=0.8, alpha = 0.8, vis_jaccard=True, jaccard_type='default',
                             fig_size=(5,5), batch_colname='batch', batch_name='0', batch_library_dict=None,
                             image_res = 'hires', adjust_image = True, border = 500, 
                             title_fontsize=20, legend_fontsize=None, title = 'Locations of', 
                             return_axis=False, axis = None,
                             save = False, path = os.getcwd(), save_name_add = '', dpi = 150):
    '''
    ## Visualizing all connected components x and y on tissue  
    ### Input  
    * feat_name_x, feat_name_y: name of the feature x and y
    * spot_size: size of the spot visualized on the tissue
    * alpha_img: transparency of the tissue, alpha: transparency of the colored spot
    * vis_jaccard: whether to visualize jaccard index on right corner of the plot
    * jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

    * fig_size: size of the drawn figure
    * batch_colname: column name to categorize the batch in .obs
    * batch_name: the name of the batch slide to visualize (should be one of the elements of batch in .obs)
    * batch_library_dict: dictionary that matches batch name with library keys in adata.uns["spatial"]
        -> can be utilized When the multiple Visium slides are merged.
        -> if not provided, then categories for batch_colname in .obs will be matched with library keys in adata.uns["spatial"]

    * image_res: resolution of the tissue image to be used in visualization ('hires' or 'lowres')
    * adjust_image: whether to adjust the image to show the whole tissue image, if False then crop and show the location of connected component spots only
    * border: border of the spots around the spots; this information is used to adjust the image
    * fontsize: size of the figure title, title: title of the figure
    * return_axis: whether to return the plot axis
    * axis: matplotlib axes for plotting single image

    * save: whether to save of figure, path: saving path
    * save_name_add: additional name to be added in the end of the filename
    * dpi: dpi for image

    ### Outut
    * axs: matplotlib axis for the plot
    '''
    # Check the feasibility of the dataset
    if set(['Comb_CC_'+feat_name_x,'Comb_CC_'+feat_name_y]) <= set(data.obs.columns):
        data_mod_x = data.copy()
        data_mod_y = data.copy()
    else:
        raise ValueError("No CC location data for the given 'feat_x' and 'feat_y'")

    library_keys = list(data.uns['spatial'].keys())
    if len(library_keys) > 1:
        # Check the feasibility of the dataset
        if batch_name not in data.obs[batch_colname].values:
            raise ValueError("'batch_name' not among the elements of 'batch_colname' in .obs")

        batch_keys = pd.unique(data.obs[batch_colname]).tolist()
        # Generate dictionary between batch keys and library keys if not given
        if batch_library_dict is None:
            if len(library_keys) == len(batch_keys): batch_library_dict = dict(zip(batch_keys, library_keys))
            else: raise ValueError("Number of library keys and batches are different")
        # Subset the dataset to contain only the batch_name slide
        data_mod_x = data_mod_x[data_mod_x.obs[batch_colname]==batch_name].copy()
    else: batch_library_dict = dict({'0': library_keys[0]})

    # Calculate intersecting spots between connected component x and y to be visualized
    cc_loc_x_df = data_mod_x.obs['Comb_CC_'+feat_name_x].astype(int)
    cc_loc_y_df = data_mod_y.obs['Comb_CC_'+feat_name_y].astype(int)
    data_mod_x.obs['Over'] = \
            ((1 * ((cc_loc_x_df != 0) & (cc_loc_y_df == 0))) + \
            (2 * ((cc_loc_x_df == 0) & (cc_loc_y_df != 0))) + \
            (3 * ((cc_loc_x_df != 0) & (cc_loc_y_df != 0)))).astype('category')
    # Calculate J_comp between the two feature pairs
    Jcomp = jaccard_and_connected_loc_(data_mod_x, feat_name_x=feat_name_x, feat_name_y=feat_name_y, J_comp=True,
                                        jaccard_type=jaccard_type, return_mode='jaccard', return_sep_loc=False)
    # Remove the spots not included in the top connected components
    data_mod_x = data_mod_x[data_mod_x.obs['Over'] != 0, :].copy()
            
    # Set figure parameters
    sc.set_figure_params(facecolor='white', frameon=False)
    if axis is None: fig, axs = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
    else: axs = axis
    
    # Adjust the image to contain the whole slide image
    if adjust_image:
        # Crop the image with certain borders
        spot_coord_arr = data.obsm['spatial']
        crop_max = np.max(spot_coord_arr, axis=0)
        crop_min = np.min(spot_coord_arr, axis=0)
        crop_coord_list = [crop_min[0]-border, crop_max[0]+border, crop_min[1]-border, crop_max[1]+border]
    else:
        crop_coord_list = None
    
    # Define the colormap with three different colors: for CC locations of feature x, feature_y and intersecting regions
    colormap = [["#A2E1CA","#FBBC05","#4285F4","#34A853"][index] for index in data_mod_x.obs['Over'].cat.categories]
    category_label = [["Others",feat_name_x,feat_name_y,"Overlap"][index] for index in data_mod_x.obs['Over'].cat.categories]

    sc.pl.spatial(data_mod_x, img_key=image_res, color='Over',
                  library_id=batch_library_dict[batch_name],
                  palette = colormap, size=spot_size, alpha_img = alpha_img,
                  alpha = alpha, legend_loc = None, ax = axs, show = False, crop_coord = crop_coord_list)
    axs.set_title(feat_name_x+' & '+feat_name_y+title, fontsize = title_fontsize)
    if vis_jaccard: axs.add_artist(offsetbox.AnchoredText(f'J_comp = {Jcomp:.3f}', loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=axs.transAxes,
                                                          frameon=False, prop=dict(size = fig_size[1]*2.5)))
    
    # Add legend to the figure
    for index, label in enumerate(category_label):
        axs.scatter([], [], c=colormap[index], label=label)
    if legend_fontsize is None: legend_fontsize=fig_size[1]*2
    axs.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=legend_fontsize)
    
    if save: fig.savefig(os.path.join(path,
                                      '_'.join(('Visium_loc_CCxy',feat_name_x,feat_name_y))+save_name_add+'.png'), dpi=dpi)

    if return_axis: return axs
    elif axis is None: plt.show()



def vis_spatial_imageST_(data, feat_name='', colorlist = None, dot_size=None, alpha = 0.8, vmax=None, vmin=None, sort_labels=True,
                         fig_size = (5,5), title_fontsize = 20, legend_fontsize = None, title = None, 
                         return_axis=False, figure = None, axis = None, save = False, path = os.getcwd(), save_name_add = '', dpi=150):
    '''
    ## Visualizing spatial distribution of features in image-based ST / Visium HD dataset
    ### Input
    * data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
    * feat_name: name of the feature to visualize
    * colorlist: color list for the visualization of CC identity
    * dot_size: size of the spot visualized on the tissue
    * alpha: transparency of the colored spot
    * vmax: maximum value in the colorbar; if None, it will automatically set the maximum value
    * vmax: minimum value in the colorbar; if None, it will automatically set the minimum value
    * sort_labels: sort the category labels in alphanumeric order if the name of categorical feature is provided to 'feat_name'

    * fig_size: size of the drawn figure
    * title_fontsize: size of the figure title, legend_fontsize: size of the legend text, title: title of the figure
    * return_axis: whether to return the plot axis
    * figure: matplotlib figure for plotting single image, axis: matplotlib axes for plotting single image

    * save: whether to save of figure, path: saving path
    * save_name_add: additional name to be added in the end of the filename
    * dpi: dpi for image

    ### Outut
    * axs: matplotlib axis for the plot
    ''' 
    try: tsimg = data.obs.loc[:,['array_col','array_row']].to_numpy()
    except: raise ValueError("'data' should contain coordinates of spots in .obs as 'array_col' and 'array_row'")
    tsimg_row = tsimg[:,1]
    tsimg_col = tsimg[:,0]

    # Set figure parameters
    sc.set_figure_params(facecolor='white', frameon=False)
    if axis is None or figure is None: fig, axs = plt.subplots(1,1, figsize=fig_size, tight_layout=True)
    else: axs = axis; fig = figure
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
        if colorlist is None: cmap = 'viridis'
        im = axs.scatter(tsimg_col, tsimg_row, s = dot_size**2, vmax=vmax, vmin=vmin,
                        c = feat_data, cmap = cmap, linewidth = 0, alpha=alpha, marker="s")
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax = cax)
        if legend_fontsize is None: legend_fontsize=fig_size[1]*2
        cbar.ax.tick_params(labelsize=legend_fontsize)
    else:
        # Factorize the data and draw sccater plot
        factorize_data = pd.factorize(feat_data, sort=sort_labels)
        cats = factorize_data[1].tolist()
        if colorlist is None:
            colorlist = ["#a2e1ca", "#110f1f", "#f09bf1", "#02531d", "#3ba7e5", 
                        "#730f44", "#15974d", "#f75ef0", "#1357ca", "#c0e15c", "#fb2076", "#859947", 
                        "#214a65", "#e7ad79", "#5a3100", "#fd8992", "#900e08", "#fbd127", "#270fe2", 
                        "#fb7810", "#922eb1", "#9f6c3b", "#fe2b27","#8adc30", "#2e0d93", "#8de6c0", 
                        "#370e01", "#e8ced5", "#113630", "#1cf1a3", "#1e1e58", "#f09ede", "#48950f", 
                        "#a93aae", "#20f53d", "#8c1132", "#38b5fc", "#805f84", "#577cf5", "#e2d923", "#69ef7b","#1e0e76"]
        if len(cats) > len(colorlist): colorlist = colorlist * ((len(cats)-1)//len(colorlist)) + colorlist[:len(cats)%len(colorlist)]
        if len(cats) < len(colorlist): colorlist = colorlist[:len(cats)]
        cmap = colors.ListedColormap(colorlist)
        axs.scatter(tsimg_col, tsimg_row, s = dot_size**2, 
                    c = factorize_data[0], cmap = cmap, linewidth = 0, alpha=alpha, marker="s")

        for index, label in enumerate(cats):
            axs.scatter([], [], c=colorlist[index], label=label)
        if legend_fontsize is None: legend_fontsize=fig_size[1]
        axs.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), ncol=((len(cats)-1)//20)+1, fontsize=legend_fontsize)

    axs.axis('off')
    if feat_name in data.var_names: add_param = ", style = 'italic'" # write in italics if the gene symbol is an input
    else: add_param = ""
    if title is None: eval("axs.set_title(str(feat_name), fontsize=title_fontsize"+add_param+")")
    else: eval("axs.set_title(title + str(feat_name), fontsize=title_fontsize"+add_param+")")

    if save: fig.savefig(os.path.join(path, '_'.join(('CosMx_spatial',feat_name))+save_name_add+'.png'), dpi=dpi)
    
    if return_axis: return axs
    elif axis is None or figure is None: plt.show()



def vis_jaccard_top_n_pair_imageST(data, feat_name_x='', feat_name_y='',
                                   top_n = 5, jaccard_type='default', ncol=4, dot_size=None, alpha = 0.8, 
                                   fig_size = (5,5), title_fontsize = 20, legend_fontsize = None,
                                   title = '', return_axis=False,
                                   save = False, path = os.getcwd(), save_name_add = '', dpi=150):
    '''
    ## Visualizing top n connected component x and y showing maximum Jaccard index in image-based ST / Visium HD dataset
        -> Overlapping conected component locations in green, exclusive locations for x and y in orange and blue, respectively
    ### Input
    * data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
    * feat_name_x, feat_name_y: name of the feature x and y
    * top_n: the number of the top connected component pairs withthe  highest Jaccard similarity index
    * jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)
    * ncol: number of columns to visualize top n CCs
    * dot_size: size of the spot visualized on the tissue
    * alpha: transparency of the colored spot

    * fig_size: size of the dn figure
    * title_fontsize: size of the figure title, legend_fontsize: size of the legend text, title: title of the figure
    * return_axis: whether to return the plot axis

    * save: whether to save of figure, path: saving path
    * save_name_add: additional name to be added in the end of the filename
    * dpi: dpi for image

    ### Outut
    * axs: matplotlib axis for the plot
    '''
    try: tsimg = data.obs.loc[:,['array_col','array_row']].to_numpy()
    except: raise ValueError("'data' should contain coordinates of spots in .obs as 'array_col' and 'array_row'")
    tsimg_row = tsimg[:,1]
    tsimg_col = tsimg[:,0]

    # Set figure parameters
    xsize = ((top_n-1)//ncol)+1
    if xsize > 1: ysize = ncol
    else: ysize = ((top_n-1)%ncol)+1
    sc.set_figure_params(figsize=(fig_size[0]*ysize, fig_size[1]*xsize), facecolor='white', frameon=False)
    fig, axs = plt.subplots(xsize, ysize, tight_layout=True, squeeze=False)

    # Calculate top n connected component location
    data_mod, J_top_n = jaccard_top_n_connected_loc_(data, CCx=None, CCy=None, feat_name_x=feat_name_x, feat_name_y=feat_name_y, 
                                                     top_n = top_n, jaccard_type=jaccard_type)

    # Define the colormap with three different colors: for CC locations of feature x, feature_y and intersecting regions
    colorlist = ["#A2E1CA","#FBBC05","#4285F4","#34A853"]
    if dot_size is None: dot_size = fig_size[0]/1.5

    for i in range(top_n):
        # Factorize the data and draw sccater plot
        factorize_data = pd.factorize(data_mod.obs['_'.join(('CCxy_top',str(i+1),feat_name_x,feat_name_y))], sort=True)
        colorlist_mod = [["#A2E1CA","#FBBC05","#4285F4","#34A853"][index] for index in factorize_data[1]]
        cmap = colors.ListedColormap(colorlist_mod)
        axs[i//ncol][i%ncol].scatter(tsimg_col, tsimg_row, s = dot_size**2, c = factorize_data[0], 
                                cmap = cmap, alpha=alpha, marker="s")

        axs[i//ncol][i%ncol].axis('off')
        axs[i//ncol][i%ncol].set_title(feat_name_x+' & '+feat_name_y+title+'\n'+"top "+str(i+1)+" CCxy", fontsize = title_fontsize)
        axs[i//ncol][i%ncol].add_artist(offsetbox.AnchoredText(f'J_local = {J_top_n[i]:.3f}', loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=axs[i//ncol][i%ncol].transAxes,
                                                        frameon=False, prop=dict(size = fig_size[0]*2.5)))
    if top_n < xsize*ysize: 
        for i in range(top_n, xsize*ysize): axs[i//ncol][i%ncol].axis('off')
        
    # Add legend to the figure
    category_label = ["Others",feat_name_x,feat_name_y,"Overlap"]
    for index, label in enumerate(category_label):
        plt.scatter([], [], c=colorlist[index], label=label)        
    if legend_fontsize is None: legend_fontsize=fig_size[1]*xsize*2
    plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=legend_fontsize)

    if save: fig.savefig(os.path.join(path,'_'.join(('Visium_J_top',str(top_n),
                                                    feat_name_x,feat_name_y))+save_name_add+'.png'), dpi=dpi)

    if return_axis: return axs
    else: plt.show()


def vis_all_connected_imageST(data, feat_name_x='', feat_name_y='',
                              dot_size=None, alpha = 0.8, vis_jaccard=True, jaccard_type='default',
                              fig_size=(5,5), title_fontsize = 20, legend_fontsize = None, 
                              title = 'Locations of', return_axis=False, axis = None,
                              save = False, path = os.getcwd(), save_name_add = '', dpi = 150):
    '''
    ## Visualizing all connected components x and y on tissue in image-based ST / Visium HD dataset
        -> Overlapping conected component locations in green, exclusive locations for x and y in red and blue, respectively
    ### Input  
    * data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
    * feat_name_x, feat_name_y: name of the feature x and y
    * dot_size: size of the spot visualized on the tissue
    * alpha: transparency of the colored spot
    * vis_jaccard: whether to visualize jaccard index on right corner of the plot
    * jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

    * fig_size: size of the dn figure
    * title_fontsize: size of the figure title, legend_fontsize: size of the legend text, title: title of the figure
    * return_axis: whether to return the plot axis
    * axis: matplotlib axes for plotting single image

    * save: whether to save of figure, path: saving path
    * save_name_add: additional name to be added in the end of the filename
    * dpi: dpi for image

    ### Outut
    * axs: matplotlib axis for the plot
    '''
    try: tsimg = data.obs.loc[:,['array_col','array_row']].to_numpy()
    except: raise ValueError("'data' should contain coordinates of spots in .obs as 'array_col' and 'array_row'")
    tsimg_row = tsimg[:,1]
    tsimg_col = tsimg[:,0]

    # Check the feasibility of the dataset
    if set(['Comb_CC_'+feat_name_x,'Comb_CC_'+feat_name_y]) <= set(data.obs.columns):
        cc_loc_x_df = data.obs['Comb_CC_'+feat_name_x].astype(int)
        cc_loc_y_df = data.obs['Comb_CC_'+feat_name_y].astype(int)
    else:
        raise ValueError("No CC location data for the given 'feat_x' and 'feat_y'")

    sc.set_figure_params(facecolor='white', frameon=False)
    if axis is None: fig, axs = plt.subplots(1,1, figsize=fig_size, tight_layout=True)
    else: axs = axis

    # Calculate overlapping locations
    data_mod = data.copy()
    data_mod.obs['Over'] = ((1 * ((cc_loc_x_df != 0) & (cc_loc_y_df == 0))) + \
                            (2 * ((cc_loc_x_df == 0) & (cc_loc_y_df != 0))) + \
                            (3 * ((cc_loc_x_df != 0) & (cc_loc_y_df != 0)))).astype('category')

    # Factorize the data and draw sccater plot
    factorize_data = pd.factorize(data_mod.obs['Over'], sort=True)
    
    # Define the colormap with three different colors: for CC locations of feature x, feature_y and intersecting regions
    colorlist = [["#A2E1CA","#FBBC05","#4285F4","#34A853"][index] for index in factorize_data[1]]
    cmap = colors.ListedColormap(colorlist)
    category_label = [["Others",feat_name_x,feat_name_y,"Overlap"][index] for index in factorize_data[1]]
    
    # Draw scatter plot
    if dot_size is None: dot_size = fig_size[0]/1.5
    axs.scatter(tsimg_col, tsimg_row, s = dot_size**2, c = factorize_data[0], 
                cmap = cmap, linewidth = 0, alpha=alpha, marker="s")

    for index, label in enumerate(category_label):
        axs.scatter([], [], c=colorlist[index], label=label)        
    if legend_fontsize is None: legend_fontsize=fig_size[1]*2
    axs.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=legend_fontsize)
    axs.axis('off')
    axs.set_title(feat_name_x+' & '+feat_name_y+title, fontsize = title_fontsize)

    if vis_jaccard: 
        Jcomp = jaccard_and_connected_loc_(data_mod, feat_name_x=feat_name_x, feat_name_y=feat_name_y, J_comp=True,
                                            jaccard_type=jaccard_type, return_mode='jaccard', return_sep_loc=False)
        axs.add_artist(offsetbox.AnchoredText(f'J_comp = {Jcomp:.3f}', loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=axs.transAxes,
                                              frameon=False, prop=dict(size = fig_size[1]*2.5)))

    if save:
      fig.savefig(os.path.join(path,'_'.join(('CosMx_loc_CCxy',feat_name_x,feat_name_y))+save_name_add+'.png'), dpi=dpi)
    
    if return_axis: return axs
    elif axis is None: plt.show()
