import scanpy as sc
from anndata import AnnData

from .topological_sim import topological_sim_multi_pairs_
from .topological_comp import save_connected_loc_data_
from .jaccard import jaccard_and_connected_loc_
from .jaccard import jaccard_top_n_connected_loc_
from .topological_vis import vis_all_connected_
from .topological_vis import vis_jaccard_top_n_pair_



class STopover(AnnData):
    '''
    ## Class to calculate connected component location and jaccard similarity indices
    
    ### Input
    adata: Anndata object for spatial transcriptomic data with count matrix ('raw') in .X
    adata_format: format of the given count matrix in adata.X ('log' for log-normalized count and 'raw' for raw count matrix)
    load_path: path to 10X-formatted Visium dataset directory or .h5ad Anndata object
    fwhm: full width half maximum value for the gaussian smoothing kernal
    min_size: minimum size of a connected component
    thres_per: lower percentile value threshold to remove the connected components
    save_path: path to save the data files
    J_count: number of jaccard similarity calculations after the first definition
    '''
    adata: AnnData
    load_path: str
    adata_format: str
    min_size: int
    fwhm: float
    thres_per: float
    save_path: str
    J_count: int

    def __init__(self, adata=None, load_path='.', adata_format = 'log', min_size=20, fwhm=2.5, thres_per=30, save_path='.', J_count=0):
        assert min_size > 0
        assert fwhm > 0
        assert (thres_per >= 0) and (thres_per <= 100)

        # Load the Visium spatial transcriptomic data if no AnnData file was provided
        if adata is None:
            print("Anndata object is not provided: searching for files in 'load_path'")
            try: 
                adata_mod = sc.read_h5ad(load_path)
                try: min_size, fwhm, thres_per = adata_mod.uns['min_size'], adata_mod.uns['fwhm'], adata_mod.uns['thres_per']
                except: adata_mod.uns['min_size'], adata_mod.uns['fwhm'], adata_mod.uns['thres_per'] = min_size, fwhm, thres_per
            except: 
                try:
                    adata_mod = sc.read_visium(load_path)
                    adata_mod.uns['min_size'], adata_mod.uns['fwhm'], adata_mod.uns['thres_per'] = min_size, fwhm, thres_per
                except: raise ValueError("'load_path': path to 10X-formatted Visium dataset directory or .h5ad Anndata object should be provided")
        else:
            adata_mod = adata.copy()
            # Add the key parameters in the .uns
            adata_mod.uns['min_size'], adata_mod.uns['fwhm'], adata_mod.uns['thres_per'] = min_size, fwhm, thres_per
        # Preserve raw .obs data in .uns
        if J_count==0: adata_mod.uns['obs_raw'] = adata_mod.obs

        # Preprocess the Visium spatial transcriptomic data
        if adata_format == 'raw':
            sc.pp.normalize_total(adata_mod, target_sum=1e4, inplace=True)
            sc.pp.log1p(adata_mod)
            super(STopover, self).__init__(X=adata_mod.X, obs=adata_mod.obs, var=adata_mod.var, uns=adata_mod.uns, obsm=adata_mod.obsm)
        elif adata_format == 'log':
            super(STopover, self).__init__(X=adata_mod.X, obs=adata_mod.obs, var=adata_mod.var, uns=adata_mod.uns, obsm=adata_mod.obsm)
        else:
            raise ValueError("'adata_format' should be either 'raw' or 'log'")

        self.adata_format = adata_format
        self.min_size = min_size
        self.fwhm = fwhm
        self.thres_per = thres_per
        self.save_path = save_path
        self.J_count = J_count


    def topological_similarity(self, feat_pairs, group_name='batch', group_list=None, J_result_name='result'):
        '''
        ## Calculate Jaccard index for given feature pairs and return dataframe
            : if the group is given, divide the spatial data according to the group and calculate topological overlap separately in each group

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
        group_list: list of the elements in the group 

        J_result_name: the name of the jaccard index data file name

        ### Output
        df_top_total: dataframe that contains spatial overlap measures represented by (Jmax, Jmean, Jmmx, Jmmy) for the feature pairs 
        and average value for the feature across the spatial spots (if group is provided, then calculate average for the spots in each group)
        data_mod: AnnData with summed location of all connected components in metadata(.obs) across all feature pairs
        '''
        df, adata = topological_sim_multi_pairs_(data=self, feat_pairs=feat_pairs, group_list=group_list, group_name=group_name,
                                                fwhm=self.fwhm, min_size=self.min_size, thres_per=self.thres_per)
        # save jaccard index result in .uns of anndata
        adata.uns['_'.join(('J',str(J_result_name),str(self.J_count)))] = df
        # Initialize the object
        self.__init__(adata=adata, adata_format='log', min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, save_path=self.save_path, J_count=self.J_count+1)
    

    def save_connected_loc_data(self, save_format='h5ad', filename = 'cc_location'):
        '''
        ## Save the anndata or metadata file to the certain location
        ### Input
        data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
        save_format: format to save the location of connected components; either 'h5ad' or 'csv'
        file_name: file name to save (default: cc_location)

        ### Output: None
        '''
        save_connected_loc_data_(data=self, save_format=save_format, path=self.save_path, filename=filename)

    
    def J_result_reset(self):
        '''
        ## Remove the results for jaccard similarity and connected component location and reset
        '''
        adata = self
        # Move the raw .obs file
        adata.obs = adata.uns['obs_raw']
        # Remove the J_result data saved in .uns
        import re
        pattern = re.compile("^J_.*_[0-9]$")
        adata_keys = list(adata.uns.keys())
        for J_result_name in adata_keys:
            if pattern.match(J_result_name): del adata.uns[J_result_name]
        # Initialize the object
        self.__init__(adata=adata, adata_format='log', min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, save_path=self.save_path, J_count=0)
    

    def jaccard_similarity(self, feat_name_x="", feat_name_y="", J_metric=False):
        '''
        ## Calculate jaccard index for connected components
        ### Input
        feat_name_x, feat_name_y: name of the feature x and y
        J_metric: whether to calculate Jaccard index (Jmax, Jmean, Jmmx, Jmmy) between CCx and CCy pair 

        ### Output
        if J_metric is True, then jaccard simliarity metrics calculated from jaccard similarity array between CCx and CCy (dim 0: CCx, dim 1: CCy)
            (Jmax, Jmean, Jmmx, Jmmy): maximum jaccard index, mean jaccard index and mean of maximum jaccard for CCx and CCy
        if J_metric is False, then return jaccard similarity array between CCx and CCy (dim 0: CCx, dim 1: CCy)
        '''
        J_result = jaccard_and_connected_loc_(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y, J_metric=J_metric, 
                                              return_mode='jaccard', return_sep_loc=False)
        return J_result


    def jaccard_top_n_connected_loc(self, feat_name_x='', feat_name_y='', top_n = 5):
        '''
        ## Calculate top n connected component locations for given feature pairs x and y
        ### Input
        feat_name_x, feat_name_y: name of the feature x and y
        top_n: the number of the top connected components to be found

        ### Output
        AnnData with intersecting location of top n connected components between feature x and y saved in metadata(.obs)
        -> top 1, 2, 3, ... intersecting connected component locations are separately saved
        '''
        adata = jaccard_top_n_connected_loc_(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y, top_n = top_n)
        # Initialize the object
        self.__init__(adata=adata, adata_format='log', min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, save_path=self.save_path, J_count=self.J_count+1)


    def vis_jaccard_top_n_pair(self, top_n = 5, cmap='tab20', spot_size=1,
                               alpha_img=0.8, alpha = 0.8, feat_name_x='', feat_name_y='',
                               fig_size = (10,10), batch_colname='batch', batch_name='0', batch_library_dict=None,
                               image_res = 'hires', adjust_image = True, border = 50, 
                               fontsize = 30, title = 'J', return_axis=False,
                               save = False, save_name_add = '', dpi=300):
        '''
        ## Visualizing top n connected component x and y showing maximum Jaccard index
        ### Input
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
        axis = vis_jaccard_top_n_pair_(data=self, top_n=top_n, cmap=cmap, spot_size=spot_size,
                                       alpha_img=alpha_img, alpha=alpha, feat_name_x=feat_name_x, feat_name_y=feat_name_y,
                                       fig_size=fig_size, batch_colname=batch_colname, batch_name=batch_name, batch_library_dict=batch_library_dict,
                                       image_res=image_res, adjust_image=adjust_image, border=border, 
                                       fontsize=fontsize, title=title, return_axis=return_axis,
                                       save = save, path = self.save_path, save_name_add = save_name_add, dpi=dpi)
        return axis
    

    def vis_all_connected(self, vis_intersect_only = False, cmap='tab20', spot_size=1, 
                          alpha_img=0.8, alpha = 0.8, feat_name_x='', feat_name_y='',
                          fig_size=(20,10), batch_colname='batch', batch_name='0', batch_library_dict=None,
                          image_res = 'hires', adjust_image = True, border = 50, 
                          fontsize=30, title = 'Locations of', return_axis=False,
                          save = False, save_name_add = '', dpi = 300):
        '''
        ## Visualizing all connected components x and y on tissue  
        ### Input  
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
        axis = vis_all_connected_(data=self, vis_intersect_only = vis_intersect_only, cmap=cmap, spot_size=spot_size, 
                                  alpha_img = alpha_img, alpha = alpha, feat_name_x=feat_name_x, feat_name_y=feat_name_y,
                                  fig_size = fig_size, batch_colname=batch_colname, batch_name = batch_name, batch_library_dict=batch_library_dict,
                                  image_res = image_res, adjust_image = adjust_image, border = border, 
                                  fontsize = fontsize, title = title, return_axis=return_axis,
                                  save = save, path = self.save_path, save_name_add = save_name_add, dpi = dpi)
        return axis