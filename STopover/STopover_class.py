import os
import scanpy as sc
from anndata import AnnData

from .cosmx_utils import *
from .topological_sim import topological_sim_pairs_
from .topological_comp import save_connected_loc_data_
from .jaccard import jaccard_and_connected_loc_
from .jaccard import jaccard_top_n_connected_loc_
from .topological_vis import *

import pkg_resources

class STopover_visium(AnnData):
    '''
    ## Class to calculate connected component location and jaccard similarity indices in visium dataset
    
    ### Input
    sp_adata: Anndata object for spatial transcriptomic data with count matrix ('raw') in .X
    sp_load_path: path to 10X-formatted Visium dataset directory or .h5ad Anndata object
    lognorm: whether to lognormalize (total count normalize and log transform) the count matrix saved in adata.X
    min_size: minimum size of a connected component
    fwhm: full width half maximum value for the gaussian smoothing kernel as the multiple of the central distance between the adjacent spots
    thres_per: lower percentile value threshold to remove the connected components
    save_path: path to save the data files
    J_count: number of jaccard similarity calculations after the first definition
    '''
    sp_adata: AnnData
    sp_load_path: str
    lognorm: bool
    min_size: int
    fwhm: float
    thres_per: float
    save_path: str
    J_count: int

    def __init__(self, sp_adata=None, sp_load_path='.', lognorm=False, min_size=20, fwhm=2.5, thres_per=30, save_path='.', J_count=0):
        assert min_size > 0
        assert fwhm > 0
        assert (thres_per >= 0) and (thres_per <= 100)

        # Load the Visium spatial transcriptomic data if no AnnData file was provided
        if sp_adata is None:
            print("Anndata object is not provided: searching for files in 'sp_load_path'")
            try: 
                adata_mod = sc.read_h5ad(sp_load_path)
                try: min_size, fwhm, thres_per = adata_mod.uns['min_size'], adata_mod.uns['fwhm'], adata_mod.uns['thres_per']
                except: pass
            except: 
                try: adata_mod = sc.read_visium(sp_load_path)
                except: raise ValueError("'sp_load_path': path to 10X-formatted Visium dataset directory or .h5ad Anndata object should be provided")
        else:
            adata_mod = sp_adata.copy()

        # Add the key parameters in the .uns
        adata_mod.uns['min_size'], adata_mod.uns['fwhm'], adata_mod.uns['thres_per'] = min_size, fwhm, thres_per
        # Save Jcount value
        try: J_result_num = [int(key_names.split("_")[2]) for key_names in adata_mod.uns.keys() if key_names.startswith("J_result_")]
        except: J_result_num = []
        if len(J_result_num) > 0: J_count = max(J_result_num) + 1
        # Make feature names unique
        adata_mod.var_names_make_unique()
        # Preserve raw .obs data in .uns
        if J_count==0: adata_mod.uns['obs_raw'] = adata_mod.obs

        # Preprocess the Visium spatial transcriptomic data
        if lognorm:
            if 'log1p' in adata_mod.uns.keys(): print("'adata' seems to be already log-transformed")
            sc.pp.normalize_total(adata_mod, target_sum=1e4, inplace=True)
            sc.pp.log1p(adata_mod)
        super(STopover_visium, self).__init__(X=adata_mod.X, obs=adata_mod.obs, var=adata_mod.var, uns=adata_mod.uns, obsm=adata_mod.obsm, raw=adata_mod.raw)

        self.min_size = min_size
        self.fwhm = fwhm
        self.thres_per = thres_per
        self.save_path = save_path
        self.J_count = J_count
        self.spatial_type = 'visium'
    

    def reinitalize(self, sp_adata, lognorm, min_size, fwhm, thres_per, save_path, J_count):
        '''
        ## Reinitialize the class
        '''
        self.__init__(sp_adata=sp_adata, lognorm=lognorm, min_size=min_size, fwhm=fwhm, thres_per=thres_per, save_path=save_path, J_count=J_count)


    def return_celltalkdb(self, lr_db_species='human'):
        '''
        ## Return CellTalkDB database as pandas dataframe

        ### Input
        lr_db_species: select species to utilize in CellTalkDB database

        ### Output
        CellTalkDB database as pandas dataframe
        '''
        assert lr_db_species in ['human', 'mouse'], "'lr_db_species' should be either 'human' or 'mouse'"
        lr_db = pkg_resources.resource_stream(__name__, 'data/CellTalkDB_'+lr_db_species+'_lr_pair.txt')
        feat_pairs = pd.read_csv(lr_db, delimiter='\t')
        return feat_pairs


    def topological_similarity(self, feat_pairs=None, use_lr_db=False, lr_db_species='human',
                                     group_name='batch', group_list=None, jaccard_type='default', J_result_name='result', 
                                     num_workers=os.cpu_count(), progress_bar=True):
        '''
        ## Calculate Jaccard index between topological connected components of feature pairs and return dataframe
            : if the group is given, divide the spatial data according to the group and calculate topological overlap separately in each group

        ### Input
        data: spatial data (format: anndata) containing log-normalized gene expression
        feat_pairs: 
            list of features with the format [('A','B'),('C','D')] or the pandas equivalent
            -> (A and C) should be same data format: all in metadata (.obs.columns) or all in gene names(.var.index)
            -> (C and D) should be same data format: all in metadata (.obs.columns) or all in gene names(.var.index)
            -> If the data format is not same the majority of the data format will be automatically searched
            -> and the rest of the features with different format will be removed from the pairs
        use_lr_db: whether to use list of features in CellTalkDB L-R database
        lr_db_species: select species to utilize in CellTalkDB database

        group_name: 
            the column name for the groups saved in metadata(.obs)
            spatial data is divided according to the group and calculate topological overlap separately in each group
        group_list: list of the elements in the group 
        jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

        J_result_name: the name of the jaccard index data file name
        num_workers: number of workers to use for multiprocessing
        progress_bar: whether to show the progress bar during multiprocessing

        ### Output
        df_top_total: dataframe that contains spatial overlap measures represented by (Jmax, Jmean, Jmmx, Jmmy) for the feature pairs 
        and average value for the feature across the spatial spots (if group is provided, then calculate average for the spots in each group)
        data_mod: AnnData with summed location of all connected components in metadata(.obs) across all feature pairs
        '''
        if use_lr_db:
            feat_pairs = self.return_celltalkdb(lr_db_species)
            feat_pairs = feat_pairs[['ligand_gene_symbol','receptor_gene_symbol']]
            print("Using CellTalkDB ligand-receptor dataset")
        
        df, adata = topological_sim_pairs_(data=self, feat_pairs=feat_pairs, spatial_type=self.spatial_type, group_list=group_list, group_name=group_name,
                                            fwhm=self.fwhm, min_size=self.min_size, thres_per=self.thres_per, jaccard_type=jaccard_type,
                                            num_workers=num_workers, progress_bar=progress_bar)
        # save jaccard index result in .uns of anndata
        adata.uns['_'.join(('J',str(J_result_name),str(self.J_count)))] = df
        # Initialize the object
        self.reinitalize(sp_adata=adata, lognorm=False, min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, save_path=self.save_path, J_count=self.J_count+1)
    

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
        self.reinitalize(sp_adata=adata, lognorm=False, min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, save_path=self.save_path, J_count=0)
    

    def jaccard_similarity_arr(self, feat_name_x="", feat_name_y="", jaccard_type='default', J_comp=False):
        '''
        ## Calculate jaccard index for connected components of feature x and y
        ### Input
        feat_name_x, feat_name_y: name of the feature x and y
        J_comp: whether to calculate Jaccard index Jcomp between CCx and CCy pair 
        jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

        ### Output
        if J_comp is True, then jaccard simliarity metrics calculated from jaccard similarity array between CCx and CCy (dim 0: CCx, dim 1: CCy)
        if J_comp is False, then return pairwise jaccard similarity array between CCx and CCy (dim 0: CCx, dim 1: CCy)
        '''
        J_result = jaccard_and_connected_loc_(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y, J_comp=J_comp, 
                                              jaccard_type=jaccard_type, return_mode='jaccard', return_sep_loc=False)
        return J_result


    def jaccard_top_n_connected_loc(self, feat_name_x='', feat_name_y='', top_n = 2, jaccard_type='default'):
        '''
        ## Calculate top n connected component locations for given feature pairs x and y
        ### Input
        feat_name_x, feat_name_y: name of the feature x and y
        top_n: the number of the top connected components to be found
        jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

        ### Output
        AnnData with intersecting location of top n connected components between feature x and y saved in metadata(.obs)
        -> top 1, 2, 3, ... intersecting connected component locations are separately saved
        '''
        adata, J_top_n = jaccard_top_n_connected_loc_(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y, top_n = top_n, jaccard_type=jaccard_type)
        adata.uns['_'.join(('J_top',feat_name_x, feat_name_y, str(top_n)))] = J_top_n
        # Initialize the object
        self.reinitalize(sp_adata=adata, lognorm=False, min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, save_path=self.save_path, J_count=self.J_count+1)


    def vis_jaccard_top_n_pair(self, feat_name_x='', feat_name_y='',
                               top_n = 2, jaccard_type='default', ncol = 2, spot_size=1, alpha_img=0.8, alpha = 0.8, 
                               fig_size = (10,10), batch_colname='batch', batch_name='0', batch_library_dict=None,
                               image_res = 'hires', adjust_image = True, border = 500, 
                               title_fontsize = 20, legend_fontsize = None, title = '', return_axis=False,
                               save = False, save_name_add = '', dpi=150):
        '''
        ## Visualizing top n connected component x and y showing maximum Jaccard index
        ### Input
        feat_name_x, feat_name_y: name of the feature x and y
        top_n: the number of the top connected component pairs withthe  highest Jaccard similarity index
        jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)
        ncol: number of columns to visualize top n CCs
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
        axis = vis_jaccard_top_n_pair_visium(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y,
                                             top_n=top_n, jaccard_type=jaccard_type, ncol = ncol, spot_size=spot_size, alpha_img=alpha_img, alpha=alpha, 
                                             fig_size=fig_size, batch_colname=batch_colname, batch_name=batch_name, batch_library_dict=batch_library_dict,
                                             image_res=image_res, adjust_image=adjust_image, border=border, 
                                             title_fontsize=title_fontsize, legend_fontsize = legend_fontsize, title=title, return_axis=return_axis,
                                             save = save, path = self.save_path, save_name_add = save_name_add, dpi=dpi)
        return axis
    

    def vis_all_connected(self, feat_name_x='', feat_name_y='',
                          spot_size=1, alpha_img=0.8, alpha = 0.8, vis_jaccard=True, jaccard_type='default',
                          fig_size=(10,10), batch_colname='batch', batch_name='0', batch_library_dict=None,
                          image_res = 'hires', adjust_image = True, border = 500, 
                          title_fontsize=20, legend_fontsize = None, title = '', return_axis=False, axis = None, 
                          save = False, save_name_add = '', dpi = 150):
        '''
        ## Visualizing all connected components x and y on tissue  
        ### Input  
        feat_name_x, feat_name_y: name of the feature x and y
        spot_size: size of the spot visualized on the tissue
        alpha_img: transparency of the tissue, alpha: transparency of the colored spot
        vis_jaccard: whether to visualize jaccard index on right corner of the plot
        jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

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
        axis: matplotlib axes for plotting single image

        save: whether to save of figure, path: saving path
        save_name_add: additional name to be added in the end of the filename
        dpi: dpi for image

        ### Outut
        axs: matplotlib axis for the plot
        '''
        axis = vis_all_connected_visium(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y,
                                        spot_size=spot_size, alpha_img = alpha_img, alpha = alpha, vis_jaccard = vis_jaccard, jaccard_type=jaccard_type,
                                        fig_size = fig_size, batch_colname=batch_colname, batch_name = batch_name, batch_library_dict=batch_library_dict,
                                        image_res = image_res, adjust_image = adjust_image, border = border, 
                                        title_fontsize=title_fontsize, legend_fontsize = legend_fontsize, title = title, return_axis=return_axis, axis = axis,
                                        save = save, path = self.save_path, save_name_add = save_name_add, dpi = dpi)
        return axis



class STopover_cosmx(STopover_visium):
    '''
    ## Class to calculate connected component location and jaccard similarity indices in CosMx dataset
    
    ### Input
    sp_adata: Anndata object for CosMx SMI data with count matrix ('raw') in .X
    sp_load_path: path to CosMx SMI data directory or .h5ad Anndata object

    sc_adata: single-cell reference anndata for cell type annotation of CosMx SMI data
        -> raw count matrix should be saved in .X
        -> If .h5ad file directory is provided, it will load the h5ad file
        -> If None, then leiden cluster numbers will be used to annotate CosMx SMI data
    sc_celltype_colname: column name for cell type annotation information in metadata of single-cell (.obs)
    sc_norm_total: scaling factor for the total count normalization per cell

    tx_file_name, cell_exprmat_file_name, cell_metadata_file_name: CosMx file for transcript count, cell-level expression matrix, cell-level metadata
    fov_colname, cell_id_colname: column name for barcodes corresponding to fov and cell ID
    tx_xcoord_colname, tx_ycoord_colname, transcript_colname: column name for global x, y coordinates of the transcript and transcript name
    meta_xcoord_colname, meta_ycoord_colname: column name for global x, y coordinates in cell-level metadata file
    x_bins, y_bins: number of bins to divide the CosMx SMI data (for grid-based aggregation)

    min_size: minimum size of a connected component
    fwhm: full width half maximum value for the gaussian smoothing kernel as the multiple of the central distance between the adjacent grid
    thres_per: lower percentile value threshold to remove the connected components
    save_path: path to save the data files
    J_count: number of jaccard similarity calculations after the first definition
    '''
    sp_adata: AnnData
    sp_load_path: str
    x_bins: int
    y_bins: int
    min_size: int
    fwhm: float
    thres_per: float
    save_path: str
    J_count: int

    def __init__(self, sp_adata=None, sp_load_path='.', sc_adata=None, sc_celltype_colname = 'celltype', sc_norm_total=1e3,
                 tx_file_name = 'tx_file.csv', cell_exprmat_file_name='exprMat_file.csv', cell_metadata_file_name='metadata_file.csv', 
                 fov_colname = 'fov', cell_id_colname='cell_ID', tx_xcoord_colname='x_global_px', tx_ycoord_colname='y_global_px', transcript_colname='target',
                 meta_xcoord_colname='CenterX_global_px', meta_ycoord_colname='CenterY_global_px',
                 x_bins=100, y_bins=100, 
                 min_size=20, fwhm=2.5, thres_per=30, save_path='.', J_count=0):

        assert (x_bins > 0) and (y_bins > 0)
        assert min_size > 0
        assert fwhm > 0
        assert (thres_per >= 0) and (thres_per <= 100)

        # Load the CosMx spatial transcriptomic data if no AnnData file was provided
        if sp_adata is None:
            print("Anndata object is not provided: searching for files in 'sp_load_path'")
            try: 
                adata_mod = sc.read_h5ad(sp_load_path)
                try: min_size, fwhm, thres_per, x_bins, y_bins, sc_norm_total, sc_celltype_colname, transcript_colname = \
                    adata_mod.uns['min_size'], adata_mod.uns['fwhm'], adata_mod.uns['thres_per'], adata_mod.uns['x_bins'], \
                        adata_mod.uns['y_bins'], adata_mod.uns['sc_norm_total'], adata_mod.uns['sc_celltype_colname'], adata_mod.uns['transcript_colname']
                except: pass
                # Save Jcount value
                J_result_num = [int(key_names.split("_")[2]) for key_names in adata_mod.uns.keys() if key_names.startswith("J_result_")]
                if len(J_result_num) > 0: J_count = max(J_result_num) + 1
            except:
                if isinstance(sc_adata, str):
                    try: sc_adata = sc.read_h5ad(sc_adata)
                    except: 
                        print("Path to 'sc_adata' h5ad file not found: replacing with None")
                        sc_adata = None
                adata_mod, adata_cell = read_cosmx(sp_load_path, sc_adata=sc_adata, sc_celltype_colname=sc_celltype_colname, sc_norm_total=sc_norm_total,
                                                   tx_file_name = tx_file_name, cell_exprmat_file_name=cell_exprmat_file_name, cell_metadata_file_name=cell_metadata_file_name, 
                                                   fov_colname = fov_colname, cell_id_colname=cell_id_colname, 
                                                   tx_xcoord_colname=tx_xcoord_colname, tx_ycoord_colname=tx_ycoord_colname, transcript_colname=transcript_colname,
                                                   meta_xcoord_colname=meta_xcoord_colname, meta_ycoord_colname=meta_ycoord_colname,
                                                   x_bins=x_bins, y_bins=y_bins)
                adata_mod.uns['adata_cell'] = adata_cell
        else:
            adata_mod = sp_adata.copy()
        # Make feature names unique
        adata_mod.var_names_make_unique()

        adata_mod.uns['x_bins'] = x_bins
        adata_mod.uns['y_bins'] = y_bins
        adata_mod.uns['sc_norm_total'] = sc_norm_total
        adata_mod.uns['sc_celltype_colname'] = sc_celltype_colname
        adata_mod.uns['transcript_colname'] = transcript_colname
    
        # Generate object with the help of STopover_visium
        super(STopover_cosmx, self).__init__(sp_adata=adata_mod, lognorm=False, min_size=min_size, fwhm=fwhm, thres_per=thres_per, save_path=save_path, J_count=J_count)

        self.x_bins = x_bins
        self.y_bins = y_bins
        self.sc_celltype_colname = sc_celltype_colname
        self.transcript_colname = transcript_colname
        self.sc_norm_total = sc_norm_total
        self.spatial_type = 'cosmx'


    def reinitalize(self,sp_adata, lognorm=False, sc_celltype_colname=None, sc_norm_total=None, x_bins=None, y_bins=None, 
                    min_size=None, fwhm=None, thres_per=None, save_path=None, J_count=None, inplace=True):
        '''
        ## Reinitialize the class
        '''
        if (sc_celltype_colname is None) or (sc_norm_total is None) or (x_bins is None) or (y_bins is None):
            sc_celltype_colname = self.sc_celltype_colname
            sc_norm_total = self.sc_norm_total
            x_bins = self.x_bins
            y_bins = self.y_bins
        if inplace:
            self.__init__(sp_adata=sp_adata, sc_celltype_colname=sc_celltype_colname, sc_norm_total=sc_norm_total, 
                          x_bins=x_bins, y_bins=y_bins, min_size=min_size, fwhm=fwhm, thres_per=thres_per, save_path=save_path, J_count=J_count)
        else:
            sp_adata_mod = STopover_cosmx(sp_adata, sc_celltype_colname=sc_celltype_colname, sc_norm_total=sc_norm_total, 
                                          x_bins=x_bins, y_bins=y_bins, min_size=min_size, fwhm=fwhm, thres_per=thres_per, 
                                          save_path=save_path, J_count=J_count)
            return sp_adata_mod


    def celltype_specific_adata(self, cell_types=['']):
        '''
        ## Replace count matrix saved in .X with cell type specific transcript count matrix
        ### Input
        cell_types: the cell types to extract cell type-specific count information

        ### Output
        grid_tx_count_celltype: list of celltype specific grid-based count matrix as sparse.csr_matrix format
        '''
        grid_count_celltype_list = celltype_specific_mat(sp_adata=self, tx_info_name='tx_by_cell_grid', celltype_colname=self.sc_celltype_colname, 
                                                         cell_types=cell_types, transcript_colname=self.transcript_colname, sc_norm_total=self.sc_norm_total)
        grid_count_celltype_list = [STopover_cosmx(celltype_stopover, sc_celltype_colname=self.sc_celltype_colname, 
                                    sc_norm_total=self.sc_norm_total, x_bins=self.x_bins, y_bins=self.y_bins, 
                                    min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, save_path=self.save_path) for celltype_stopover in grid_count_celltype_list]                                                
        return grid_count_celltype_list


    def topological_similarity_celltype_pair(self, celltype_x='', celltype_y='', feat_pairs=None, use_lr_db=False, lr_db_species='human',
                                             group_name='batch', group_list=None, J_result_name='result', num_workers=os.cpu_count(), progress_bar=True):
        '''
        ## Calculate Jaccard index between the two cell type-specific expression anndata of CosMx data
        ### Input
        celltype_x: name of the cell type x (should be among the column names of .obs)
        celltype_y: name of the cell type y (should be among the column names of .obs)
            when use_lr_db=True, then the ligand expression in celltype x and receptor expression in celltype y will be searched
        other parameters: refer to the topological_similarity method
        '''
        adata_x, adata_y = self.celltype_specific_adata(cell_types=[celltype_x, celltype_y])
        # Create combined anndata for two cell type specific count matrices
        comb_var_names = (celltype_x+': '+adata_x.var_names).tolist() + (celltype_y+': '+adata_y.var_names).tolist()
        adata_xy = AnnData(X=sparse.hstack([adata_x.X, adata_y.X]).tocsr(), obs=adata_x.obs)
        adata_xy.var_names = comb_var_names
        adata_xy = STopover_cosmx(adata_xy, sc_celltype_colname=self.sc_celltype_colname, sc_norm_total=self.sc_norm_total, 
                                  x_bins=self.x_bins, y_bins=self.y_bins, min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, 
                                  save_path=self.save_path, J_count=self.J_count)
        if use_lr_db:
            feat_pairs = self.return_celltalkdb(lr_db_species)
            feat_pairs = feat_pairs[['ligand_gene_symbol','receptor_gene_symbol']]
            use_lr_db = False
            print("Calculating topological similarity between genes in '%s' and '%s'" % (celltype_x, celltype_y))
            print("Using CellTalkDB ligand-receptor dataset")
        else: 
            if isinstance(feat_pairs, list): feat_pairs = pd.DataFrame(feat_pairs)

        # Modify the column name of the feature pair dataframe
        celltype_list = [celltype_x, celltype_y]
        for index, colname in enumerate(feat_pairs.columns):
            feat_pairs[colname] = celltype_list[index]+': '+feat_pairs[colname]
        
        # Calculate topological similarites between the pairs from the two cell types  
        adata_xy.topological_similarity(feat_pairs=feat_pairs, use_lr_db=use_lr_db, lr_db_species=lr_db_species,
                                        group_name=group_name, group_list=group_list, J_result_name=J_result_name, num_workers=num_workers, progress_bar=progress_bar)
        return adata_xy


    def vis_spatial_cosmx(self, feat_name='', colorlist = None, dot_size=None, alpha = 0.8, vmax = None, vmin = None, sort_labels=True,
                          fig_size = (10,10), title_fontsize = 20, legend_fontsize = None, title = None, 
                          return_axis=False, figure = None, axis = None, save = False, save_name_add = '', dpi=150):
        '''
        ## Visualizing spatial distribution of features in CosMx dataset
        ### Input
        data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
        feat_name: name of the feature to visualize
        colorlist: color list for the visualization of CC identity
        dot_size: size of the spot visualized on the tissue
        alpha: transparency of the colored spot
        vmax: maximum value in the colorbar; if None, it will automatically set the maximum value
        vmax: minimum value in the colorbar; if None, it will automatically set the minimum value
        sort_labels: sort the category labels in alphanumeric order if the name of categorical feature is provided to 'feat_name'

        fig_size: size of the drawn figure
        title_fontsize: size of the figure title, legend_fontsize: size of the legend text, title: title of the figure
        return_axis: whether to return the plot axis
        figure: matplotlib figure for plotting single image, axis: matplotlib axes for plotting single image

        save: whether to save of figure, path: saving path
        save_name_add: additional name to be added in the end of the filename
        dpi: dpi for image

        ### Outut
        axs: matplotlib axis for the plot
        '''
        axis = vis_spatial_cosmx_(data=self, feat_name=feat_name, colorlist = colorlist, dot_size=dot_size, alpha = alpha, vmax=vmax, vmin=vmin, sort_labels=sort_labels,
                                  fig_size = fig_size, title_fontsize = title_fontsize, legend_fontsize = legend_fontsize, title = title, 
                                  return_axis=return_axis, figure=figure, axis = axis, save = save, path = self.save_path, save_name_add = save_name_add, dpi=dpi)
        return axis


    def vis_jaccard_top_n_pair(self, feat_name_x='', feat_name_y='',
                               top_n = 2, jaccard_type='default', ncol = 2, dot_size=None, alpha = 0.8, 
                               fig_size = (10,10), title_fontsize = 20, legend_fontsize = None,
                               title = '', return_axis=False,
                               save = False, save_name_add = '', dpi=150):
        '''
        ## Visualizing top n connected component x and y showing maximum Jaccard index in CosMx dataset
        -> Overlapping conected component locations in green, exclusive locations for x and y in red and blue, respectively
        ### Input
        data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
        feat_name_x, feat_name_y: name of the feature x and y
        top_n: the number of the top connected component pairs withthe  highest Jaccard similarity index
        jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)
        ncol: number of columns to visualize top n CCs
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
        axis = vis_jaccard_top_n_pair_cosmx(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y,
                                             top_n = top_n, jaccard_type=jaccard_type, ncol = ncol, dot_size= dot_size, alpha = alpha, 
                                             fig_size = fig_size, title_fontsize = title_fontsize, legend_fontsize = legend_fontsize,
                                             title = title, return_axis=return_axis,
                                             save = save, path = self.save_path, save_name_add = save_name_add, dpi=dpi)
        return axis
    

    def vis_all_connected(self, feat_name_x='', feat_name_y='',
                          dot_size=None, alpha = 0.8, vis_jaccard=True, jaccard_type='default', 
                          fig_size=(10,10), title_fontsize = 20, legend_fontsize = None, 
                          title = '', return_axis = False, axis = None,
                          save = False, save_name_add = '', dpi = 150):
        '''
        ## Visualizing all connected components x and y on tissue in CosMx dataset
        -> Overlapping conected component locations in green, exclusive locations for x and y in red and blue, respectively
        ### Input  
        data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
        feat_name_x, feat_name_y: name of the feature x and y
        dot_size: size of the spot visualized on the tissue
        alpha: transparency of the colored spot
        vis_jaccard: whether to visualize jaccard index on right corner of the plot
        jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

        fig_size: size of the drawn figure
        title_fontsize: size of the figure title, legend_fontsize: size of the legend text, title: title of the figure
        return_axis: whether to return the plot axis
        axis: matplotlib axes for plotting single image

        save: whether to save of figure, path: saving path
        save_name_add: additional name to be added in the end of the filename
        dpi: dpi for image

        ### Outut
        axs: matplotlib axis for the plot
        '''
        axis = vis_all_connected_cosmx(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y,
                                       dot_size=dot_size, alpha = alpha, vis_jaccard = vis_jaccard, jaccard_type=jaccard_type, 
                                       fig_size= fig_size, title_fontsize = title_fontsize, legend_fontsize = legend_fontsize, 
                                       title = title, return_axis = return_axis, axis = axis,
                                       save = save, path = self.save_path, save_name_add = save_name_add, dpi = dpi)
        return axis
