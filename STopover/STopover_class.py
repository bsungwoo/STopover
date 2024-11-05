import os
import copy
import scanpy as sc
from anndata import AnnData

from .imageST_utils import *
from .topological_sim import topological_sim_pairs_
from .permutation import run_permutation_test
from .topological_comp import save_connected_loc_data_
from .jaccard import jaccard_and_connected_loc_
from .jaccard import jaccard_top_n_connected_loc_
from .topological_vis import *

import pkg_resources

class STopover_visium(AnnData):
    '''
    ## Class to calculate connected component location and jaccard similarity indices in visium dataset
    
    ### Input
    * sp_adata: Anndata object for Visium spatial transcriptomic data  
    * sp_load_path: path to 10X-formatted Visium dataset directory or .h5ad Anndata object  
    * lognorm: whether to lognormalize (total count normalize and log transform) the count matrix saved in adata.X  
    * min_size: minimum size of a connected component  
    * fwhm: full width half maximum value for the gaussian smoothing kernel as the multiple of the central distance between the adjacent spots  
    * thres_per: lower percentile value threshold to remove the connected components  
    * save_path: path to save the data files  
    * J_count: number of jaccard similarity calculations after the first definition  
    '''
    def __init__(self, sp_adata: AnnData = None, sp_load_path: str = '.', 
                 lognorm: bool = False, min_size: int = 20, fwhm: float = 2.5, thres_per: float = 30, 
                 save_path: str = '.', J_count: int = 0):
        assert min_size > 0
        assert fwhm > 0
        assert (thres_per >= 0) and (thres_per <= 100)

        # Load the Visium spatial transcriptomic data if no AnnData file was provided
        if sp_adata is None:
            try: 
                print("Anndata object is not provided: searching for the .h5ad file in 'sp_load_path'")
                adata_mod = sc.read_h5ad(sp_load_path)
                try: min_size, fwhm, thres_per = adata_mod.uns['min_size'], adata_mod.uns['fwhm'], adata_mod.uns['thres_per']
                except: pass
            except:
                print("Failed\nReading Visium data files in 'sp_load_path'")
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
        # Preserve  .obs data in .uns
        if J_count==0: adata_mod.uns['obs_raw'] = adata_mod.obs

        # Preprocess the Visium spatial transcriptomic data
        if lognorm:
            if 'log1p' in adata_mod.uns.keys(): print("'adata' seems to be already log-transformed")
            sc.pp.normalize_total(adata_mod, target_sum=1e4, inplace=True)
            sc.pp.log1p(adata_mod)
        super(STopover_visium, self).__init__(X=adata_mod.X, obs=adata_mod.obs, var=adata_mod.var, uns=adata_mod.uns, obsm=adata_mod.obsm, layers=adata_mod.layers, raw=adata_mod.raw)
        
        # Create directory to save
        os.makedirs(save_path, exist_ok=True)

        self.min_size = min_size
        self.fwhm = fwhm
        self.thres_per = thres_per
        self.save_path = save_path
        self.J_count = J_count
        self.spatial_type = 'visium'

    def __getitem__(self, index):
        """
        Overrides the __getitem__ method to ensure that subsetting returns an instance of STopover_visium.
        """
        subset = super().__getitem__(index)
        return STopover_visium(
            sp_adata=subset,
            min_size=self.min_size,
            fwhm=self.fwhm,
            thres_per=self.thres_per,
            save_path=self.save_path,
            J_count=self.J_count
        )

    def __repr__(self):
        if self.is_view:
            return "View of " + self._gen_repr(self.n_obs, self.n_vars).replace("AnnData object", "STopover_visium object")
        else:
            return self._gen_repr(self.n_obs, self.n_vars).replace("AnnData object", "STopover_visium object")

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        # Create a new instance of the class
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result       
        # Copy all attributes
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def reinitalize(self, sp_adata, lognorm, min_size, fwhm, thres_per, save_path, J_count, inplace=True):
        '''
        ## Reinitialize the class
        '''
        if inplace:
            self.__init__(sp_adata=sp_adata, lognorm=lognorm, min_size=min_size, fwhm=fwhm, thres_per=thres_per, save_path=save_path, J_count=J_count)
        else:
            sp_adata_re = STopover_visium(sp_adata=sp_adata, lognorm=lognorm, min_size=min_size, fwhm=fwhm, thres_per=thres_per, save_path=save_path, J_count=J_count)
            return sp_adata_re

    def return_lr_db(self, lr_db_species='human', db_name='CellTalk'):
        '''
        ## Return ligand-receptor database as pandas dataframe
            -> CellTalk, CellChat or Omnipath databases can be extracted

        ### Input
        lr_db_species: select species to utilize in CellTalk, CellChat or Omnipath databases
            -> Either 'human', 'mouse', or, 'rat' (only in Omnipath)

        ### Output
        CellTalk, CellChat or Omnipath database as pandas dataframe
        '''
        if lr_db_species not in ['human','mouse','rat']: 
            raise ValueError("'lr_db_species' should be either 'human', 'mouse', or, 'rat'")
        if db_name in ['CellTalk','CellChat']:
            if lr_db_species=="rat": raise NotImplementedError("'lr_db_species' can only be 'human' or 'mouse'")
        elif db_name != "Omnipath": raise ValueError("'db_name should be either 'Omnipath', 'CellTalk' or 'CellChat'")

        if db_name=="Omnipath":
            lr_db = pkg_resources.resource_stream(__name__, 'data/interaction_input_Omnipath_'+lr_db_species+'.csv')
            feat_pairs = pd.read_csv(lr_db)
        elif db_name=="CellTalk": 
            lr_db = pkg_resources.resource_stream(__name__, 'data/CellTalkDB_'+lr_db_species+'_lr_pair.txt')
            feat_pairs = pd.read_csv(lr_db, delimiter='\t')
        elif db_name=="CellChat":
            lr_db = pkg_resources.resource_stream(__name__, 'data/interaction_input_CellChatDB_'+lr_db_species+'.csv')
            feat_pairs = pd.read_csv(lr_db)
        return feat_pairs


    def topological_similarity(self, feat_pairs=None, use_lr_db=False, lr_db_species='human', db_name='CellTalk',
                               group_name='batch', group_list=None, jaccard_type='default', J_result_name='result', 
                               num_workers=os.cpu_count(), progress_bar=True):
        if use_lr_db:
            feat_pairs = self.return_lr_db(lr_db_species=lr_db_species, db_name=db_name)
            if db_name=="Omnipath": 
                feat_pairs = feat_pairs[['source_genesymbol','target_genesymbol']]
                feat_pairs['source_genesymbol'] = feat_pairs['source_genesymbol'].str.split('_')
                feat_pairs['target_genesymbol'] = feat_pairs['target_genesymbol'].str.split('_')
                feat_pairs = feat_pairs.explode('source_genesymbol', ignore_index=True)
                feat_pairs = feat_pairs.explode('target_genesymbol', ignore_index=True)
                feat_pairs = feat_pairs.drop_duplicates(subset = ['source_genesymbol', 'target_genesymbol'], 
                                                        keep = 'first').reset_index(drop = True)
            elif db_name=="CellTalk": feat_pairs = feat_pairs[['ligand_gene_symbol','receptor_gene_symbol']]
            elif db_name=="CellChat":
                # Modify the dataframe to contain only the ligand and receptor pairs
                df = feat_pairs.assign(
                ligand_gene_symbol = lambda dataframe: dataframe['interaction_name_2'].map(lambda x: x.replace(" ","").split("-")[0]),
                receptor = lambda dataframe: dataframe['interaction_name_2'].map(lambda x: x.replace(" ","").split("-")[1]),
                receptor1 = lambda dataframe: dataframe['receptor'].map(lambda x: x.split("+")[0][1:] if "(" in x else x),
                receptor2 = lambda dataframe: dataframe['receptor'].map(lambda x: x.split("+")[1][:-1] if ")" in x else None)
                )
                feat_pairs = pd.concat([df.loc[:,['ligand_gene_symbol','receptor1']].rename(columns={'receptor1':'receptor_gene_symbol'}),
                                        df[df['receptor2'].notna()].loc[:,['ligand_gene_symbol','receptor2']].rename(columns={'receptor2':'receptor_gene_symbol'})], 
                                        axis = 0).reset_index(drop=True)
            print("Using "+db_name+" ligand-receptor dataset")
        
        df, adata = topological_sim_pairs_(data=self, feat_pairs=feat_pairs, spatial_type=self.spatial_type, group_list=group_list, group_name=group_name,
                                            fwhm=self.fwhm, min_size=self.min_size, thres_per=self.thres_per, jaccard_type=jaccard_type,
                                            num_workers=num_workers, progress_bar=progress_bar)
        # save jaccard index result in .uns of anndata
        adata.uns['_'.join(('J',str(J_result_name),str(self.J_count)))] = df
        # Initialize the object
        self.reinitalize(sp_adata=adata, lognorm=False, min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, save_path=self.save_path, J_count=self.J_count+1)

      
    def run_significance_test(self, feat_pairs_sig_test=None, nperm=1000, seed=0, 
                              jaccard_type='default', num_workers=os.cpu_count(), progress_bar=True):
        '''
        ## Perform a significant test using a permutation test and calculate p-values
        * feat_pairs_sig_test: feature pairs for the significance test (default: None -> Test all saved in .uns)
        * nperm: number of the random permutation (default: 1000)
        * seed: the seed for the random number generator (default: 0)
        * jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)
        * num_workers: number of workers to use for multiprocessing
        * progress_bar: whether to show the progress bar during multiprocessing
        '''
        print("Run permutation test for the given LR pairs")
        import re
        pattern = re.compile("^J_.*_[0-9]$")
        adata_keys = sorted([i for i in self.uns.keys() if pattern.match(i)])
        
        df, adata = run_permutation_test(self, feat_pairs_sig_test, nperm=nperm, seed=seed, spatial_type = self.spatial_type,
                                         fwhm=self.fwhm, min_size=self.min_size, thres_per=self.thres_per, jaccard_type=jaccard_type,
                                         num_workers=num_workers, progress_bar=progress_bar)
        
        # save jaccard index result in .uns of anndata
        adata.uns['_'.join((adata_keys[-1], 'sig'))] = df
        # Initialize the object
        self.reinitalize(sp_adata=adata, lognorm=False, min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, save_path=self.save_path, J_count=self.J_count)     


    def save_connected_loc_data(self, save_format='h5ad', filename = 'cc_location'):
        '''
        ## Save the anndata or metadata file to the certain location
        ### Input
        * data: AnnData with summed location of all connected components in metadata(.obs) across feature pairs
        * save_format: format to save the location of connected components; either 'h5ad' or 'csv'
        * file_name: file name to save (default: cc_location)
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
        pattern = re.compile(r"^J_.*_\d+(_sig)?$")
        adata_keys = list(adata.uns.keys())
        for J_result_name in adata_keys:
            if pattern.match(J_result_name): del adata.uns[J_result_name]
        # Initialize the object
        self.reinitalize(sp_adata=adata, lognorm=False, min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, save_path=self.save_path, J_count=0)
    

    def jaccard_similarity_arr(self, feat_name_x="", feat_name_y="", jaccard_type='default', J_comp=False):
        '''
        ## Calculate jaccard index for connected components of feature x and y
        ### Input
        * feat_name_x, feat_name_y: name of the feature x and y
        * J_comp: whether to calculate Jaccard index Jcomp between CCx and CCy pair 
        * jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

        ### Output
        * if J_comp is True, then jaccard simliarity metrics calculated from jaccard similarity array between CCx and CCy (dim 0: CCx, dim 1: CCy)
        * if J_comp is False, then return pairwise jaccard similarity array between CCx and CCy (dim 0: CCx, dim 1: CCy)
        '''
        J_result = jaccard_and_connected_loc_(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y, J_comp=J_comp, 
                                              jaccard_type=jaccard_type, return_mode='jaccard', return_sep_loc=False)
        return J_result


    def jaccard_top_n_connected_loc(self, feat_name_x='', feat_name_y='', top_n = 2, jaccard_type='default'):
        '''
        ## Calculate top n connected component locations for given feature pairs x and y
        ### Input
        * feat_name_x, feat_name_y: name of the feature x and y
        * top_n: the number of the top connected components to be found
        * jaccard_type: type of the jaccard index output ('default': jaccard index or 'weighted': weighted jaccard index)

        ### Output
        * AnnData with intersecting location of top n connected components between feature x and y saved in metadata(.obs)
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
        axis = vis_all_connected_visium(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y,
                                        spot_size=spot_size, alpha_img = alpha_img, alpha = alpha, vis_jaccard = vis_jaccard, jaccard_type=jaccard_type,
                                        fig_size = fig_size, batch_colname=batch_colname, batch_name = batch_name, batch_library_dict=batch_library_dict,
                                        image_res = image_res, adjust_image = adjust_image, border = border, 
                                        title_fontsize=title_fontsize, legend_fontsize = legend_fontsize, title = title, return_axis=return_axis, axis = axis,
                                        save = save, path = self.save_path, save_name_add = save_name_add, dpi = dpi)
        return axis



class STopover_imageST(STopover_visium):
    '''
    ## Class to calculate connected component location and jaccard similarity indices in image-based ST dataset
    
    ### Input
    * sp_adata: Anndata object for image-based ST data.
    * grid_sp_adata: whether to convert the given cell-level anndata object sp_adata to grid-based dataset.
    * annotate_sp_adata: whether to annotate provided sp_adata (raw count matrix should be contained in .X)
    * sp_load_path: path to image-based ST data directory or .h5ad Anndata object

    * sc_adata: single-cell reference anndata for cell type annotation of image-based ST data
        -> raw count matrix should be saved in .X
        -> If .h5ad file directory is provided, it will load the h5ad file
        -> If None, then leiden cluster numbers will be used to annotate umage-based ST data
    * sc_celltype_colname: column name for cell type annotation information in metadata of single-cell (.obs)
    * ST_type: type of the ST data to be read: cosmx, xenium, merfish (default: 'cosmx')
    * grid_method: type of the method to assign transcript to grid, either transcript coordinate based method and cell coordinate based method (default='transcript')
    * annot_method: cell type annotation method to use. Either 'ingest' or 'tacco' (default='tacco')
    * sc_norm_total: scaling factor for the total count normalization per cell (default = 1e3)
    * min_counts: minimum number of counts required for a cell in spatial data to pass filtering (scanpy.pp.filter_cells) (default = 10).
    * min_cells: minimum number of cells expressed required for a gene in spatial data to pass filtering (scanpy.pp.filter_genes) (default = 5).
    
    * tx_file_name, cell_exprmat_file_name, cell_metadata_file_name: image-based ST file for transcript count, cell-level expression matrix, cell-level metadata
    * fov_colname, cell_id_colname: column name for barcodes corresponding to fov and cell ID
    * tx_xcoord_colname, tx_ycoord_colname, transcript_colname: column name for global x, y coordinates of the transcript and transcript name
    * meta_xcoord_colname, meta_ycoord_colname: column name for global x, y coordinates in cell-level metadata file
    * x_bins, y_bins: number of bins to divide the image-based ST data (for grid-based aggregation)

    * min_size: minimum size of a connected component
    * fwhm: full width half maximum value for the gaussian smoothing kernel as the multiple of the central distance between the adjacent grid
    * thres_per: lower percentile value threshold to remove the connected components
    * save_path: path to save the data files
    * J_count: number of jaccard similarity calculations after the first definition
    '''
    def __init__(self, sp_adata: AnnData = None, grid_sp_adata: bool = True,
                 annotate_sp_adata: bool = False, sp_load_path: str = '.', 
                 sc_adata: AnnData = None, sc_celltype_colname: str = 'celltype', ST_type: str = 'cosmx', grid_method: str = 'transcript', annot_method: str = 'tacco', sc_norm_total: float = 1e3,
                 min_counts: int = 10, min_cells: int = 5, tx_file_name: str = 'tx_file.csv', cell_exprmat_file_name: str ='exprMat_file.csv', cell_metadata_file_name: str = 'metadata_file.csv', 
                 fov_colname: str = 'fov', cell_id_colname: str = 'cell_ID', tx_xcoord_colname: str = 'x_global_px', tx_ycoord_colname: str = 'y_global_px', transcript_colname: str = 'target',
                 meta_xcoord_colname: str = 'CenterX_global_px', meta_ycoord_colname: str = 'CenterY_global_px',
                 x_bins: int = 100, y_bins: int = 100, min_size: int = 20, fwhm: float = 2.5, thres_per: float = 30, save_path: str = '.', J_count: int = 0):

        assert (min_counts >= 0) and (min_cells >= 0)
        assert (x_bins > 0) and (y_bins > 0)
        assert min_size > 0
        assert fwhm > 0
        assert (thres_per >= 0) and (thres_per <= 100)

        # Load the image-based spatial transcriptomics data if no AnnData file was provided
        if sp_adata is None:
            try:
                print("Anndata object is not provided: searching for the .h5ad file in 'sp_load_path'")
                adata_mod = sc.read_h5ad(sp_load_path)
                try: min_size, fwhm, thres_per, x_bins, y_bins, sc_norm_total, min_counts, min_cells, sc_celltype_colname, transcript_colname, grid_method, ST_type = \
                    adata_mod.uns['min_size'], adata_mod.uns['fwhm'], adata_mod.uns['thres_per'], adata_mod.uns['x_bins'], adata_mod.uns['y_bins'], \
                        adata_mod.uns['sc_norm_total'], adata_mod.uns['min_counts'], adata_mod.uns['min_cells'], adata_mod.uns['sc_celltype_colname'], \
                            adata_mod.uns['transcript_colname'], adata_mod.uns['grid_method'], adata_mod.uns['ST_type']
                except: pass
                # Save Jcount value
                J_result_num = [int(key_names.split("_")[2]) for key_names in adata_mod.uns.keys() if key_names.startswith("J_result_")]
                if len(J_result_num) > 0: J_count = max(J_result_num) + 1
            except:
                print("Failed\nReading image-based ST data files in 'sp_load_path'")
                if isinstance(sc_adata, str):
                    try: sc_adata = sc.read_h5ad(sc_adata)
                    except: 
                        print("Path to 'sc_adata' h5ad file not found: replacing with None")
                        sc_adata = None
                try: 
                    adata_mod, adata_cell = read_imageST(sp_load_path, sc_adata=sc_adata, sc_celltype_colname=sc_celltype_colname, ST_type=ST_type, 
                                                         grid_method=grid_method, annot_method=annot_method, 
                                                         min_counts=min_counts, min_cells=min_cells, sc_norm_total=sc_norm_total,
                                                         tx_file_name = tx_file_name, cell_exprmat_file_name=cell_exprmat_file_name, cell_metadata_file_name=cell_metadata_file_name, 
                                                         fov_colname = fov_colname, cell_id_colname=cell_id_colname, 
                                                         tx_xcoord_colname=tx_xcoord_colname, tx_ycoord_colname=tx_ycoord_colname, transcript_colname=transcript_colname,
                                                         meta_xcoord_colname=meta_xcoord_colname, meta_ycoord_colname=meta_ycoord_colname,
                                                         x_bins=x_bins, y_bins=y_bins)
                    adata_mod.uns['adata_cell'] = STopover_imageST(sp_adata=adata_cell, sc_celltype_colname = sc_celltype_colname, save_path=save_path)
                except:
                    raise ValueError("Error while preprocessing image-based ST files from: '"+sp_load_path+"'")        
        else:
            if grid_sp_adata:
                adata_mod, adata_cell = read_imageST(sp_adata_cell=sp_adata, sc_adata=sc_adata, sc_celltype_colname=sc_celltype_colname, 
                                                     ST_type=ST_type, grid_method=grid_method, annot_method=annot_method, 
                                                     min_counts=min_counts, min_cells=min_cells, sc_norm_total=sc_norm_total,
                                                     x_bins=x_bins, y_bins=y_bins, annotate_sp_adata=annotate_sp_adata)
            else:
                if annotate_sp_adata:
                    adata_mod = annotate_ST(adata_mod, sc_norm_total = sc_norm_total, sc_celltype_colname = sc_celltype_colname, 
                                            annot_method = annot_method, return_df = False)
                adata_mod = sp_adata.copy()
        # Make feature names unique
        adata_mod.var_names_make_unique()

        adata_mod.uns['x_bins'], adata_mod.uns['y_bins'] = x_bins, y_bins
        adata_mod.uns['ST_type'], adata_mod.uns['grid_method'] = ST_type, grid_method
        adata_mod.uns['sc_celltype_colname'] = sc_celltype_colname
        adata_mod.uns['transcript_colname'] = transcript_colname
        adata_mod.uns['sc_norm_total'] = sc_norm_total
        adata_mod.uns['min_counts'], adata_mod.uns['min_cells'] = min_counts, min_cells 
    
        # Generate object with the help of STopover_visium
        super(STopover_imageST, self).__init__(sp_adata=adata_mod, lognorm=False, min_size=min_size, fwhm=fwhm, thres_per=thres_per, save_path=save_path, J_count=J_count)

        self.x_bins, self.y_bins = x_bins, y_bins
        self.ST_type, self.grid_method = ST_type, grid_method
        self.sc_celltype_colname = sc_celltype_colname
        self.transcript_colname = transcript_colname
        self.sc_norm_total = sc_norm_total
        self.min_counts, self.min_cells= min_counts, min_cells
        self.spatial_type = 'imageST'

    def __getitem__(self, index):
        """
        Overrides the __getitem__ method to ensure that subsetting returns an instance of STopover_imageST.
        """
        subset = super(STopover_visium, self).__getitem__(index)
        return STopover_imageST(
            sp_adata=subset,
            sc_celltype_colname=self.sc_celltype_colname,
            ST_type=self.ST_type,
            grid_method=self.grid_method, 
            sc_norm_total=self.sc_norm_total,
            min_counts=self.min_counts,
            min_cells=self.min_cells,
            transcript_colname=self.transcript_colname,
            x_bins=self.x_bins,
            y_bins=self.y_bins,
            min_size=self.min_size,
            fwhm=self.fwhm,
            thres_per=self.thres_per,
            save_path=self.save_path,
            J_count=self.J_count
        )

    def __repr__(self):
        if self.is_view:
            return "View of " + self._gen_repr(self.n_obs, self.n_vars).replace("AnnData object", "STopover_imageST object")
        else:
            return self._gen_repr(self.n_obs, self.n_vars).replace("AnnData object", "STopover_imageST object")
        
    def reinitalize(self,sp_adata, lognorm=False, sc_celltype_colname=None, ST_type=None, grid_method=None, 
                    sc_norm_total=None, min_counts=None, min_cells=None, x_bins=None, y_bins=None, transcript_colname=None,
                    min_size=None, fwhm=None, thres_per=None, save_path=None, J_count=None, inplace=True):
        '''
        ## Reinitialize the class
        '''        
        if (sc_celltype_colname is None) or (sc_norm_total is None) or (x_bins is None) or (y_bins is None) \
            (ST_type is None) or (grid_method is None) or (transcript_colname is None) or (min_counts is None) or (min_cells is None):
            sc_celltype_colname = self.sc_celltype_colname
            sc_norm_total = self.sc_norm_total
            x_bins, y_bins = self.x_bins, self.y_bins
            ST_type, grid_method = self.ST_type, self.grid_method
            transcript_colname = self.transcript_colname
            min_counts, min_cells = self.min_counts, self.min_cells

        if inplace:
            self.__init__(sp_adata=sp_adata, sc_celltype_colname=sc_celltype_colname, sc_norm_total=sc_norm_total, 
                          x_bins=x_bins, y_bins=y_bins, ST_type=ST_type, grid_method=grid_method, 
                          transcript_colname=transcript_colname, min_counts=min_counts, min_cells=min_cells,
                          min_size=min_size, fwhm=fwhm, thres_per=thres_per, save_path=save_path, J_count=J_count)
        else:
            sp_adata_mod = STopover_imageST(sp_adata, sc_celltype_colname=sc_celltype_colname, sc_norm_total=sc_norm_total, 
                                            x_bins=x_bins, y_bins=y_bins, ST_type=ST_type, grid_method=grid_method,
                                            transcript_colname=transcript_colname, min_counts=min_counts, min_cells=min_cells,
                                            min_size=min_size, fwhm=fwhm, thres_per=thres_per, save_path=save_path, J_count=J_count)
            return sp_adata_mod


    def celltype_specific_adata(self, cell_types=['']):
        grid_count_celltype_list = celltype_specific_mat(sp_adata=self, grid_method=self.grid_method, tx_info_name='tx_by_cell_grid', celltype_colname=self.sc_celltype_colname, 
                                                         cell_types=cell_types, transcript_colname=self.transcript_colname, sc_norm_total=self.sc_norm_total)
        grid_count_celltype_list = [STopover_imageST(celltype_stopover, sc_celltype_colname=self.sc_celltype_colname, 
                                    sc_norm_total=self.sc_norm_total, x_bins=self.x_bins, y_bins=self.y_bins, 
                                    min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, save_path=self.save_path) for celltype_stopover in grid_count_celltype_list]                                                
        return grid_count_celltype_list


    def topological_similarity_celltype_pair(self, celltype_x='', celltype_y='', feat_pairs=None, use_lr_db=False, lr_db_species='human', db_name='CellTalk',
                                             group_name='batch', group_list=None, jaccard_type='default', J_result_name='result', num_workers=os.cpu_count(), progress_bar=True):
        '''
        ## Calculate Jaccard index between the two cell type-specific expression anndata of image-based ST data
        ### Input
        * celltype_x: name of the cell type x (should be among the column names of .obs)
        * celltype_y: name of the cell type y (should be among the column names of .obs)
            when use_lr_db=True, then the ligand expression in celltype x and receptor expression in celltype y will be searched
        * other parameters: refer to the topological_similarity method
        '''
        adata_x, adata_y = self.celltype_specific_adata(cell_types=[celltype_x, celltype_y])
        # Create combined anndata for two cell type specific count matrices
        comb_var_names = (celltype_x+': '+adata_x.var_names).tolist() + (celltype_y+': '+adata_y.var_names).tolist()
        adata_xy = AnnData(X=sparse.hstack([adata_x.X, adata_y.X]).tocsr(), obs=adata_x.obs)
        adata_xy.var_names = comb_var_names
        adata_xy = STopover_imageST(adata_xy, sc_celltype_colname=self.sc_celltype_colname, sc_norm_total=self.sc_norm_total, 
                                    x_bins=self.x_bins, y_bins=self.y_bins, min_size=self.min_size, fwhm=self.fwhm, thres_per=self.thres_per, 
                                    save_path=self.save_path, J_count=self.J_count)
        if use_lr_db:
            feat_pairs = self.return_lr_db(lr_db_species=lr_db_species, db_name=db_name)
            if db_name=="Omnipath":
                feat_pairs = feat_pairs[['source_genesymbol','target_genesymbol']]
                feat_pairs['source_genesymbol'] = feat_pairs['source_genesymbol'].str.split('_')
                feat_pairs['target_genesymbol'] = feat_pairs['target_genesymbol'].str.split('_')
                feat_pairs = feat_pairs.explode('source_genesymbol', ignore_index=True)
                feat_pairs = feat_pairs.explode('target_genesymbol', ignore_index=True)
            if db_name=="CellTalk": 
                feat_pairs = feat_pairs[['ligand_gene_symbol','receptor_gene_symbol']]
            elif db_name=="CellChat":
                # Modify the dataframe to contain only the ligand and receptor pairs
                df = feat_pairs.assign(
                ligand_gene_symbol = lambda dataframe: dataframe['interaction_name_2'].map(lambda x: x.strip().split("-")[0]),
                receptor = lambda dataframe: dataframe['interaction_name_2'].map(lambda x: x.strip().split("-")[1]),
                receptor1 = lambda dataframe: dataframe['receptor'].map(lambda x: x.split("+")[0][1:] if "(" in x else x),
                receptor2 = lambda dataframe: dataframe['receptor'].map(lambda x: x.split("+")[1][:-1] if "(" in x else None)
                )
                feat_pairs = pd.concat([df.loc[:,['ligand_gene_symbol','receptor1']].rename(columns={'receptor1':'receptor_gene_symbol'}),
                                        df[df['receptor2'].notna()].loc[:,['ligand_gene_symbol','receptor2']].rename(columns={'receptor2':'receptor_gene_symbol'})], 
                                        axis = 0).reset_index(drop=True)
            use_lr_db = False
            print("Calculating topological similarity between genes in '%s' and '%s'" % (celltype_x, celltype_y))
            print("Using "+db_name+" ligand-receptor dataset")
        else: 
            if isinstance(feat_pairs, list): feat_pairs = pd.DataFrame(feat_pairs)

        # Modify the column name of the feature pair dataframe
        celltype_list = [celltype_x, celltype_y]
        for index, colname in enumerate(feat_pairs.columns):
            feat_pairs[colname] = celltype_list[index]+': '+feat_pairs[colname]
        
        # Calculate topological similarites between the pairs from the two cell types  
        adata_xy.topological_similarity(feat_pairs=feat_pairs, use_lr_db=use_lr_db, lr_db_species=lr_db_species, db_name=db_name,
                                        group_name=group_name, group_list=group_list, jaccard_type=jaccard_type, J_result_name=J_result_name, num_workers=num_workers, progress_bar=progress_bar)
        return adata_xy


    def vis_spatial_imageST(self, feat_name='', colorlist = None, dot_size=None, alpha = 0.8, vmax = None, vmin = None, sort_labels=True,
                          fig_size = (10,10), title_fontsize = 20, legend_fontsize = None, title = None, 
                          return_axis=False, figure = None, axis = None, save = False, save_name_add = '', dpi=150):
        axis = vis_spatial_imageST_(data=self, feat_name=feat_name, colorlist = colorlist, dot_size=dot_size, alpha = alpha, vmax=vmax, vmin=vmin, sort_labels=sort_labels,
                                    fig_size = fig_size, title_fontsize = title_fontsize, legend_fontsize = legend_fontsize, title = title, 
                                    return_axis=return_axis, figure=figure, axis = axis, save = save, path = self.save_path, save_name_add = save_name_add, dpi=dpi)
        return axis


    def vis_jaccard_top_n_pair(self, feat_name_x='', feat_name_y='',
                               top_n = 2, jaccard_type='default', ncol = 2, dot_size=None, alpha = 0.8, 
                               fig_size = (10,10), title_fontsize = 20, legend_fontsize = None,
                               title = '', return_axis=False,
                               save = False, save_name_add = '', dpi=150):
        axis = vis_jaccard_top_n_pair_imageST(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y,
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
        axis = vis_all_connected_imageST(data=self, feat_name_x=feat_name_x, feat_name_y=feat_name_y,
                                         dot_size=dot_size, alpha = alpha, vis_jaccard = vis_jaccard, jaccard_type=jaccard_type, 
                                         fig_size= fig_size, title_fontsize = title_fontsize, legend_fontsize = legend_fontsize, 
                                         title = title, return_axis = return_axis, axis = axis,
                                         save = save, path = self.save_path, save_name_add = save_name_add, dpi = dpi)
        return axis
    
    
class STopover_visiumHD(STopover_imageST):
    '''
    ## Class to calculate connected component location and jaccard similarity indices in visium HD dataset
    
    ### Input
    * sp_adata: Anndata object for visiumHD data.
    * annotate_sp_adata: whether to annotate provided sp_adata (raw count matrix should be contained in .X)
    * sp_load_path: path to image-based ST data directory or .h5ad Anndata object

    * sc_adata: single-cell reference anndata for cell type annotation of image-based ST data
        -> raw count matrix should be saved in .X
        -> If .h5ad file directory is provided, it will load the h5ad file
        -> If None, then leiden cluster numbers will be used to annotate umage-based ST data
    * sc_celltype_colname: column name for cell type annotation information in metadata of single-cell (.obs)
    * annot_method: cell type annotation method to use. Either 'ingest' or 'tacco' (default='tacco')
    * sc_norm_total: scaling factor for the total count normalization per cell (default = 1e3)
    
    * bin_path (str, optional): path to the binned output. Defaults to "binned_outputs/square_002um/".
    * read_mode: how the visiumHD dataset is read, whether it is read as a unit of cells or as a unit of bins.
    * source_image_name (str, optional): name of the source image. Defaults to "Visium_HD_Mouse_Brain_tissue_image.tif".
    * min_cells (int, optional): minimum number of counts required for a cell to pass filtering (scanpy.pp.filter_cells). Defaults to 3.
    * min_counts (int, optional): minimum number of cells expressed required for a gene to pass filtering (scanpy.pp.filter_genes). Defaults to 1.
    * mpp: microns per pixel and translates to how many micrometers are captured in each pixel of the input. 
        -> For example, if using the array coordinates (present as .obs["array_row"] and .obs["array_col"]) as an image, each of the pixels would have 2 micrometers in it, so the mpp of that particular representation is 2.
        -> In local testing of the mouse brain, using an mpp of 0.5 has worked well with both GEX and H&E segmentation. The StarDist models were trained on images with an mpp closer to 0.3.
    * prob_thresh_hne: threshold for the probability in H&E image, lowering it makes the model more lenient with regard to what it calls as nuclei
        -> the default setting is quite stringent, while we want to seed a good number of putative cells in the object.
    * prob_thresh_gex: threshold for the probability in total count distribution image, lowering it makes the model more lenient with regard to what it calls as nuclei
    * nms_thresh: threshold to determine whether the putative objects overlap for them to be merged into a single label, increase it in the tissue with high cellularity.
    * sigma: Gaussian filter with a sigma of 5 (measured in pixels) applied for a little smoothing of total count distribution.
    * mask_arr_row_min, mask_arr_row_max, mask_arr_col_min, mask_arr_col_max: minimum or maximum row or column values to crop the image for plotting
    * show_plot: whether to show the plots during the preprocessing
    * save_path (str, optional): _description_. Defaults to '.'.
    * min_counts: minimum number of counts required for a cell to pass filtering (scanpy.pp.filter_cells) (default = 10).
    * min_cells: minimum number of cells expressed required for a gene to pass filtering (scanpy.pp.filter_genes) (default = 5).
    * bin_counts: minimum count value inside of cells required for a cell to psas filtering. Default is 5.
    
    * x_grid_size, y_grid_size: size of the grid in x- and y-direction in number of bins (2 micron if 2 micron bin is used) to divide the visiumHD data (for grid-based aggregation)
    * min_size: minimum size of a connected component
    * fwhm: full width half maximum value for the gaussian smoothing kernel as the multiple of the central distance between the adjacent grid
    * thres_per: lower percentile value threshold to remove the connected components
    * save_path: path to save the data files
    * J_count: number of jaccard similarity calculations after the first definition
    '''    
    def __init__(self, sp_adata: AnnData = None, annotate_sp_adata: bool =False, sp_load_path: str = '.', 
                 sc_adata: AnnData = None, sc_celltype_colname = 'celltype', annot_method: str = 'tacco', sc_norm_total: float = 1e3,
                 bin_path: str = "binned_outputs/square_016um/", source_image_name: str = "Visium_HD_Mouse_Brain_tissue_image.tif", read_mode: str = 'bin',
                 min_counts: int = 1, min_cells: int = 3, bin_counts: int = 5, x_grid_size: float = (55/2), y_grid_size: float = (55/2), 
                 mpp = 0.5, prob_thresh_hne = 0.01, prob_thresh_gex = 0.05, nms_thresh = 0.5, sigma = 5,
                 mask_arr_row_min = 1450, mask_arr_row_max = 1550, mask_arr_col_min = 250, mask_arr_col_max = 450,
                 show_plot = False, min_size: int = 20, fwhm: float = 2.5, thres_per: float = 30, save_path: str = '.', J_count: int = 0):
        
        assert (min_counts >= 0) and (min_cells >= 0)
        assert (x_grid_size > 0) and (y_grid_size > 0)
        assert min_size > 0
        assert fwhm > 0
        assert (thres_per >= 0) and (thres_per <= 100)

        # Load the image-based spatial transcriptomics data if no AnnData file was provided
        if sp_adata is None:
            try:
                print("Anndata object is not provided: searching for the .h5ad file in 'sp_load_path'")
                adata_mod = sc.read_h5ad(sp_load_path)
                try: min_size, fwhm, thres_per, x_grid_size, y_grid_size, sc_norm_total, min_counts, min_cells, bin_counts, sc_celltype_colname = \
                    adata_mod.uns['min_size'], adata_mod.uns['fwhm'], adata_mod.uns['thres_per'], \
                        adata_mod.uns['x_grid_size'], adata_mod.uns['y_grid_size'], \
                        adata_mod.uns['sc_norm_total'], adata_mod.uns['min_counts'], adata_mod.uns['min_cells'], adata_mod.uns['bin_counts'], adata_mod.uns['sc_celltype_colname']
                except: pass
                # Save Jcount value
                J_result_num = [int(key_names.split("_")[2]) for key_names in adata_mod.uns.keys() if key_names.startswith("J_result_")]
                if len(J_result_num) > 0: J_count = max(J_result_num) + 1
            except:
                print("Failed\nReading VisiumHD data files in 'sp_load_path'")
                if isinstance(sc_adata, str):
                    try: sc_adata = sc.read_h5ad(sc_adata)
                    except: 
                        print("Path to 'sc_adata' h5ad file not found: replacing with None")
                        sc_adata = None
                try:
                    from .visiumHD_utils import read_visiumHD
                    adata_mod, adata_cell = read_visiumHD(bin_path=os.path.join(sp_load_path, bin_path), 
                                                          source_image_path = os.path.join(sp_load_path, source_image_name),
                                                          spaceranger_image_path = os.path.join(sp_load_path, bin_path, "spatial"), read_mode=read_mode,
                                                          sc_adata=sc_adata, sc_celltype_colname = sc_celltype_colname, 
                                                          annot_method = "tacco", sc_norm_total = sc_norm_total, x_grid_size=x_grid_size, y_grid_size=y_grid_size,
                                                          min_cells = min_cells, min_counts = min_counts, bin_counts = bin_counts, mpp = mpp, 
                                                          prob_thresh_hne = prob_thresh_hne, prob_thresh_gex = prob_thresh_gex, nms_thresh = nms_thresh, sigma = sigma,
                                                          mask_arr_row_min = mask_arr_row_min, mask_arr_row_max = mask_arr_row_max, mask_arr_col_min = mask_arr_col_min, mask_arr_col_max = mask_arr_col_max,
                                                          show_plot = show_plot, save_path=save_path)
                    if read_mode == 'cell':
                        adata_mod.uns['adata_cell'] = STopover_visiumHD(sp_adata=adata_cell, sc_celltype_colname = sc_celltype_colname, save_path=save_path)
                except:
                    raise ValueError("Error while preprocessing VisiumHD files from: '"+sp_load_path+"'")        
        else:
            if annotate_sp_adata:
                adata_mod = annotate_ST(adata_mod, sc_norm_total = sc_norm_total, sc_celltype_colname = sc_celltype_colname, 
                                        annot_method = annot_method, return_df = False, return_prob=True)
            else:
                adata_mod = sp_adata.copy()
        # Make feature names unique
        adata_mod.var_names_make_unique()

        adata_mod.uns['x_grid_size'], adata_mod.uns['y_grid_size'] = x_grid_size, y_grid_size
        adata_mod.uns['sc_celltype_colname'] = sc_celltype_colname
        adata_mod.uns['sc_norm_total'] = sc_norm_total
        adata_mod.uns['min_counts'], adata_mod.uns['min_cells'], adata_mod.uns['bin_counts'] = min_counts, min_cells, bin_counts
    
        # Generate object with the help of STopover_imageST
        array_col_name = 'array_col' if read_mode=='bin' else 'grid_array_col'
        array_row_name = 'array_row' if read_mode=='bin' else 'grid_array_row'
        super(STopover_visiumHD, self).__init__(sp_adata = adata_mod, annotate_sp_adata = False, sc_celltype_colname = sc_celltype_colname,
                                                grid_method = 'cell', annot_method = annot_method, sc_norm_total = sc_norm_total,
                                                min_counts = min_counts, min_cells = min_cells,
                                                x_bins = adata_mod.obs[array_col_name].astype(int).max() + 1,
                                                y_bins = adata_mod.obs[array_row_name].astype(int).max() + 1,
                                                min_size = min_size, fwhm = fwhm, thres_per = thres_per, 
                                                save_path = save_path, J_count = J_count)

        self.x_grid_size = int([i.split('_')[-1][:3] for i in bin_path.split('/') if 'square' in i][0]) if read_mode=='bin' else x_grid_size
        self.y_grid_size = int([i.split('_')[-1][:3] for i in bin_path.split('/') if 'square' in i][0]) if read_mode=='bin' else y_grid_size
        self.sc_celltype_colname = sc_celltype_colname
        self.sc_norm_total = sc_norm_total
        self.min_counts, self.min_cells, self.bin_counts = min_counts, min_cells, 0 if read_mode=='bin' else bin_counts
        self.spatial_type = 'visiumHD'

    def __getitem__(self, index):
        """
        Overrides the __getitem__ method to ensure that subsetting returns an instance of STopover_imageST.
        """
        subset = super(STopover_imageST, self).__getitem__(index)                 
        return STopover_visiumHD(
            sp_adata=subset,
            annotate_sp_adata=False,
            sc_celltype_colname=self.sc_celltype_colname,
            sc_norm_total=self.sc_norm_total,
            min_counts=self.min_counts,
            min_cells=self.min_cells,
            bin_counts=self.bin_counts,
            x_grid_size=self.x_grid_size,
            y_grid_size=self.y_grid_size,
            min_size=self.min_size,
            fwhm=self.fwhm,
            thres_per=self.thres_per,
            save_path=self.save_path,
            J_count=self.J_count
        )

    def __repr__(self):
        if self.is_view:
            return "View of " + self._gen_repr(self.n_obs, self.n_vars).replace("AnnData object", "STopover_visiumHD object")
        else:
            return self._gen_repr(self.n_obs, self.n_vars).replace("AnnData object", "STopover_visiumHD object")
        
    def vis_spatial_visiumHD(self, feat_name='', colorlist = None, dot_size=None, alpha = 0.8, vmax = None, vmin = None, sort_labels=True,
                             fig_size = (10,10), title_fontsize = 20, legend_fontsize = None, title = None, 
                             return_axis=False, figure = None, axis = None, save = False, save_name_add = '', dpi=150):
        vis_spatial_imageST_(self, feat_name=feat_name, colorlist=colorlist, dot_size=dot_size, alpha=alpha, vmax=vmax, vmin=vmin, sort_labels=sort_labels,
                             fig_size=fig_size, title_fontsize=title_fontsize, 
                             legend_fontsize = legend_fontsize, title = title, 
                             return_axis=return_axis, figure = figure, axis = axis, 
                             save = save, save_name_add = save_name_add, dpi=dpi)
        
# Copy docstrings
STopover_visium.topological_similarity.__doc__ = topological_sim_pairs_.__doc__
STopover_visium.vis_jaccard_top_n_pair.__doc__ = vis_jaccard_top_n_pair_imageST.__doc__
STopover_visium.vis_all_connected.__doc__ = vis_all_connected_visium.__doc__
STopover_imageST.vis_spatial_imageST.__doc__ = vis_spatial_imageST_.__doc__
STopover_imageST.vis_jaccard_top_n_pair.__doc__ = vis_jaccard_top_n_pair_imageST.__doc__
STopover_imageST.celltype_specific_adata.__doc__ = celltype_specific_mat.__doc__
STopover_imageST.vis_all_connected.__doc__ = vis_all_connected_imageST.__doc__
STopover_visiumHD.vis_spatial_visiumHD.__doc__ = vis_spatial_imageST_.__doc__