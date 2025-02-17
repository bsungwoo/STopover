## Python Code Example  
Please refer to .ipynb files in /notebooks  
### 1. Create STopover object  
* sp_adata: Anndata object for Visium/VisiumHD/image-based ST data  
* sp_load_path: path to Visium/VisiumHD/image-based ST data directory or '*.h5ad' file  
* annotate_sp_adata: whether to annotate provided sp_adata (raw count matrix should be contained in .X)  
* save_path: path to save file    
#### Visium  
* lognorm: whether to lognormalize the count matrix saved in adata.X  
#### Image-based ST & VisiumHD
* sc_adata: single-cell reference anndata for cell type annotation of CosMx SMI data  
  -> If the path to "*.h5ad" file is provided, then it will be used as a reference single-cell data  
* sc_celltype_colname: column name for cell type annotation information in metadata of single-cell (.obs)  
* ST_type: type of the ST data to be read: cosmx, xenium, merfish  
* grid_method: type of the method to assign transcript to grid, either transcript coordinate based method and cell coordinate based method. Either 'cell' or 'transcript'  
* annot_method: cell type annotation method to use. Either 'ingest' or 'tacco'  
* sc_norm_total: scaling factor for the total count normalization per cell  
* min_counts: minimum number of counts required for a cell in spatial data to pass filtering (scanpy.pp.filter_cells)  
* min_cells: minimum number of cells expressed required for a gene in spatial data to pass filtering (scanpy.pp.filter_genes)  
* tx_file_name, cell_exprmat_file_name, cell_metadata_file_name: image-based ST file for transcript count, cell-level expression matrix, cell-level metadata  
* fov_colname, cell_id_colname: column name for barcodes corresponding to fov and cell ID  
* tx_xcoord_colname, tx_ycoord_colname, transcript_colname: column name for global x, y coordinates of the transcript and transcript name  
* meta_xcoord_colname, meta_ycoord_colname: column name for global x, y coordinates in cell-level metadata file  
* x_bins, y_bins: number of bins to divide the image-based ST data (for grid-based aggregation)  
#### 1-0. Optimal parameter choices (Visium & VisiumHD & imageST)  
* minimum size of connected components (min_size) = 20  
* Full width half maximum of Gaussian smoothing kernel (fwhm) = 2.5  
* Lower percentile value threshold to remove the connected components (thres_per) = 30  

#### 1-1-1. Create object for Visium dataset  
```Plain Text
from STopover import STopover_visium

sp_adata = STopover_visium(sp_adata=sp_adata, lognorm=False, min_size=20, fwhm=2.5, thres_per=30, save_path='.')
```
#### 1-1-2. Create object for CosMx dataset  
```Plain Text
from STopover import STopover_cosmx

sp_adata = STopover_cosmx(sp_adata=sp_adata, sc_adata=sc_adata, sc_celltype_colname = 'celltype', sc_norm_total=1e3,
                          tx_file_name = 'tx_file.csv', cell_exprmat_file_name='exprMat_file.csv', 
                          cell_metadata_file_name='metadata_file.csv', 
                          x_bins=100, y_bins=100, min_size=20, fwhm=2.5, thres_per=30, save_path='.')
```
#### 1-2. Create object with saved .h5ad file or 10X-formatted Visium/VisiumHD/Image-based ST directory  
```Plain Text
sp_adata = STopover_visium(sp_load_path='~/*.h5ad', adata_type='visium', lognorm=False, min_size=20, fwhm=2.5, thres_per=30, save_path='.')
sp_adata = STopover_visium(sp_load_path='~/Visium_dir', adata_type='visium', lognorm=True, min_size=20, fwhm=2.5, thres_per=30, save_path='.')
sp_adata = STopover_cosmx(sp_load_path='~/CosMx dir', sc_adata=sc_adata, sc_celltype_colname = 'celltype', sc_norm_total=1e3,
                          tx_file_name = 'tx_file.csv', cell_exprmat_file_name='exprMat_file.csv', 
                          cell_metadata_file_name='metadata_file.csv', 
                          x_bins=100, y_bins=100, min_size=20, fwhm=2.5, thres_per=30, save_path='.')
```
#### 1-3. Calculate cell type-specific gene expression in Image-based ST/VisiumHD dataset  
```Plain Text
sp_adata_tumor, sp_adata_cd8 = sp_adata.celltype_specific_adata(cell_types=['Tumor','Cytotoxic CD8+ T'])
```
### 2. Calculate topological similarity between the two values (expression or metadata)  
feat_pairs: list of features (genes or metadata) with the format [('A','B'),('C','D')] or the pandas dataframe equivalent  
use_lr_db: whether to use list of features in CellTalkDB L-R database  
lr_db_species: select species to utilize in CellTalkDB database  
group_name: name of the group to seprately evaluate the topological similarity  
  -> when there is only one slide, then group_name = None  
  -> when there are multiple slides, then group_name = (group name to identify slides; e.g. 'batch')  
group_list: list of elements of the given group  
J_result_name: name to save the jaccard similarity index results in adata.uns  
num_workers: number of workers to use for multiprocessing (default: os.cpu_count())  

Jaccard indexes bewteen all feature pairs are saved in adata.uns under the name 'J_'  
Connected component locations are saved in adata.obs  
```Plain Text
## Analysis for the dataset having 1 Visium slide or 1 CosMx dataset
# Between two gene expression patterns (CD274 & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('CD274','PDCD1')], J_result_name='result')   
# Between cell fraction metadata and gene (Tumor & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('Tumor','PDCD1')], J_result_name='result')  

## Analysis for the dataset containing 4 Visium slides with batch number 0 ~ 3  
# Between two gene expression patterns (CD274 & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('CD274','PDCD1')], group_name='batch', group_list=[str(i) for i in range(4)], J_result_name='result')   
# Between cell fraction metadata and gene (Tumor & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('Tumor','PDCD1')], group_name='batch', group_list=[str(i) for i in range(4)], J_result_name='result')  

## Return CellTalkDB in pandas dataframe
sp_adata.return_lrdb(lr_db_species='human')

## L-R interaction analysis in Visium dataset
sp_adata.topological_similarity(use_lr_db=True, lr_db_species='human', jaccard_type='default', J_result_name='result')

## L-R interaction analysis using cell type-specific expression data in CosMx
# Between ligand expression in celltype x and receptor expression in celltype y
sp_adata_lr_celltype = sp_adata.topological_similarity_celltype_pair(celltype_x='Tumor', celltype_y='Cytotoxic CD8+ T', 
                                                                     use_lr_db=True, lr_db_species='human', J_result_name='result')
```
### 3. Save the data file  
```Plain Text
sp_adata.save_connected_loc_data(save_format='h5ad', filename = 'cc_loc')  
```
### 4-1. Visium: visualize the overlapping connected components between two values  
```Plain Text  
# Visium: Visualization of connected component locations of feature x and y
sp_adata.vis_all_connected(feat_name_x='CD274', feat_name_y='PDCD1', 
                           spot_size=1, alpha_img=0.8, alpha=0.8, vis_jaccard=True,  
                           fig_size=(5,5), 
                           # batch_colname ='batch', batch_name='0', # For multiple slides
                           image_res='hires',  
                           adjust_image=True, border=50,
                           title_fontsize=20, legend_fontsize=10,
                           title='', return_axis=False, axis=None,
                           save=False, save_name_add='test', dpi=150)  

# Visium: Visualize location of top 2 connected components
sp_adata.vis_jaccard_top_n_pair(feat_name_x='CD274', feat_name_y='PDCD1',  
                                top_n=2, ncol=2, spot_size=1, alpha_img=0.8, alpha=0.8, 
                                fig_size=(5,5), 
                                # batch_colname='batch', batch_name='0', # For multiple slides
                                image_res='hires', 
                                adjust_image=True, border=50,
                                title_fontsize=20, legend_fontsize=10,
                                title='', return_axis=False,  
                                save=False, save_name_add='test', dpi=150)  
```
### 4-2. CosMx: visualize the overlapping connected components between two values  
```Plain Text  
# CosMx: Spatial mapping of feature in grid-based data
sp_adata.vis_spatial_imageST(feat_name='CD274', 
                           dot_size=3, alpha=0.8, vmax=None, vmin=None, 
                           fig_size=(5,5), title_fontsize=20, legend_fontsize=12, 
                           title='Spatial mapping: ', return_axis=False, axis=None,
                           save=False, save_name_add='test', dpi=150)

# CosMx: Visualization of connected component locations of feature x and y
sp_adata.vis_all_connected(feat_name_x='CD274', feat_name_y='PDCD1', 
                           dot_size=3, alpha = 0.8, 
                           fig_size=(5,5), title_fontsize=20, legend_fontsize=12, 
                           title='', return_axis=False, axis=None,
                           save=False, save_name_add='test', dpi=150)

# CosMx: Visualize top 2 connected component locations  
sp_adata.vis_jaccard_top_n_pair(feat_name_x='CD274', feat_name_y='PDCD1', 
                                top_n=2, ncol=2, dot_size=3, alpha=0.8, 
                                fig_size=(5,5), title_fontsize=20, legend_fontsize=12,
                                title='', return_axis=False,
                                save=False, save_name_add='test', dpi=150)
```
### 5. Initialize the STopover object for recalculation  
```Plain Text 
sp_adata.J_result_reset()
```
