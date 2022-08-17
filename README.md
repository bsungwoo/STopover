# STopover: graph filtration for extraction of spatial overlap patterns in spatial transcriptomic data

## Python environment for implementation     
### Install conda environment and add jupyter kernel  
```Plain Text  
  conda create -n STopover python=3.8
  conda activate STopover
  ## For the private code (using token)
  pip install git+https://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@github.com/bsungwoo/STopover.git
  ## For the public code
  (pip install git+https://github.com/bsungwoo/STopover.git)
  python -m ipykernel install --user --name STopover --display-name STopover
```
### Dependency (python)  
```Plain Text
python 3.8
scanpy 1.9.1
numpy 1.20.3
pandas 1.4.3
matplotlib 3.4.3
pyarrow 8.0.0
pyqt5 5.15.7
scikit-learn 0.24.2
scipy 1.7.3
networkx 2.6.3
```

## Run GUI for STopover  
```Plain Text
from STopover import app

app.main()
```

## Code Example  
### 1. Create STopover object  
sp_adata: Anndata object for VisiumCosMx SMI data with count matrix ('raw') in .X  
sp_load_path: path to Visium/CosMx directory or '*.h5ad' file  
save_path: path to save file    
#### Visium  
lognorm: whether to lognormalize the count matrix saved in adata.X  
#### CosMx  
x_bins, y_bins: number of bins to divide the CosMx SMI data (for grid-based aggregation)   
sc_adata: single-cell reference anndata for cell type annotation of CosMx SMI data  
sc_celltype_colname: column name for cell type annotation information in metadata of single-cell (.obs)  
sc_norm_total: scaling factor for the total count normalization per cell  
tx_file_name: CosMx file for transcript count  
cell_exprmat_file_name: CosMx file for cell-level expression matrix  
cell_metadata_file_name: CosMx file for cell-level metadata  

#### 1-0. Optimal parameter choices (Visium & CosMx)  
  minimum size of connected components (min_size) = 20  
  Full width half maximum of Gaussian smoothing kernel (fwhm) = 2.5  
  Lower percentile value threshold to remove the connected components (thres_per) = 30  

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
#### 1-2. Create object with saved .h5ad file or 10X-formatted Visium/CosMx directory  
```Plain Text
sp_adata = STopover_visium(sp_load_path='~/*.h5ad', adata_type='visium', lognorm=False, min_size=20, fwhm=2.5, thres_per=30, save_path='.')
sp_adata = STopover_visium(sp_load_path='~/Visium_dir', adata_type='visium', lognorm=True, min_size=20, fwhm=2.5, thres_per=30, save_path='.')
sp_adata = STopover_cosmx(sp_load_path='~/CosMx dir', sc_adata=sc_adata, sc_celltype_colname = 'celltype', sc_norm_total=1e3,
                          tx_file_name = 'tx_file.csv', cell_exprmat_file_name='exprMat_file.csv', 
                          cell_metadata_file_name='metadata_file.csv', 
                          x_bins=100, y_bins=100, min_size=20, fwhm=2.5, thres_per=30, save_path='.')
```
#### 1-3. Calculate cell type-specific gene expression in CosMx dataset  
```Plain Text
sp_adata_Tumor, sp_adata_cd8 = sp_adata.celltype_specific_adata(cell_types=['Tumor','Cytotoxic CD8+ T'])
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

Jaccard indexes bewteen all feature pairs are saved in adata.uns under the name 'J_'  
Connected component locations are saved in adata.obs  
```Plain Text
## Analysis for the dataset having 1 Visium slide or 1 CosMx dataset
# Between two gene expression patterns (CD274 & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('CD274','PDCD1')], J_result_name='result')   
# Between cell fraction metadata and gene (Tumor & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('Tumor','PDCD1')], J_result_name='result')  

## L-R interaction analysis in Visium dataset
sp_adata.topological_similarity(use_lr_db=True, lr_db_species='human', J_result_name='result')

## L-R interaction analysis using cell type-specific expression data in CosMx
# Between ligand expression in celltype x and receptor expression in celltype y
sp_adata_lr_celltype = sp_adata.topological_similarity_celltype_pair(celltype_x='Tumor', celltype_y='Cytotoxic CD8+ T', 
                                                                     use_lr_db=True, lr_db_species='human', J_result_name='result')

## Analysis for the dataset containing 4 Visium slides with batch number 0 ~ 3  
# Between two gene expression patterns (CD274 & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('CD274','PDCD1')], group_name='batch', group_list=[str(i) for i in range(4)], J_result_name='result')   
# Between cell fraction metadata and gene (Tumor & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('Tumor','PDCD1')], group_name='batch', group_list=[str(i) for i in range(4)], J_result_name='result')  
```

### 3. Save the data file  
```Plain Text
sp_adata.save_connected_loc_data(save_format='h5ad', filename = 'cc_loc')  
```

### 4-1. Visium: visualize the overlapping connected components between two values  
```Plain Text  
# Visium: Visualization of connected component locations of feature x and y
sp_adata.vis_all_connected(spot_size=1, alpha_img=0.8, alpha=0.8,  
                           feat_name_x='CD274', feat_name_y='PDCD1',  
                           fig_size=(5,5), 
                           # batch_colname ='batch', batch_name='0', # For multiple slides
                           image_res='hires',  
                           adjust_image=True, border = 50,
                           fontsize=20, title = 'Locations of', return_axis=False,  
                           save=False, save_name_add='test', dpi=150)  

# Visium: Visualize location of top 2 connected components
sp_adata.vis_jaccard_top_n_pair(top_n=2, spot_size=1, alpha_img=0.8, alpha=0.8, 
                                feat_name_x='CD274', feat_name_y='PDCD1',  
                                fig_size=(5,5), 
                                # batch_colname='batch', batch_name='0', # For multiple slides
                                image_res='hires', 
                                adjust_image=True, border=50,
                                fontsize=20, title = 'J', return_axis=False,  
                                save=False, save_name_add='test', dpi=150)  
```
### 4-2. CosMx: visualize the overlapping connected components between two values  
```Plain Text  
# CosMx: Spatial mapping of feature in grid-based data
sp_adata.vis_spatial_cosmx(feat_name='CD274', 
                           alpha = 0.8, dot_size=3,
                           fig_size = (5,5), title_fontsize = 20, legend_fontsize = 12, 
                           title = 'Spatial mapping', return_axis=False, 
                           save = False, save_name_add = 'test', dpi=150)

# CosMx: Visualization of connected component locations of feature x and y
sp_adata.vis_all_connected(feat_name_x='CD274', feat_name_y='PDCD1', 
                           alpha = 0.8, dot_size=3,
                           fig_size=(5,5), title_fontsize = 20, legend_fontsize = 12, 
                           title = 'Locations of', return_axis=False,
                           save = False, save_name_add = 'test', dpi = 150)

# CosMx: Visualize top 2 connected component locations  
sp_adata.vis_jaccard_top_n_pair(feat_name_x='CD274', feat_name_y='PDCD1', 
                                top_n = 5, alpha = 0.8, dot_size=3,
                                fig_size = (5,5), title_fontsize = 20, legend_fontsize = 12,
                                title = 'J', return_axis=False,
                                save = False, save_name_add = 'test', dpi=150)
```

### 5. Initialize the STopover object for recalculation  
```Plain Text 
sp_adata.J_result_reset()
```