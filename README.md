# STopover: graph filtration for extraction of spatial overlap patterns in spatial transcriptomic data

## Python environment for implementation     
### Install conda environment and add jupyter kernel  
```Plain Text  
  conda create -n STopover -c conda-forge graph-tool=2.45 python=3.7  
  conda activate STopover  
  ## For the private code (using token)    
  pip install git+https://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@github.com/bsungwoo/STopover.git  
  ## For the public code  
  (pip install git+https://github.com/bsungwoo/STopover.git)  
  python -m ipykernel install --user --name STopover --display-name STopover  
```

### Dependency (python)  
```Plain Text
python 3.7  
scanpy 1.5.1  
numpy 1.21.2  
pandas 1.3.2  
h5py 2.10.0  
scikit-learn 0.24.2  
scipy 1.7.1  
graph-tool 2.45  
```

## Code Example  
### 1. Create STopover object 
load_path: path to file  
save_path: path to save file  
adata_type: type of spatial data ('visium' or 'cosmx')  
lognorm: whether to lognormalize the count matrix saved in adata.X  

#### 1-0. Optimal parameter choices  
  minimum size of connected components (min_size) = 20  
  Full width half maximum of Gaussian smoothing kernel (fwhm) = 2.5  
  Lower percentile value threshold to remove the connected components (thres_per) = 30  

#### 1-1 Create object with Anndata object (sp_adata)  
```Plain Text
from STopover import STopover  

sp_adata = STopover(adata=sp_adata, adata_type='visium', lognorm=False, min_size=20, fwhm=2.5, thres_per=30, save_path='.')  
```

#### 1-2 Create object with saved .h5ad file or 10X-formatted Visium directory  
```Plain Text
sp_adata = STopover(load_path='~/*.h5ad', adata_type='visium', lognorm=False, min_size=20, fwhm=2.5, thres_per=30, save_path='.')  
sp_adata = STopover(load_path='~/Visium_dir', adata_type='visium', lognorm=True, min_size=20, fwhm=2.5, thres_per=30, save_path='.')  
```

### 2. Calculate topological similarity between the two values (expression or metadata)  
feat_pairs: list of features (genes or metadata) with the format [('A','B'),('C','D')] or the pandas dataframe equivalent  
group_name: name of the group to seprately evaluate the topological similarity  
  -> when there is only one slide, then group_name = None  
  -> when there are multiple slides, then group_name = (group name to identify slides; e.g. 'batch')  
group_list: list of elements of the given group  
J_result_name: name to save the jaccard similarity index results in adata.uns  

Jaccard indexes bewteen all feature pairs are saved in adata.uns under the name 'J_'  
Connected component locations are saved in adata.obs  
```Plain Text
## Analysis for the dataset having 1 Visium slide
# Between two gene expression patterns (CD274 & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('CD274','PDCD1')], J_result_name='result')   
# Between cell fraction metadata and gene (Tumor & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('Tumor','PDCD1')], J_result_name='result')  

## Analysis for the dataset containing 4 Visium slides with batch number 0 ~ 3  
# Between two gene expression patterns (CD274 & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('CD274','PDCD1')], group_name='batch', group_list=[str(i) for i in range(4)], J_result_name='result')   
# Between cell fraction metadata and gene (Tumor & PDCD1)  
sp_adata.topological_similarity(feat_pairs=[('Tumor','PDCD1')], group_name='batch', group_list=[str(i) for i in range(4)], J_result_name='result')  
```

### 3. Save the data file  
```Plain Text
sp_adata.save_connected_loc_data(save_format='h5ad', filename = 'adata_cc_loc')  
```

### 4. Visualize the overlapping connected components between two values  
```Plain Text  
# All connected component location for each feature  
sp_adata.vis_all_connected(vis_intersect_only=False, cmap='tab20', spot_size=1, 
                           alpha_img=0.8, alpha=0.8,  
                           feat_name_x='CD274', feat_name_y='PDCD1',  
                           fig_size=(5,5), 
                           batch_colname ='batch', batch_name='0', image_res='hires',  
                           adjust_image=True, border = 50,  
                           fontsize=20, title = 'Locations of', return_axis=False,  
                           save=True, save_name_add='test', dpi=300)  

# Only the overlapping connected component location (color of connected components for feature x)  
sp_adata.vis_all_connected(vis_intersect_only=True, cmap='tab20', spot_size=1, 
                           alpha_img=0.8, alpha=0.8,  
                           feat_name_x='CD274', feat_name_y='PDCD1',  
                           fig_size=(5,5),  
                           batch_colname='batch', batch_name='0', image_res='hires',  
                           adjust_image=True, border=50,  
                           fontsize=20, title='Locations of', return_axis=False,  
                           save=True, save_name_add='test', dpi=300)  

# Visualize top 2 connected components  
sp_adata.vis_jaccard_top_n_pair(top_n=2, cmap='tab20', spot_size=1,  
                                alpha_img=0.8, alpha=0.8, 
                                feat_name_x='CD274', feat_name_y='PDCD1',  
                                fig_size=(5,5), 
                                batch_colname='batch', batch_name='0', image_res='hires', 
                                adjust_image=True, border=50,  
                                fontsize=20, title = 'J', return_axis=False,  
                                save=False, save_name_add='test', dpi=300)  
```
### 5. Initialize the STopover object for recalculation  
```Plain Text 
sp_adata.J_result_reset()
```