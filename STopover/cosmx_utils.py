import os
import time
import numpy as np
import numpy.matlib
import pandas as pd
import scanpy as sc
from pyarrow import csv

from scipy import sparse
from anndata import AnnData as an

import warnings
warnings.filterwarnings("ignore")


def annotate_cosmx(sp_adata, sc_adata, sc_norm_total=1e3,
                   sc_celltype_colname = 'celltype', fov_colname = 'fov', cell_id_colname='cell_ID', return_df=True):
    '''
    ## Annotate cells composing CosMx SMI data
    
    ### Input
    sp_adata: CosMx SMI anndata with raw count matrix in .X
    sc_adata: single-cell reference anndata with raw count matrix in .X (cell type annotation of CosMx SMI data)
    sc_celltype_colname: column name for cell annotation of single-cell data (in .obs)
    return_format: whether to return cell annotation dataframe ('dataframe') or anndata file ('adata')
    
    ### Output
    sp_adata: spatial anndata for cell-level CosMx expression data with cell annotation
    return_df: whether to return pandas dataframe that summarizes cell type for each cell 
    '''
    # Check if cell type annotation is included in single-cell data
    if sc_celltype_colname not in sc_adata.obs.columns:
        raise ValueError("Cell type annotation (sc_celltype_colname) not found in sc_adata.obs")
    # Normalize single-cell and spatial data
    sc_adata.var_names_make_unique()
    sc.pp.normalize_total(sc_adata, target_sum=1e4, inplace=True)
    sc.pp.normalize_total(sp_adata, target_sum=sc_norm_total, inplace=True)

    # Log transform and scale single-cell data
    sc.pp.log1p(sc_adata)
    sc_adata.raw = sc_adata
    sc.pp.scale(sc_adata, max_value=10)
    # Log transform and scale spatial (CosMx data)
    sc.pp.log1p(sp_adata)
    sp_adata.raw = sp_adata
    sc.pp.scale(sp_adata, max_value=10)
    # Find intersecting genes
    inter_var_names = sc_adata.var_names.intersection(sp_adata.var_names)
    sc_adata = sc_adata[:, inter_var_names]
    sp_adata = sp_adata[:, inter_var_names]
    
    # Perform PCA and find neighbords and umap embedding
    sc.pp.pca(sc_adata)
    sc.pp.neighbors(sc_adata)
    sc.tl.umap(sc_adata)
    # Fill nan values and designate as categorical variable
    sc_adata.obs[sc_celltype_colname] = sc_adata.obs[sc_celltype_colname].astype(object).fillna('nan')
    sc_adata.obs[sc_celltype_colname] = sc_adata.obs[sc_celltype_colname].astype('category')
    # Perform cell label transfer from single-cell to CosMx data
    sc.tl.ingest(sp_adata, sc_adata, obs=sc_celltype_colname, embedding_method='umap')

    if return_df:
        # Data frame containing annotated cell types in CosMx SMI data
        df_celltype = sp_adata.obs.loc[:,[sc_celltype_colname,
                                          fov_colname,cell_id_colname]].set_index([fov_colname,cell_id_colname])
        return sp_adata.raw.to_adata(), df_celltype 
    else:
        return sp_adata.raw.to_adata()



def read_cosmx(load_path, sc_adata, sc_celltype_colname = 'celltype', sc_norm_total=1e3,
               tx_file_name = 'tx_file.csv', cell_exprmat_file_name='exprMat_file.csv', cell_metadata_file_name='metadata_file.csv', 
               fov_colname = 'fov', cell_id_colname='cell_ID', tx_xcoord_colname='x_global_px', tx_ycoord_colname='y_global_px', transcript_colname='target',
               meta_xcoord_colname='CenterX_global_px', meta_ycoord_colname='CenterY_global_px',
               x_bins=100, y_bins=100):
    '''
    ## Load CosMx dataset and preprocess data
    ### Input
    load_path: path to load the CosMx SMI files
    sc_adata: single-cell reference anndata for cell type annotation of CosMx SMI data
    sc_celltype_colname: column name for cell type annotation information in metadata of single-cell (.obs)
    sc_norm_total: scaling factor for the total count normalization per cell
    tx_file_name, cell_exprmat_file_name, cell_metadata_file_name: CosMx file for transcript count, cell-level expression matrix, cell-level metadata
    fov_colname, cell_id_colname: column name for barcodes corresponding to fov and cell ID
    tx_xcoord_colname, tx_ycoord_colname, transcript_colname: column name for global x, y coordinates of the transcript and transcript name
    meta_xcoord_colname, meta_ycoord_colname: column name for global x, y coordinates in cell-level metadata file
    x_bins, y_bins: number of bins to divide the CosMx SMI data (for grid-based aggregation)

    ### Output
    sp_adata_grid: grid-based log-normalized count anndata with cell abundance information saved in .obs
    sp_adata_cell: cell-based log-normalized count anndata
    '''
    # Check data feasibility
    if sc_celltype_colname not in sc_adata.obs.columns:
        raise ValueError("Cell type annotation (sc_celltype_colname) not found in sc_adata.obs")
        
    ## Read transcript information file
    start_time = time.time()
    tx_coord_all = csv.read_csv(os.path.join(load_path, tx_file_name)).to_pandas().loc[:,[fov_colname,cell_id_colname,
                                                                                          tx_xcoord_colname,tx_ycoord_colname,transcript_colname]]
    # Remove transcript data not included in a cell
    tx_coord_all = tx_coord_all[tx_coord_all[cell_id_colname] != 0]

    ## Grid-based aggregation of CosMx: divide coordinates by x_bins and y_bins and aggregate
    # Find the x and y coordinate arrays
    x_coord = tx_coord_all[[tx_xcoord_colname]].to_numpy()
    y_coord = tx_coord_all[[tx_ycoord_colname]].to_numpy()
    # Find the coordinates that equally divides the x and y axis into x_bins and y_bins
    x_div_arr = np.linspace(np.min(x_coord), np.max(x_coord), num=x_bins, endpoint=False)[1:]
    y_div_arr = np.linspace(np.min(y_coord), np.max(y_coord), num=y_bins, endpoint=False)[1:]
    # Assigning the grid column and row number to each transcript based on the coordinates by x_div_arr and y_div_arr
    tx_coord_all['array_col'] = (np.matlib.repmat(x_div_arr.reshape(1,-1), len(x_coord), 1) < np.matlib.repmat(x_coord, 1, len(x_div_arr))).sum(axis=1).astype(int)
    tx_coord_all['array_row'] = (np.matlib.repmat(y_div_arr.reshape(1,-1), len(y_coord), 1) < np.matlib.repmat(y_coord, 1, len(y_div_arr))).sum(axis=1).astype(int)
    print("End of grid-based aggregation of CosMx: %.2f seconds" % (time.time()-start_time))

    ## Normalize the transcript number in each grid by total count in the cell
    tx_by_cell_grid = tx_coord_all.groupby([fov_colname,cell_id_colname,'array_col','array_row',transcript_colname])[transcript_colname].count().to_frame('count')
    tx_by_cell_grid['tx_fx_by_grid'] = tx_by_cell_grid['count'] / tx_by_cell_grid.groupby([fov_colname,cell_id_colname]).transform('sum')['count']
    # Generate normalization count matrix by grid
    grid_tx_count = tx_by_cell_grid.pivot_table(index=['array_col','array_row'], columns=transcript_colname, values='tx_fx_by_grid', aggfunc=['sum']).fillna(0)
    # Saving grid barcode and gene symbol names
    var_names = grid_tx_count.columns.to_frame()[transcript_colname].to_numpy()
    grid_metadata = grid_tx_count.index.to_frame()
    grid_metadata.index = grid_metadata['array_col'].astype(str) + '_' + grid_metadata['array_row'].astype(str)
    # Log transformation of grid based count
    grid_tx_count = (sc_norm_total*sparse.csr_matrix(grid_tx_count, dtype=np.float32)).log1p()
    print("End of generating grid-based count matrix: %.2f seconds" % (time.time()-start_time))

    # Cell annotation for spatial data
    ## Generate AnnData for the problem
    # Load expression matrix
    exp_mat = csv.read_csv(os.path.join(load_path, cell_exprmat_file_name)).to_pandas()
    exp_mat = exp_mat[exp_mat[fov_colname] != 0].set_index([fov_colname,cell_id_colname])
    # Generate cell barcodes for CosMx SMI data
    cell_names_expmat = exp_mat.index.to_frame()
    cell_names_expmat = (cell_names_expmat[fov_colname].astype(str) + '_' + cell_names_expmat[cell_id_colname].astype(str)).to_numpy()
    # Load CosMx SMI cell metadata
    cell_meta = csv.read_csv(os.path.join(load_path, cell_metadata_file_name)).to_pandas().loc[:,[meta_xcoord_colname, meta_ycoord_colname]]
    cell_meta.columns = ['array_col', 'array_row']
    cell_meta = pd.concat([cell_meta, exp_mat.index.to_frame().reset_index(drop=True)], axis=1)
    # Generate CosMx SMI spatial anndata file
    sp_adata_cell = an(X = sparse.csr_matrix(exp_mat, dtype=np.float32), obs=cell_meta)
    sp_adata_cell.var_names = var_names
    sp_adata_cell.obs_names = cell_names_expmat
    # Remove cells with total transcript count of 0
    sc.pp.filter_cells(sp_adata_cell, min_counts=1)
    print("End of creating CosMx cell-level anndata: %.2f seconds" % (time.time()-start_time))

    ## Annotation of cell-level CosMx SMI data
    sp_adata_cell, df_celltype = annotate_cosmx(sp_adata_cell, sc_adata, sc_norm_total=sc_norm_total,
                                                sc_celltype_colname = sc_celltype_colname, 
                                                fov_colname = fov_colname, cell_id_colname=cell_id_colname, return_df=True)
    print("End of annotating CosMx cell-level anndata: %.2f seconds" % (time.time()-start_time))

    ## Create dataframe with cell type abundance in each grid
    # Create dataframe with transcript count according to cell ID and grid number: cell type information added
    tx_by_cell_grid = tx_by_cell_grid.join(df_celltype, how='inner')
    grid_celltype = tx_by_cell_grid.pivot_table(index=['array_col','array_row'], columns=[sc_celltype_colname], values='tx_fx_by_grid', aggfunc=['sum']).fillna(0)
    # Assign column names to the dataframe
    grid_celltype.columns = grid_celltype.columns.to_frame()[sc_celltype_colname]
    # Assign index names to the dataframe
    grid_index = grid_celltype.index.to_frame()
    grid_celltype.index = grid_index['array_col'].astype(str) + '_' + grid_index['array_row'].astype(str)
    # Modify metadata to contain cell type information in each grid
    grid_metadata = grid_metadata.join(grid_celltype, how='inner')
    print("End of generating grid-based cell type abundance metadata: %.2f seconds" % (time.time()-start_time))

    ## Generating grid-based CosMx SMI spatial anndata
    sp_adata_grid = an(X = grid_tx_count, obs=grid_metadata)
    sp_adata_grid.var_names = var_names
    sp_adata_grid.uns['tx_by_cell_grid'] = tx_by_cell_grid.reset_index()
    print("End of generating grid-based CosMx spatial anndata: %.2f seconds" % (time.time()-start_time))

    return sp_adata_grid, sp_adata_cell



def celltype_specific_mat(sp_adata, tx_info_name='tx_by_cell_grid', celltype_colname=None, cell_types=[''], 
                          transcript_colname='target', sc_norm_total=1e3):
    '''
    ## Return cell type specific transcript count matrix
    ### Input
    sp_adata: grid-based count anndata with cell abundance information saved in .obs
    tx_info_name: key name of the transcript information file saved in sp_adata.uns
    celltype_colname: column name of the dataframe; explains which cell each transcript is beloning to
    cell_types: the cell types to extract cell type-specific count information
    transcript_colname: column name for the transcipt name
    sc_norm_total: scaling factor for the total count normalization per cell

    ### Output
    grid_tx_count_celltype: celltype specific grid-based count matrix as sparse.csr_matrix format
    '''
    if tx_info_name not in sp_adata.uns.keys():
        raise ValueError("'sp_adata.uns' should contain 'tx_info_name' which explains transcript information by cells and grids")
    
    tx_by_cell_grid = sp_adata.uns[tx_info_name]
    if celltype_colname not in tx_by_cell_grid.columns:
        raise ValueError("'celltype_colname' should be among the column names of dataframe: 'tx_info_name'")
    # Boolean pandas series for the celltype inclusion
    if not (set(cell_types) <= set(tx_by_cell_grid[celltype_colname].cat.categories)): 
        raise ValueError("Some of the cell types in 'cell_types' are not found in the sp_data")
    
    # Subset the transcript information by cell type
    grid_adata_celltype_list = []
    for celltype in cell_types:
        # Find transcript count information for a specific cell type
        tx_by_cell_grid_ = tx_by_cell_grid[tx_by_cell_grid[celltype_colname]==celltype]
        # Generate normalization count matrix by grid
        grid_tx_count_celltype = tx_by_cell_grid_.pivot_table(index=['array_col','array_row'], columns=transcript_colname, values='tx_fx_by_grid', aggfunc=['sum']).fillna(0)
        # Reindex the grid_tx_count_celltype to match with the original grid index
        grid_index = grid_tx_count_celltype.index.to_frame()
        grid_tx_count_celltype.index = grid_index['array_col'].astype(str) + '_' + grid_index['array_row'].astype(str)
        grid_tx_count_celltype = grid_tx_count_celltype.reindex(sp_adata.obs_names, fill_value=0)
        # Log transformation of grid based count and make sparse matrix
        grid_tx_count_celltype_ = (sc_norm_total*sparse.csr_matrix(grid_tx_count_celltype, dtype=np.float32)).log1p()
        # Create anndata for cell type specific count matrix
        grid_adata_celltype = an(X=grid_tx_count_celltype_, obs=sp_adata.obs)
        grid_adata_celltype.var_names = grid_tx_count_celltype.columns.to_frame()[transcript_colname].to_numpy()
        grid_adata_celltype_list.append(grid_adata_celltype)

    return grid_adata_celltype_list