import os
import time
import numpy as np
# from numpy.matlib import repmat
import pandas as pd
import scanpy as sc
from pyarrow import csv

from scipy import sparse
from anndata import AnnData as an

import warnings
warnings.filterwarnings("ignore")


def annotate_ST(sp_adata, sc_adata=None, sc_norm_total=1e3, 
                sc_celltype_colname = 'celltype', annot_method='tacco', return_prob=False,
                cell_id = ['fov','cell_ID'], return_df=True):
    '''
    ## Annotate cells composing image-based ST SMI data

    ### Input
    sp_adata: image-based ST anndata with raw count matrix in .X
    sc_adata: single-cell reference anndata with raw count matrix in .X (cell type annotation of image-based ST data)
    sc_norm_total: scaling factor for the total count normalization per cell
    sc_celltype_colname: column name for cell annotation of single-cell data (in .obs)
    annot_method: cell type annotation methods: either 'ingest' or 'tacco' (default='ingest')
    return_prob: return probability for the cell type annotation at each data point
    cell_id: list of column names that represents cell ids in sp_adata.obs
    return_df: whether to return pandas dataframe that summarizes cell type for each cell

    ### Output
    sp_adata: spatial anndata for cell-level image-based ST expression data with cell annotation
    df_celltype: pandas dataframe that summarizes cell type for each cell
    '''
    # Check feasibility of the variable
    if annot_method not in ["ingest","tacco"]: raise ValueError("'annotate_method' should be either 'ingest' or 'tacco'")
    if return_df and ((not isinstance(cell_id, list)) or len(cell_id)==0 or len(set(cell_id) - set(sp_adata.obs.columns))>0):
        raise TypeError("'cell_id' should be a list of column names that represents cell ids in sp_adata.obs")
    
    if sc_adata is not None:
        print("Using '"+annot_method+"' to annotate cells with reference single-cell data")
        sc_adata_ = sc_adata.copy()
        sc_adata_.var_names_make_unique()
        if annot_method=="ingest":
            print("Running Ingest Annotation..")
            # Normalize single-cell data
            sc.pp.normalize_total(sc_adata_, target_sum=1e4, inplace=True)
            sc.pp.normalize_total(sp_adata, target_sum=sc_norm_total, inplace=True)
            # Log transform and single-cell and spatial dataset
            sc.pp.log1p(sc_adata_)
            sc.pp.log1p(sp_adata)
            sp_adata.raw = sp_adata.copy()
            # Find intersecting genes
            inter_var_names = sc_adata_.var_names.intersection(sp_adata.var_names)
            sc_adata_ = sc_adata_[:, inter_var_names]
            sp_adata = sp_adata[:, inter_var_names]
            sc.pp.scale(sc_adata_, max_value=10)
            sc.pp.scale(sp_adata, max_value=10)
            # Perform PCA and find neighbords and umap embedding
            sc.pp.pca(sc_adata_)
            sc.pp.neighbors(sc_adata_)
            sc.tl.umap(sc_adata_)
            # Fill nan values and designate as categorical variable
            sc_adata_.obs[sc_celltype_colname] = sc_adata_.obs[sc_celltype_colname].astype(object).fillna('nan')
            sc_adata_.obs[sc_celltype_colname] = sc_adata_.obs[sc_celltype_colname].astype('category')
            # Scale single-cell and spatial data
            # Perform cell label transfer from single-cell to image-based ST data
            sc.tl.ingest(sp_adata, sc_adata_, obs=sc_celltype_colname, embedding_method='umap')
            sp_adata = sp_adata.raw.to_adata()
        elif annot_method=="tacco":
            print("Running TACCO Annotation..")
            import tacco as tc
            sp_adata.raw = sp_adata.copy()
            # Find intersecting genes
            inter_var_names = sc_adata_.var_names.intersection(sp_adata.var_names)
            sc_adata_ = sc_adata_[:, inter_var_names]
            sp_adata = sp_adata[:, inter_var_names]
            print("_"*60)
            df_annot = tc.tl.annotate(sp_adata, sc_adata_, annotation_key=sc_celltype_colname)
            if return_prob:
                sp_adata.obs = sp_adata.obs.join(df_annot)
            else:
                sp_adata.obs[sc_celltype_colname] = np.array(df_annot.columns)[np.argmax(df_annot.to_numpy(), axis = 1)]
                sp_adata.obs[sc_celltype_colname] = sp_adata.obs[sc_celltype_colname].astype('category')
            sp_adata = sp_adata.raw.to_adata()
            sc.pp.normalize_total(sp_adata, target_sum=sc_norm_total, inplace=True)
            # Log transform and scale spatial data (image-based ST)
            sc.pp.log1p(sp_adata)
            print("_"*60)
    else:
        print("Single-cell reference dataset not provided: perform unsupervised clustering")
        sc.pp.normalize_total(sp_adata, target_sum=sc_norm_total, inplace=True)
        # Log transform and scale spatial data (image-based ST)
        sc.pp.log1p(sp_adata)
        # Find highly variable genes in spatial data
        sc.pp.highly_variable_genes(sp_adata, flavor="seurat", n_top_genes=2000)
        # Perform PCA and cluster the spots
        sc.pp.pca(sp_adata)
        sc.pp.neighbors(sp_adata)
        sc.tl.umap(sp_adata)
        # Leiden clustering and annotate cell based on the clusters
        sc.tl.leiden(sp_adata, key_added=sc_celltype_colname)

    if return_df:
        # Data frame containing annotated cell types in image-based ST data
        df_celltype = sp_adata.obs.loc[:,[sc_celltype_colname]+cell_id].set_index(cell_id)
        return sp_adata, df_celltype
    else:
        return sp_adata



def read_imageST(load_path=None, sp_adata_cell=None, sc_adata=None, min_counts=10, min_cells=5, sc_celltype_colname = 'celltype', 
                 ST_type='cosmx', grid_method = 'transcript', annot_method='tacco', sc_norm_total=1e3,
                 tx_file_name = 'tx_file.csv', cell_exprmat_file_name='exprMat_file.csv', cell_metadata_file_name='metadata_file.csv',
                 fov_colname = 'fov', cell_id_colname='cell_ID', tx_xcoord_colname='x_global_px', tx_ycoord_colname='y_global_px', transcript_colname='target',
                 meta_xcoord_colname='CenterX_global_px', meta_ycoord_colname='CenterY_global_px',
                 x_bins=100, y_bins=100, annotate_sp_adata=False):
    '''
    ## Load image-based ST dataset and preprocess data
    ### Input
    load_path: path to load the image-based ST files
    sp_adata_cell: cell-level spatial anndata for the image-based ST dataset. 
    sc_adata: single-cell reference anndata for cell type annotation of image-based ST data
        -> raw count matrix should be saved in .X
        -> If None, then leiden cluster numbers will be used to annotate image-based ST data
    min_counts: minimum number of counts required for a cell to pass filtering.
    min_cells: minimum number of cells expressed required for a gene to pass filtering.
    sc_celltype_colname: column name for cell type annotation information in metadata of single-cell (.obs)
    ST_type: type of the ST data to be read: cosmx, xenium, merfish (default: 'cosmx')
    grid_method: type of the method to assign transcript to grid, either transcript coordinate based method and cell coordinate based method (default='transcript')
    annot_method: cell type annotation methods: either 'ingest' or 'tacco' (default='tacco')
    sc_norm_total: scaling factor for the total count normalization per cell 
    tx_file_name, cell_exprmat_file_name, cell_metadata_file_name: image-based ST file for transcript count, cell-level expression matrix, cell-level metadata
    fov_colname, cell_id_colname: column name for barcodes corresponding to fov and cell ID
    tx_xcoord_colname, tx_ycoord_colname, transcript_colname: column name for global x, y coordinates of the transcript and transcript name
    meta_xcoord_colname, meta_ycoord_colname: column name for global x, y coordinates in cell-level metadata file
    x_bins, y_bins: number of bins to divide the image-based ST data (for grid-based aggregation)
    annotate_sp_adata: whether to perform the cell annotation process on cell-level ST anndata

    ### Output
    sp_adata_grid: grid-based log-normalized count anndata with cell abundance information saved in .obs
    sp_adata_cell: cell-based log-normalized count anndata
    '''
    # Check data feasibility
    if sc_adata is None:
        if annotate_sp_adata: 
            print("Reference single-cell data not provided: leiden clustering of image-based ST data will be used for annotation")
    else:
        if sc_celltype_colname not in sc_adata.obs.columns:
            print("Cell type annotation (sc_celltype_colname) not found in sc_adata.obs")
    if ST_type == "cosmx": 
        if grid_method in ["transcript","cell"]: cell_id = [fov_colname, cell_id_colname]
        else: raise ValueError("'grid_method' should be either 'transcript' or 'cell'")
    elif ST_type in ["merfish","xenium"]:
        if grid_method!="cell": raise NotImplementedError("'grid_method' should be 'cell' when 'ST_type' is 'merfish' or 'xenium'")
        cell_id = [cell_id_colname]
    else: raise ValueError("'ST_type' should be among 'cosmx','xenium', or 'merfish'")
    start_time = time.time()
    
    # Cell annotation for spatial data
    ## Generate AnnData for the problem
    # Load expression matrix
    if sp_adata_cell is None:
        if load_path is None:
            raise ValueError("path to the spatial dataset should be provided as 'load_path'")
        if ST_type in ["cosmx","merfish"]:
            exp_mat = csv.read_csv(os.path.join(load_path, cell_exprmat_file_name)).to_pandas()
        elif ST_type=="xenium":
            exp_mat = sc.read_10x_h5(os.path.join(load_path, cell_exprmat_file_name))
            exp_mat = pd.DataFrame(exp_mat.X.toarray(), index=exp_mat.obs_names, columns=exp_mat.var_names)

        # Subset the expression matrix to remove data not included in a cell and from negative probes
        if ST_type=="cosmx": 
            exp_mat = exp_mat[exp_mat[cell_id_colname] != 0].loc[:, ~exp_mat.columns.str.contains('NegPrb')].set_index(cell_id)
            # Generate cell barcodes for image-based ST data
            cell_names_expmat = exp_mat.index.to_frame()
            cell_names_expmat = (cell_names_expmat[fov_colname].astype(str) + '_' + cell_names_expmat[cell_id_colname].astype(str)).to_numpy()
            # Load image-based ST cell metadata
            cell_meta = csv.read_csv(os.path.join(load_path, cell_metadata_file_name)).to_pandas().loc[:,cell_id+[meta_xcoord_colname,meta_ycoord_colname]]
            cell_meta.columns = cell_id+['array_col','array_row']
            cell_meta = pd.merge(exp_mat.index.to_frame().reset_index(drop=True), cell_meta, on=cell_id, how='inner')
        else:
            if ST_type=="merfish": exp_mat = exp_mat.loc[:, ~exp_mat.columns.str.contains('Blank-')].set_index(cell_id)
            cell_names_expmat = exp_mat.index.values.astype(str)
            # Load image-based ST cell metadata
            cell_meta = csv.read_csv(os.path.join(load_path, cell_metadata_file_name)).to_pandas()
            cell_meta = cell_meta.loc[:,[cell_meta.columns[0]]+[meta_xcoord_colname,meta_ycoord_colname]]
            cell_meta.columns = cell_id+['array_col','array_row']
            cell_meta[cell_id_colname] = cell_meta[cell_id_colname].astype(str)
            exp_mat.index = exp_mat.index.astype(str)
            if ST_type=="merfish": 
                cell_meta = pd.merge(exp_mat.index.to_frame().reset_index(drop=True), cell_meta, on=cell_id, how='inner')
            elif ST_type=="xenium":
                cell_meta = pd.merge(exp_mat.index.to_frame().reset_index(drop=True).rename(columns = {0: cell_id_colname}), 
                                    cell_meta, on=cell_id, how='inner')
            # Generate image-based ST anndata file
            sp_adata_cell = an(X = sparse.csr_matrix(exp_mat, dtype=np.float32), obs=cell_meta)
            sp_adata_cell.var_names = exp_mat.columns
            sp_adata_cell.obs_names = cell_names_expmat
    else:
        if not annotate_sp_adata and sc_celltype_colname not in sp_adata_cell.obs.columns:
            raise ValueError(f"'{sc_celltype_colname}' not found in 'sp_adata.obs.columns'")
        df_celltype = sp_adata_cell.obs[[sc_celltype_colname]]

    # Remove cells with total transcript count below min_counts and genes with number of expressed cells (>0) below min_cells
    sc.pp.filter_cells(sp_adata_cell, min_counts=min_counts)
    sc.pp.filter_genes(sp_adata_cell, min_cells=min_cells)
    print("End of creating image-based ST cell-level anndata: %.2f seconds" % (time.time()-start_time))

    ## Annotation of cell-level image-based ST data
    if annotate_sp_adata:
        sp_adata_cell, df_celltype = annotate_ST(sp_adata_cell, sc_adata, sc_norm_total=sc_norm_total, 
                                                 sc_celltype_colname = sc_celltype_colname, annot_method=annot_method,
                                                 cell_id = cell_id, return_df=True)
    print("End of annotating image-based ST cell-level anndata: %.2f seconds" % (time.time()-start_time))

    if grid_method == "transcript":
        ## Read transcript information file
        tx_coord_all = csv.read_csv(os.path.join(load_path, tx_file_name)).to_pandas().loc[:,cell_id+[tx_xcoord_colname,tx_ycoord_colname,transcript_colname]]
        # Remove transcript data not included in a cell and from negative probes
        if ST_type == "cosmx": tx_coord_all = tx_coord_all[(tx_coord_all[cell_id_colname] != 0) & (~tx_coord_all[transcript_colname].str.contains("NegPrb"))]

        ## Grid-based aggregation of image-based ST: divide coordinates by x_bins and y_bins and aggregate
        # Find the x and y coordinate arrays
        x_coord = tx_coord_all[[tx_xcoord_colname]].to_numpy()
        y_coord = tx_coord_all[[tx_ycoord_colname]].to_numpy()
        # Find the coordinates that equally divides the x and y axis into x_bins and y_bins
        x_div_arr = np.linspace(np.min(x_coord), np.max(x_coord), num=x_bins, endpoint=False)[1:]
        y_div_arr = np.linspace(np.min(y_coord), np.max(y_coord), num=y_bins, endpoint=False)[1:]
        # Assigning the grid column and row number to each transcript based on the coordinates by x_div_arr and y_div_arr
        # tx_coord_all['array_col'] = (repmat(x_div_arr.reshape(1,-1), len(x_coord), 1) < repmat(x_coord, 1, len(x_div_arr))).sum(axis=1).astype(int)
        # tx_coord_all['array_row'] = (repmat(y_div_arr.reshape(1,-1), len(y_coord), 1) < repmat(y_coord, 1, len(y_div_arr))).sum(axis=1).astype(int)
        tx_coord_all['array_col'] = np.searchsorted(x_div_arr, x_coord, side='right')
        tx_coord_all['array_row'] = np.searchsorted(y_div_arr, y_coord, side='right')
        print("End of grid-based aggregation of"+ST_type+": %.2f seconds" % (time.time()-start_time))

        ## Normalize the transcript number in each grid by total count in the cell
        tx_by_cell_grid = tx_coord_all.groupby(cell_id+['array_col','array_row',transcript_colname])[transcript_colname].count().to_frame('count')
        # Calculate fraction of each transcript (transcript count in a cell/total count in a cell) in a grid
        tx_by_cell_grid['tx_fx_by_grid'] = tx_by_cell_grid['count'] / tx_by_cell_grid.groupby(cell_id).transform('sum')['count']
        # Generate normalization count matrix by grid
        grid_tx_count = tx_by_cell_grid.pivot_table(index=['array_col','array_row'], columns=transcript_colname, values='tx_fx_by_grid', aggfunc=['sum']).fillna(0)
        # Saving grid barcode and gene symbol names
        var_names = grid_tx_count.columns.to_frame()[transcript_colname].to_numpy()
        grid_metadata = grid_tx_count.index.to_frame()
        grid_metadata.index = grid_metadata['array_col'].astype(str) + '_' + grid_metadata['array_row'].astype(str)
        # Log transformation of grid based count
        grid_tx_count = (sc_norm_total*sparse.csr_matrix(grid_tx_count, dtype=np.float32)).log1p()
        print("End of generating grid-based count matrix: %.2f seconds" % (time.time()-start_time))

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
        grid_metadata = grid_metadata.join(grid_celltype, how='left').fillna(0)
        print("End of generating grid-based cell type abundance metadata: %.2f seconds" % (time.time()-start_time))
    else:
        ## Grid-based aggregation of image-based ST: divide coordinates by x_bins and y_bins and aggregate
        # Find the x and y coordinate arrays
        x_coord = sp_adata_cell.obs[['array_col']].to_numpy()
        y_coord = sp_adata_cell.obs[['array_row']].to_numpy()
        # Find the coordinates that equally divides the x and y axis into x_bins and y_bins
        x_div_arr = np.linspace(np.min(x_coord), np.max(x_coord), num=x_bins, endpoint=False)[1:]
        y_div_arr = np.linspace(np.min(y_coord), np.max(y_coord), num=y_bins, endpoint=False)[1:]
        # Assigning the grid column and row number to each transcript based on the coordinates by x_div_arr and y_div_arr
        # sp_adata_cell.obs['grid_array_col'] = (repmat(x_div_arr.reshape(1,-1), len(x_coord), 1) < repmat(x_coord, 1, len(x_div_arr))).sum(axis=1).astype(int)
        # sp_adata_cell.obs['grid_array_row'] = (repmat(y_div_arr.reshape(1,-1), len(y_coord), 1) < repmat(y_coord, 1, len(y_div_arr))).sum(axis=1).astype(int)
        sp_adata_cell.obs['grid_array_col'] = np.searchsorted(x_div_arr, x_coord, side='right')
        sp_adata_cell.obs['grid_array_row'] = np.searchsorted(y_div_arr, y_coord, side='right')
        print("End of grid-based aggregation of "+ST_type+": %.2f seconds" % (time.time()-start_time))

        ## Normalize the transcript number in each grid by total count in the cell
        tx_by_cell_grid = pd.concat([sp_adata_cell.obs.loc[:,['grid_array_col','grid_array_row']], 
                                     pd.DataFrame(np.expm1(sp_adata_cell.X.toarray()), 
                                                  index=sp_adata_cell.obs_names, columns=sp_adata_cell.var_names)], axis=1)
        # Generate normalization count matrix by grid
        grid_tx_count = tx_by_cell_grid.groupby(['grid_array_col','grid_array_row']).sum()
        # Saving grid barcode and gene symbol names
        var_names = grid_tx_count.columns
        grid_metadata = grid_tx_count.index.to_frame(name=['array_col','array_row'])
        grid_metadata.index = grid_metadata['array_col'].astype(str) + '_' + grid_metadata['array_row'].astype(str)
        # Log transformation of grid based count
        grid_tx_count = (sparse.csr_matrix(grid_tx_count, dtype=np.float32)).log1p()
        print("End of generating grid-based count matrix: %.2f seconds" % (time.time()-start_time))
        
        # Create dataframe with transcript count according to cell ID and grid number: cell type information added
        tx_by_cell_grid = tx_by_cell_grid.join(df_celltype, how='inner')

        ## Create dataframe with cell type abundance in each grid
        # Create dataframe with transcript count according to cell ID and grid number: cell type information added
        grid_celltype = sp_adata_cell.obs.loc[:,['grid_array_col','grid_array_row']].join(df_celltype, how='inner')
        grid_celltype['count'] = 1
        grid_celltype = grid_celltype.pivot_table(index=['grid_array_col','grid_array_row'], columns=[sc_celltype_colname], values='count', aggfunc=['sum']).fillna(0)
        # Assign column names to the dataframe
        grid_celltype.columns = grid_celltype.columns.to_frame()[sc_celltype_colname]
        # Assign index names to the dataframe
        grid_index = grid_celltype.index.to_frame()
        grid_celltype.index = grid_index['grid_array_col'].astype(str) + '_' + grid_index['grid_array_row'].astype(str)
        # Modify metadata to contain cell type information in each grid
        grid_metadata = grid_metadata.join(grid_celltype, how='left').fillna(0)
        print("End of generating grid-based cell type abundance metadata: %.2f seconds" % (time.time()-start_time))

    ## Generating grid-based image-based ST anndata
    sp_adata_grid = an(X = grid_tx_count, obs=grid_metadata)
    sp_adata_grid.var_names = var_names
    sp_adata_grid.uns['tx_by_cell_grid'] = tx_by_cell_grid.reset_index()
    print("End of generating grid-based image-based ST anndata: %.2f seconds" % (time.time()-start_time))

    return sp_adata_grid, sp_adata_cell



def celltype_specific_mat(sp_adata, grid_method = 'transcript', 
                          tx_info_name='tx_by_cell_grid', celltype_colname=None, cell_types=[''],
                          transcript_colname='target', sc_norm_total=1e3):
    '''
    ## Split and return cell type specific anndata
    ### Input
    sp_adata: grid-based count anndata with cell abundance information saved in .obs
    grid_method: type of the method to assign transcript to grid, either transcript coordinate based method and cell coordinate based method (default='transcript')
    tx_info_name: key name of the transcript information file saved in sp_adata.uns
    celltype_colname: column name of the dataframe; explains which cell type each transcript is beloning to or the cell types for each cell
    cell_types: the cell types to extract cell type-specific count information
    transcript_colname: column name for the transcipt name
    sc_norm_total: scaling factor for the total count normalization per cell

    ### Output
    grid_tx_count_celltype: anndata with celltype specific grid-based count matrix saved in .X
    '''
    if tx_info_name not in sp_adata.uns.keys():
        raise ValueError("'sp_adata.uns' should contain 'tx_info_name' which explains transcript information by cells and grids")

    tx_by_cell_grid = sp_adata.uns[tx_info_name]
    if celltype_colname not in tx_by_cell_grid.columns:
        raise ValueError("'celltype_colname' should be among the column names of dataframe: 'tx_info_name'")
    # Boolean pandas series for the celltype inclusion
    if not (set(cell_types) <= set(tx_by_cell_grid[celltype_colname].cat.categories)):
        raise ValueError("Some of the cell types in 'cell_types' are not found in the sp_data")
    
    if grid_method not in ["transcript","cell"]: 
        raise ValueError("'grid_method' should be either 'transcript' or 'cell'")

    # Subset the transcript information by cell type
    grid_adata_celltype_list = []
    for celltype in cell_types:
        # Find transcript count information for a specific cell type
        tx_by_cell_grid_ = tx_by_cell_grid[tx_by_cell_grid[celltype_colname]==celltype]
        # Generate normalization count matrix by grid
        if grid_method == "transcript":
            grid_tx_count_celltype = tx_by_cell_grid_.pivot_table(index=['array_col','array_row'], columns=transcript_colname, values='tx_fx_by_grid', aggfunc=['sum']).fillna(0)
            # Reindex the grid_tx_count_celltype to match with the original grid index
            grid_index = grid_tx_count_celltype.index.to_frame()
            grid_tx_count_celltype.index = grid_index['array_col'].astype(str) + '_' + grid_index['array_row'].astype(str)
            grid_tx_count_celltype = grid_tx_count_celltype.reindex(sp_adata.obs_names, fill_value=0)
            # Log transformation of grid based count and make sparse matrix
            grid_tx_count_celltype_ = (sc_norm_total*sparse.csr_matrix(grid_tx_count_celltype, dtype=np.float32)).log1p()
            # Create anndata for cell type specific count matrix
            grid_adata_celltype = an(X=grid_tx_count_celltype_, obs=sp_adata.obs[['array_col','array_row']])
            grid_adata_celltype.var_names = grid_tx_count_celltype.columns.to_frame()[transcript_colname].to_numpy()
            grid_adata_celltype_list.append(grid_adata_celltype)
        elif grid_method == "cell":
            grid_tx_count_celltype = tx_by_cell_grid_.drop(columns=['index',celltype_colname]).groupby(['grid_array_col','grid_array_row']).sum()
            # Saving grid barcode and gene symbol names
            grid_index = grid_tx_count_celltype.index.to_frame(name=['array_col','array_row'])
            grid_tx_count_celltype.index = grid_index['array_col'].astype(str) + '_' + grid_index['array_row'].astype(str)
            # Reindex the grid_tx_count_celltype to match with the original grid index
            grid_tx_count_celltype = grid_tx_count_celltype.reindex(sp_adata.obs_names, fill_value=0)
            # Log transformation of grid based count
            grid_tx_count_celltype_ = (sparse.csr_matrix(grid_tx_count_celltype, dtype=np.float32)).log1p()
            # Create anndata for cell type specific count matrix
            grid_adata_celltype = an(X=grid_tx_count_celltype_, obs=sp_adata.obs[['array_col','array_row']])
            grid_adata_celltype.var_names = grid_tx_count_celltype.columns.values
            grid_adata_celltype_list.append(grid_adata_celltype)

    return grid_adata_celltype_list