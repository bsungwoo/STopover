import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import bin2cell as b2c

from scipy import sparse
from anndata import AnnData as an
from .imageST_utils import annotate_ST

import warnings
warnings.filterwarnings("ignore")


def bin2cell_process(bin_path="binned_outputs/square_002um/", source_image_path = "Visium_HD_Mouse_Brain_tissue_image.tif",
                     spaceranger_image_path = "spatial", min_cells = 3, min_counts = 1, mpp = 0.5, 
                     prob_thresh_hne = 0.01, prob_thresh_gex = 0.05, nms_thresh = 0.5, sigma = 5,
                     mask_arr_row_min = 1450, mask_arr_row_max = 1550, mask_arr_col_min = 250, mask_arr_col_max = 450,
                     show_plot = False, save_path='.'):
    """
    ## Function to preprocess visiumHD datasets
    
    ### Input
    * bin_path (str, optional): path to the 2 micrometers binned output. Defaults to "binned_outputs/square_002um/".
    * source_image_path (str, optional): path to the source image. Defaults to "Visium_HD_Mouse_Brain_tissue_image.tif".
    * spaceranger_image_path (str, optional): path to the spaceranger image. Defaults to "spatial".
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
    * save_path (str, optional): path to save the results. Defaults to '.'.
    
    ### Output
    
    """
    # Create directory for stardist input/output files
    os.makedirs(os.path.join(save_path, "stardist"), exist_ok=True)
    # Read visium HD files
    adata = b2c.read_visium(bin_path, source_image_path = source_image_path, 
                            spaceranger_image_path = spaceranger_image_path)
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    
    # Process H&E images and save on anndata
    # Define a mask to easily pull out this region of the object in the future
    mask = ((adata.obs['array_row'] >= mask_arr_row_min) & 
            (adata.obs['array_row'] <= mask_arr_row_max) & 
            (adata.obs['array_col'] >= mask_arr_col_min) & 
            (adata.obs['array_col'] <= mask_arr_col_max)
        )
    # The function crops the image to an area around the actual spatial grid present in the object, and the new coordinates are captured in .obsm["spatial_cropped"]
    b2c.scaled_he_image(adata, mpp=mpp, save_path=os.path.join(save_path, "stardist/he.tiff"))
    # Destripping by scaling the count matrix
    b2c.destripe(adata)
    # Plot the image
    if show_plot:
        bdata = adata[mask]
        sc.pl.spatial(bdata, color=[None, "n_counts", "n_counts_adjusted"], img_key=str(mpp)+"_mpp", basis="spatial_cropped")
    # Run stardist function
    b2c.stardist(image_path=os.path.join(save_path, "stardist/he.tiff"), 
                 labels_npz_path=os.path.join(save_path, "stardist/he.npz"), 
                 stardist_model="2D_versatile_he", 
                 prob_thresh=prob_thresh_hne)
    # Load segmentation results on the anndata
    b2c.insert_labels(adata, 
                      labels_npz_path=os.path.join(save_path, "stardist/he.npz"), 
                      basis="spatial", 
                      spatial_key="spatial_cropped",
                      mpp=mpp, 
                      labels_key="labels_he")
    # Expand the labels to contain cells
    b2c.expand_labels(adata, labels_key='labels_he', 
                      expanded_labels_key="labels_he_expanded")
    if show_plot:
        bdata = adata[mask]
        #the labels obs are integers, 0 means unassigned
        bdata = bdata[bdata.obs['labels_he_expanded']>0]
        bdata.obs['labels_he_expanded'] = bdata.obs['labels_he_expanded'].astype(str)
        sc.pl.spatial(bdata, color=[None, "labels_he_expanded"], img_key=str(mpp)+"_mpp", basis="spatial_cropped")
        
    # Convert total count to image and segment cells
    b2c.grid_image(adata, "n_counts_adjusted", mpp=mpp, sigma=sigma, 
                   save_path=os.path.join(save_path, "stardist/gex.tiff"))
    b2c.stardist(image_path=os.path.join(save_path, "stardist/gex.tiff"), 
                 labels_npz_path=os.path.join(save_path, "stardist/gex.npz"), 
                 stardist_model="2D_versatile_fluo", 
                 prob_thresh=prob_thresh_gex, nms_thresh=nms_thresh)
    if show_plot:
        bdata = adata[mask]
        #the labels obs are integers, 0 means unassigned
        bdata = bdata[bdata.obs['labels_gex']>0]
        bdata.obs['labels_gex'] = bdata.obs['labels_gex'].astype(str)
        sc.pl.spatial(bdata, color=[None, "labels_gex"], img_key=str(mpp)+"_mpp", basis="spatial_cropped")

    # Combine H&E + Total count based segmentation results (Add GEX cells that do not overlap with any H&E labels)
    b2c.salvage_secondary_labels(adata, primary_label="labels_he_expanded", 
                                 secondary_label="labels_gex", labels_key="labels_joint")
    if show_plot:
        bdata = adata[mask]
        #the labels obs are integers, 0 means unassigned
        bdata = bdata[bdata.obs['labels_joint']>0]
        bdata.obs['labels_joint'] = bdata.obs['labels_joint'].astype(str)
        sc.pl.spatial(bdata, color=[None, "labels_joint_source", "labels_joint"], img_key=str(mpp)+"_mpp", basis="spatial_cropped")
        
    # Group the bins into cells
    cdata = b2c.bin_to_cell(adata, labels_key="labels_joint", spatial_keys=["spatial", "spatial_cropped"])
    if show_plot:
        cell_mask = ((cdata.obs['array_row'] >= mask_arr_row_min) & 
                    (cdata.obs['array_row'] <= mask_arr_row_max) & 
                    (cdata.obs['array_col'] >= mask_arr_col_min) & 
                    (cdata.obs['array_col'] <= mask_arr_col_max))
        ddata = cdata[cell_mask]
        sc.pl.spatial(ddata, color=["bin_count", "labels_joint_source"], img_key=str(mpp)+"_mpp", basis="spatial_cropped")
    
    return cdata
    
    
def read_visiumHD(bin_path="binned_outputs/square_002um/", source_image_path = "Visium_HD_Mouse_Brain_tissue_image.tif",
                  spaceranger_image_path = "spatial", 
                  sc_adata=None, sc_celltype_colname = 'celltype', 
                  annot_method='tacco', sc_norm_total=1e3, x_grid_size=100, y_grid_size=100,
                  min_cells = 3, min_counts = 1, bin_counts = 5, mpp = 0.5, 
                  prob_thresh_hne = 0.01, prob_thresh_gex = 0.05, nms_thresh = 0.5, sigma = 5,
                  mask_arr_row_min = 1450, mask_arr_row_max = 1550, mask_arr_col_min = 250, mask_arr_col_max = 450,
                  show_plot = False, save_path='.'):
    '''
    ## Load visiumHD dataset and preprocess data
    ### Input
    * bin_path (str, optional): path to the 2 micrometers binned output. Defaults to "binned_outputs/square_002um/".
    * source_image_path (str, optional): path to the source image. Defaults to "Visium_HD_Mouse_Brain_tissue_image.tif".
    * spaceranger_image_path (str, optional): path to the spaceranger image. Defaults to "spatial".
    
    * sc_adata: single-cell reference anndata for cell type annotation of visiumHD data.
        -> raw count matrix should be saved in .X
        -> If None, then leiden cluster numbers will be used to annotate visiumHD data.
    * sc_celltype_colname: column name for cell type annotation information in metadata of single-cell (.obs)
    * annot_method: cell type annotation methods: either 'ingest' or 'tacco' (default='tacco')
    * sc_norm_total: scaling factor for the total count normalization per cell 
    * x_grid_size, y_grid_size: size of the grid in x- and y-direction in number of bins (2 micron if 2 micron bin is used) to divide the visiumHD data (for grid-based aggregation)

    * min_cells: minimum number of counts required for a cell to pass filtering (scanpy.pp.filter_cells). Defaults to 3.
    * min_counts: minimum number of cells expressed required for a gene to pass filtering (scanpy.pp.filter_genes). Defaults to 1.
    * bin_counts: minimum count value inside of cells required for a cell to psas filtering. Default is 5.
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
    * save_path (str, optional): path to save the results. Defaults to '.'.

    ### Output
    sp_adata_grid: grid-based log-normalized count anndata with cell abundance information saved in .obs
    sp_adata_cell: cell-based log-normalized count anndata
    '''
    start_time = time.time()

    # Generate image-based ST anndata file
    sp_adata_cell = bin2cell_process(bin_path=bin_path, source_image_path = source_image_path, spaceranger_image_path = spaceranger_image_path,
                                     min_cells = min_cells, min_counts = min_counts, mpp = mpp, 
                                     prob_thresh_hne = prob_thresh_hne, prob_thresh_gex = prob_thresh_gex, nms_thresh = nms_thresh, sigma = sigma,
                                     mask_arr_row_min = mask_arr_row_min, mask_arr_row_max = mask_arr_row_max, mask_arr_col_min = mask_arr_col_min, mask_arr_col_max = mask_arr_col_max,
                                     show_plot = show_plot, save_path = save_path)
    sp_adata_cell = sp_adata_cell[sp_adata_cell.obs['bin_count'] > bin_counts]
    sp_adata_cell.obs['cell_id'] = sp_adata_cell.obs_names
    # Make counts integer for downstream analysis
    sp_adata_cell.X.data = np.round(sp_adata_cell.X.data)
    print("End of creating cell-level anndata for visiumHD using bin2cell: %.2f seconds" % (time.time()-start_time))

    ## Annotation of cell-level image-based ST data
    sp_adata_cell, df_celltype = annotate_ST(sp_adata_cell, sc_adata, sc_norm_total=sc_norm_total, 
                                             sc_celltype_colname = sc_celltype_colname, annot_method=annot_method,
                                             cell_id = 'cell_id', return_df=True)
    print("End of annotating image-based ST cell-level anndata: %.2f seconds" % (time.time()-start_time))

    ## Grid-based aggregation of image-based ST: divide coordinates by x_bins and y_bins and aggregate
    # Find the x and y coordinate arrays
    x_coord = sp_adata_cell.obs[['array_col']].to_numpy()
    y_coord = sp_adata_cell.obs[['array_row']].to_numpy()
    # Find the coordinates that equally divides the x and y axis into x_bins and y_bins
    x_div_arr = np.linspace(np.min(x_coord), np.max(x_coord), num=((x_coord.max()-x_coord.min())//x_grid_size), endpoint=False)[1:]
    y_div_arr = np.linspace(np.min(y_coord), np.max(y_coord), num=((y_coord.max()-y_coord.min())//y_grid_size), endpoint=False)[1:]
    # Assigning the grid column and row number to each transcript based on the coordinates by x_div_arr and y_div_arr
    sp_adata_cell.obs['grid_array_col'] = np.searchsorted(x_div_arr, x_coord, side='right')
    sp_adata_cell.obs['grid_array_row'] = np.searchsorted(y_div_arr, y_coord, side='right')
    print("End of grid-based aggregation of visiumHD: %.2f seconds" % (time.time()-start_time))

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
    print("End of generating grid-based visiumHD anndata: %.2f seconds" % (time.time()-start_time))

    return sp_adata_grid, sp_adata_cell