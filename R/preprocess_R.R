#' Install and load miniconda environment
#' @description Install and load miniconda environment for implementation of STopover
#' @param conda.env.name name of the conda environment to use for STopover analysis (default: 'STopover')
#' @export
install_load_env <- function(conda.env.name='STopover'){
  if (!reticulate:::is_python_initialized()){
    try(reticulate::install_miniconda(), silent=T)
    # Setting virtual environment with reticulate
    if (!(conda.env.name %in% reticulate::conda_list()[['name']])){
      reticulate::conda_create(envname = conda.env.name, python_version = '3.8.12')
      # Create conda env and install dependencies
      reticulate::conda_install(conda.env.name, ignore_installed=T,
                                pip = T, "git+https://github.com/bsungwoo/STopover.git")
    }
    reticulate::use_condaenv(conda.env.name, required = T)
  }
}


#' Convert Seurat object to anndata object
#' @description Convert spatial transcriptomic data from Seurat to anndata
#' @param sp_object spatial data (Seurat object) to be used in converting
#' @param features a vector of features to convert to anndata: either gene symbols or column names for metadata (default = NULL)
#' @param conda.env.name name of the conda environment to use for STopover analysis (default = 'STopover')
#' @param assay assay to extract sparse matrix in Seurat object (default = 'Spatial')
#' @param slot slot to extract sparse matrix in Seurat object (default = 'data')
#' @param add_coord whether to add coordinate information in the .obs of anndata (default = F)
#' @param add_uns named list to add in the .uns of anndata (default = list())
#' @return Anndata object that contains the count data of Seurat object (for the specific assay and slot) in .X and coordinates and metadata in .obs.
#' @export
convert_to_anndata <- function(sp_object, features=NULL, conda.env.name='STopover',
                               assay='Spatial', slot='data', add_coord=F, add_uns=list()){
  if (is.null(features)) cat("'features' is NULL: using all features in Seurat object\n")

  # Install and load environment
  install_load_env(conda.env.name)
  ## Import anndata
  ann <- reticulate::import('anndata', convert = FALSE)
  ## Import STopover
  STopover <- reticulate::import('STopover', convert = FALSE)

  ## Convert Seurat object to sparse matrix format
  if (!is.null(features)) {
    gene_intersect <- intersect(features, rownames(sp_object))
    meta_intersect <- intersect(features, colnames(sp_object@meta.data))
    if (identical(gene_intersect, character(0))){
      sparse_mtx <- t(data.frame(row.names = colnames(sp_object)))
    } else {
      sparse_mtx <- Seurat::GetAssayData(sp_object, slot = slot, assay = assay)[gene_intersect, ]
    }
    if (identical(meta_intersect, character(0))){
      obs <- data.frame(row.names = colnames(sp_object))
    } else {
      obs <- sp_object@meta.data[meta_intersect]
    }
  } else {
    sparse_mtx <- Seurat::GetAssayData(sp_object, slot = slot, assay = assay)
    obs <- sp_object@meta.data
  }
  # Define var (reference from sceasy library)
  var <- data.frame(row.names = rownames(sparse_mtx))

  # Extract coordinates of the spots or grids
  if (add_coord){
    if (assay!='Spatial'){warning("Coordinates in the assay '",assay,"' will be used.")}
    df_coord <- data.frame()
    for (slide_name in names(sp_object@images)){
      df_image <- sp_object@images[[slide_name]]@coordinates[,c("col","row")]
      df_image[['batch']] <- slide_name
      df_coord <- rbind(df_coord, df_image)
    }
    colnames(df_coord) <- c("array_col","array_row","batch")
    obs <- cbind(df_coord, obs)
  }
  # Create anndata object for analysis
  adata <- ann$AnnData(
    X = Matrix::t(sparse_mtx),
    obs = obs,
    var = var,
    uns = add_uns
  )
  return(adata)
}


#' Load CosMx SMI file as Seurat object (as a Slideseq class)
#' @description Convert Seurat object for spatial transcriptomic data to anndata
#' @param sp_load_path path to CosMx SMI data directory or .h5ad Anndata object (default = '.')
#' @param sc_object single-cell reference anndata/path to reference "*h5ad" file/Seurat object for cell type annotation of CosMx SMI data (default = NULL)
#' \itemize{
#'    \item If NULL, then Leiden cluster numbers will be used to annotate CosMx SMI data
#' }
#' @param conda.env.name name of the conda environment to use for STopover analysis (default = 'STopover')
#' @param sc_celltype_colname column name for cell type annotation information in metadata of single-cell (default = 'celltype')
#' @param sc_norm_total scaling factor for the total count normalization per cell (default = 1e3)
#' @param tx_file_name CosMx file for transcript count (default = 'tx_file.csv')
#' @param cell_exprmat_file_name CosMx file for cell-level expression matrix (default = 'exprMat_file.csv')
#' @param cell_metadata_file_name CosMx file for cell-level metadata (default = 'metadata_file.csv')
#' @param fov_colname column name for barcodes corresponding to fov (default = 'fov')
#' @param cell_id_colname column name for barcodes corresponding to cell ID (default = 'cell_ID')
#' @param tx_xcoord_colname column name for global x coordinates of the transcript (default = 'x_global_px')
#' @param tx_ycoord_colname column name for global y coordinates of the transcript (default = 'y_global_px')
#' @param transcript_colname column name for the transcript name (default = 'target')
#' @param meta_xcoord_colname column name for global x, y coordinates in cell-level metadata file (default = 'CenterX_global_px')
#' @param meta_ycoord_colname column name for global x, y coordinates in cell-level metadata file (default = 'CenterY_global_px')
#' @param x_bins number of bins to divide the CosMx SMI data (for grid-based aggregation) (default = 100)
#' @param y_bins number of bins to divide the CosMx SMI data (for grid-based aggregation) (default = 100)
#' @return list of Seurat objects for grid-based and cell-level data, sequentially. The objects contains grid-based or cell-level log-normalized count matrix in 'counts & data' slots, coordinates in '@images$image', and transcript information in '@assays$Spatial@misc' (only for grid-based data)
#' @export
preprocess_cosmx <- function(sp_load_path='.', sc_object=NULL, conda.env.name='STopover',
                             sc_celltype_colname = 'celltype', sc_norm_total=1e3,
                             tx_file_name = 'tx_file.csv', cell_exprmat_file_name='exprMat_file.csv',
                             cell_metadata_file_name='metadata_file.csv',
                             fov_colname = 'fov', cell_id_colname='cell_ID',
                             tx_xcoord_colname='x_global_px', tx_ycoord_colname='y_global_px',
                             transcript_colname='target',
                             meta_xcoord_colname='CenterX_global_px',
                             meta_ycoord_colname='CenterY_global_px',
                             x_bins=100, y_bins=100){
  # Install and load environment
  install_load_env(conda.env.name)
  ## Import anndata
  ann <- reticulate::import('anndata', convert = FALSE)
  ## Import STopover
  STopover <- reticulate::import('STopover', convert = FALSE)

  # Check format of the single-cell dataset if it exists
  cat("Reading CosMx SMI data: annotating cells and creating grid-based data\n")
  if (typeof(sc_object)=="character"){
    cosmx_output_dir <- file.path(getwd(),"cosmx_output")
    if (!file.exists(cosmx_output_dir)) dir.create(cosmx_output_dir)
    STopover_cosmx_dir <- system.file("preprocess_cosmx_R.py", package="STopover")
    string_output <- system(paste0("python ", STopover_cosmx_dir,
                                   " --sp_load_path",sp_load_path," --sc_load_path",sc_object,
                                   " --sc_celltype_colname ",sc_celltype_colname,
                                   " --sc_norm_total ",sc_norm_total,
                                   " --tx_file_name ",tx_file_name,
                                   " --cell_exprmat_file_name ",cell_exprmat_file_name,
                                   " --cell_metadata_file_name",cell_metadata_file_name,
                                   " --fov_colname",fov_colname,
                                   " --cell_id_colname",cell_id_colname,
                                   " --tx_xcoord_colname",tx_xcoord_colname,
                                   " --tx_ycoord_colname",tx_ycoord_colname,
                                   " --transcript_colname",transcript_colname,
                                   " --meta_xcoord_colname",meta_xcoord_colname,
                                   " --meta_ycoord_colname",meta_ycoord_colname,
                                   " --x_bins",x_bins," --y_bins",y_bins,
                                   " --save_path",cosmx_output_dir), intern = T)
    cat(paste0(string_output,"\n"))
  } else {
    if (typeof(sc_object)=="environment"|is.null(sc_object)){
      adata_sc <- sc_object
    } else {
      if (!sc_celltype_colname %in% colnames(sc_object@meta.data)){stop("'sc_celltype_colname' not among the metadata column names")}
      cat("Converting Seurat object to anndata\n")
      ## Convert Seurat single-cell object to sparse matrix format
      adata_sc <- convert_to_anndata(sc_object, assay='RNA', slot='counts', add_coord=F)
    }
    adata_sp_all <- STopover$STopover_cosmx(sp_load_path=sp_load_path, sc_adata=adata_sc,
                                            sc_celltype_colname = sc_celltype_colname,
                                            sc_norm_total=sc_norm_total,
                                            tx_file_name = tx_file_name,
                                            cell_exprmat_file_name=cell_exprmat_file_name,
                                            cell_metadata_file_name=cell_metadata_file_name,
                                            fov_colname = fov_colname, cell_id_colname=cell_id_colname,
                                            tx_xcoord_colname=tx_xcoord_colname,
                                            tx_ycoord_colname=tx_ycoord_colname,
                                            transcript_colname=transcript_colname,
                                            meta_xcoord_colname=meta_xcoord_colname,
                                            meta_ycoord_colname=meta_ycoord_colname,
                                            x_bins=as.integer(x_bins), y_bins=as.integer(y_bins))
  }

  # Create Seurat object for grid-based and cell-level cosmx data
  cat("Creating Seurat object for grid-based and cell-level CosMx data\n")
  sp_object_list <- list()
  for (idx in 1:2){
    if (idx==2) {adata <- adata_sp_all$uns['adata_cell']}
    else {adata <- adata_sp_all}
    sparse_mtx <- reticulate::py_to_r(adata$X$T)
    colnames(sparse_mtx) <- reticulate::py_to_r(adata$obs_names$values)
    rownames(sparse_mtx) <- reticulate::py_to_r(adata$var_names$values)
    sp_object_list[[idx]] <- Seurat::CreateSeuratObject(counts = sparse_mtx, assay = "Spatial")
    sp_object_list[[idx]][['nCount_Spatial']] <- NULL
    # Assign coordinates of the cell
    df_coord = data.frame(row=reticulate::py_to_r(adata$obs['array_row']),
                          col=reticulate::py_to_r(adata$obs['array_col']),
                          stringsAsFactors=F)
    # Invert the top-bottom of the coordinate
    df_coord['row'] <- max(df_coord['row']) - df_coord['row']
    rownames(df_coord) <- reticulate::py_to_r(adata$obs_names$values)
    if (idx==2) {
      df_celltype <- data.frame(celltype = factor(reticulate::py_to_r(adata$obs[sc_celltype_colname]$astype('object'))))
      rownames(df_celltype) = reticulate::py_to_r(adata$obs_names$values)
      sp_object_list[[idx]] <- Seurat::AddMetaData(sp_object_list[[idx]], df_celltype)
      Seurat::Idents(sp_object_list[[idx]]) <- sc_celltype_colname
    } else {
      sp_object_list[[idx]] <- Seurat::AddMetaData(sp_object_list[[idx]], reticulate::py_to_r(adata$obs))
      sp_object_list[[idx]][['batch']] <- "image"
      sp_object_list[[idx]]@assays$Spatial@misc <- reticulate::py_to_r(adata$uns['tx_by_cell_grid'])
    }
    sp_object_list[[idx]]@images$image = methods::new(Class = 'SlideSeq', assay = "Spatial", key = "image_", coordinates = df_coord)
  }
  return(sp_object_list)
}


#' Extract cell type specific transcript count matrix and create Seurat object in cosmx data
#' @param sp_object spatial data (Seurat object) containing CosMx SMI data to be used in calculating cell type-specific count matrix: non-normalized raw data should be in 'counts' slot
#' @param cell_types the cell types to extract cell type-specific count information
#' @param conda.env.name name of the conda environment to use for STopover analysis (default = 'STopover')
#' @param fov_colname column name for barcodes corresponding to fov (default = 'fov')
#' @param cell_id_colname column name for barcodes corresponding to cell ID (default = 'cell_ID')
#' @param celltype_colname column name for cell type annotation information saved in 'sp_object@assays$Spatial@misc' (default = 'celltype')
#' @param transcript_colname column name for the transcript name (default = 'target')
#' @param sc_norm_total scaling factor for the total count normalization per cell (default = 1e3)
#' @param return_mode what kind of data to return
#' \itemize{
#'    \item 'seurat': return Seurat object that contains cell type-specific count matrix
#'    \item 'anndata': return Anndata object that contains cell type-specific count matrix
#' }
#' @return spatial data with cell type-specific count information in each grid
#' @export
celltype_specific_cosmx <- function(sp_object, cell_types, conda.env.name='STopover',
                                    fov_colname = 'fov', cell_id_colname='cell_ID',
                                    celltype_colname='celltype', transcript_colname='target',
                                    sc_norm_total=1e3, return_mode='seurat'){
  if (class(sp_object@images$image)[1]!="SlideSeq") {stop("'sp_object' should be cosmx spatial data object")}
  ifelse(return_mode %in% c('anndata','seurat'),NULL,stop("'return mode' should be either 'anndata' or 'seurat'"))

  # Install and load environment
  install_load_env(conda.env.name)
  ## Import STopover
  STopover <- reticulate::import('STopover', convert = FALSE)
  pd <- reticulate::import('pandas', convert = FALSE)

  # Extract transcript by cell grid and save in uns
  tx_by_cell_grid <- data.frame(sp_object@assays$Spatial@misc)
  levels(tx_by_cell_grid[[celltype_colname]]) <- make.names(levels(tx_by_cell_grid[[celltype_colname]]))
  tx_by_cell_grid[['array_col']] <- as.integer(tx_by_cell_grid[['array_col']]) # change to integer to prevent error
  tx_by_cell_grid[['array_row']] <- as.integer(tx_by_cell_grid[['array_row']])
  adata_sp <- convert_to_anndata(sp_object, assay='Spatial', slot='data',
                                 add_coord=F, # coordinate in already in the metadata
                                 add_uns=list('tx_by_cell_grid'=tx_by_cell_grid))

  # Calculate cell type-specific count matrix
  if (length(cell_types)==1){cell_types <- list(cell_types)}
  grid_count_celltype_list <- STopover$celltype_specific_mat(adata_sp, tx_info_name='tx_by_cell_grid',
                                                             celltype_colname=celltype_colname,
                                                             cell_types=cell_types,
                                                             transcript_colname=transcript_colname,
                                                             sc_norm_total=sc_norm_total)
  grid_count_celltype_list_ <- list()
  if (return_mode=='seurat'){
    for (idx in 1:length(grid_count_celltype_list)){
      sparse_mtx <- reticulate::py_to_r(grid_count_celltype_list[[(idx-1)]]$X$T)
      colnames(sparse_mtx) <- reticulate::py_to_r(grid_count_celltype_list[[(idx-1)]]$obs_names$values)
      rownames(sparse_mtx) <- reticulate::py_to_r(grid_count_celltype_list[[(idx-1)]]$var_names$values)
      grid_count_celltype_list_[[idx]] <- Seurat::CreateSeuratObject(counts = sparse_mtx, assay = "Spatial")
      grid_count_celltype_list_[[idx]][['nCount_Spatial']] <- NULL
      # Assign coordinates of the cell
      df_coord = data.frame(row=reticulate::py_to_r(grid_count_celltype_list[[(idx-1)]]$obs['array_row']),
                            col=reticulate::py_to_r(grid_count_celltype_list[[(idx-1)]]$obs['array_col']),
                            stringsAsFactors=F)
      rownames(df_coord) = reticulate::py_to_r(grid_count_celltype_list[[(idx-1)]]$obs_names$values)
      grid_count_celltype_list_[[idx]]@images$image = methods::new(Class = 'SlideSeq', assay = "Spatial", key = "image_", coordinates = df_coord)
    }
    return(grid_count_celltype_list_)
  } else {
    return(list(grid_count_celltype_list[[0]], grid_count_celltype_list[[1]]))
  }
}


#' Extract CellTalkDB as dataframe
#' @param lr_db_species select species to utilize in CellTalkDB database (default = 'human')
#' @param conda.env.name name of the conda environment to use for STopover analysis (default = 'STopover')
#' @export
return_celltalkdb <- function(lr_db_species='human', conda.env.name='STopover'){
  # Install and load environment
  install_load_env(conda.env.name)
  ## Import python modules
  STopover <- reticulate::import('STopover', convert = FALSE)
  ann <- reticulate::import('anndata', convert = FALSE)

  adata_null <- ann$AnnData(X = matrix(0))
  adata_null <- STopover$STopover_visium(sp_adata=adata_null)
  df <- adata_null$return_celltalkdb(lr_db_species=lr_db_species)

  return(reticulate::py_to_r(df))
}
