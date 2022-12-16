#' Calculate topological overlap of feature pairs and return Jaccard dataframe
#' @description Calculate Jaccard index between topological connected components of feature pairs and return connected component location in meta.data and dataframe with Jaccard indexes in misc
#' @param sp_object spatial data (Seurat object) to be used in calculating topological similarity
#' \itemize{
#'   \item select which slot to use and whether to perform log-normalization before STopover analysis
#'   \item if log-normalized data is included in 'data' slot, then set slot as 'data' and 'lognorm' as F, which are default values
#' }
#' @param feat_pairs list of features with the format [('A','B'),('C','D')] or the pandas equivalent (default = data.frame())
#' \itemize{
#'   \item (A and C) should to be same data format: all in metadata (.obs.columns) or all in gene names (.var.index).
#'   \item (C and D) should be same data format: all in metadata (.obs.columns) or all in gene names (.var.index).
#'   \item If the data format is not same the majority of the data format will be automatically searched and the rest of the features with different format will be removed from the pairs (default = '.').
#' }
#' @param use_lr_db whether to use list of features in CellTalkDB L-R database (default = F)
#' @param lr_db_species select species to utilize in CellTalkDB database (default = 'human')
#' @param min_size minimum size of a connected component: number of spots/grids (default = 20)
#' @param fwhm full width half maximum value for the Gaussian smoothing kernel as the multiple of the central distance between the adjacent spots (default = 2.5)
#' @param thres_per lower percentile value threshold to remove the connected components (default = 30)
#' @param conda.env.name name of the conda environment to use for STopover analysis (default = 'STopover')
#' @param assay assay to extract sparse matrix in Seurat object (default = 'Spatial')
#' @param slot slot to extract sparse matrix in Seurat object (default = 'data': expecting lognormalized matrix in 'data')
#' @param lognorm whether to lognormalize the sparse matrix in the slot before calculating topological similarity (default = F)
#' @param J_result_name the name of the jaccard index data file name (default = 'result')
#' @param num_workers number of the workers for the multiprocessing (default = NULL)
#' \itemize{
#'   \item if NULL, the maximum number of CPUs will be used for the multiprocessing
#' }
#' @return spatial data (Seurat object) with connected component location described as integers in metadata (meta.data)
#' @export
topological_similarity <- function(sp_object, feat_pairs=data.frame(),
                                   use_lr_db=F, lr_db_species='human',
                                   min_size=20, fwhm=2.5, thres_per=30,
                                   conda.env.name="STopover",
                                   assay='Spatial', slot='data', lognorm=F,
                                   J_result_name='result', num_workers=NULL){
  # if (dim(feat_pairs)[2]!=2){stop("There should be two columns in 'feat_pairs'")}
  # Check the data type
  spatial_type <- ifelse(class(sp_object@images$image)[1]=="SlideSeq","cosmx","visium")
  # Install and load environment
  install_load_env(conda.env.name)
  ## Import STopover
  STopover <- reticulate::import('STopover', convert = FALSE)

  ## Convert Seurat spatial object to sparse matrix format
  adata_sp <- convert_to_anndata(sp_object, features = unlist(feat_pairs),
                                 assay=assay, slot=slot, add_coord=T)

  # Preprocess spatial data and run topological analysis
  try({
    if (spatial_type=="cosmx"){
      stopover_object <- STopover$STopover_cosmx(sp_adata=adata_sp, min_size=min_size, fwhm=fwhm, thres_per=thres_per)
    } else {
      stopover_object <- STopover$STopover_visium(sp_adata=adata_sp, min_size=min_size, fwhm=fwhm, thres_per=thres_per, lognorm=lognorm)
    }
    group_list <- names(sp_object@images)
    if (length(group_list)==1){group_list <- list(group_list)}
    # Calculate topological similarity between two features
    if (is.null(num_workers)){
      stopover_object$topological_similarity(feat_pairs=feat_pairs,
                                             use_lr_db=use_lr_db,
                                             lr_db_species=lr_db_species,
                                             group_name='batch', group_list=group_list,
                                             J_result_name = J_result_name)
    } else {
      stopover_object$topological_similarity(feat_pairs=feat_pairs,
                                             use_lr_db=use_lr_db,
                                             lr_db_species=lr_db_species,
                                             group_name='batch', group_list=group_list,
                                             J_result_name = J_result_name, num_workers = num_workers)
    }
    # Saving CC location data into the metadata of spatial Seurat object
    cc_loc_df <- reticulate::py_to_r(stopover_object$obs)
    cc_loc_names <- grep(colnames(cc_loc_df), pattern = "Comb_CC_", value=T)
    cc_loc_df <- cc_loc_df[,cc_loc_names]
    sp_object <- Seurat::AddMetaData(sp_object, cc_loc_df)
    # Saving jaccard index result into the @misc of spatial Seurat object
    sp_object@misc <- reticulate::py_to_r(stopover_object$uns[[paste(c('J',J_result_name,0),collapse="_")]])
  })

  return(sp_object)
}


#' Calculate spatial overlap of feature pairs from a specific cell type pair and return Jaccard dataframe
#' @description Calculate spatial overlap between topological connected components of feature pairs from a specific cell type pair and return connected component location in meta.data and dataframe with Jaccard indexes in misc
#' @param sp_object spatial data (Seurat object) to be used in calculating topological similarity
#' \itemize{
#'   \item select which slot to use and whether to perform log-normalization before STopover analysis
#'   \item if log-normalized data is included in 'data' slot, then set slot as 'data' and 'lognorm' as F, which are default values
#' }
#' @param celltype_x cell type corresponding to the first feature (default = '')
#' @param celltype_y cell type corresponding to the second feature (default = '')
#' @param feat_pairs list of features with the format [('A','B'),('C','D')] or the pandas equivalent (default = data.frame())
#' \itemize{
#'   \item (A and C) should to be same data format: all in metadata (.obs.columns) or all in gene names (.var.index).
#'   \item (C and D) should be same data format: all in metadata (.obs.columns) or all in gene names (.var.index).
#'   \item If the data format is not same the majority of the data format will be automatically searched and the rest of the features with different format will be removed from the pairs (default = '.').
#' }
#' @param use_lr_db whether to use list of features in CellTalkDB L-R database (default = F)
#' @param lr_db_species select species to utilize in CellTalkDB database (default = 'human')
#' @param min_size minimum size of a connected component: number of spots/grids (default = 20)
#' @param fwhm full width half maximum value for the Gaussian smoothing kernel as the multiple of the central distance between the adjacent spots (default = 2.5)
#' @param thres_per lower percentile value threshold to remove the connected components (default = 30)
#' @param conda.env.name name of the conda environment to use for STopover analysis (default = 'STopover')
#' @param assay assay to extract sparse matrix in Seurat object (default = 'Spatial')
#' @param slot slot to extract sparse matrix in Seurat object (default = 'data': expecting lognormalized matrix in 'data')
#' @param fov_colname column name for barcodes corresponding to fov (default = 'fov')
#' @param cell_id_colname column name for barcodes corresponding to cell ID (default = 'cell_ID')
#' @param celltype_colname column name for cell type annotation information saved in 'sp_object@assays$Spatial@misc' (default = 'celltype')
#' @param transcript_colname column name for the transcript name (default = 'target')
#' @param sc_norm_total scaling factor for the total count normalization per cell (default = 1e3)
#' @param assay assay to extract sparse matrix in Seurat object while calculating topological similarity between cell types in visium dataset (default = 'Spatial')
#' @param slot slot to extract sparse matrix in Seurat object while calculating topological similarity between cell types in visium dataset (default = 'data')
#' @param lognorm whether to lognormalize the sparse matrix in the slot before calculating topological similarity in visium dataset (default = F)
#' @param J_result_name the name of the jaccard index data file name (default = 'result')
#' @param num_workers number of the workers for the multiprocessing (default = NULL)
#' \itemize{
#'   \item if NULL, the maximum number of CPUs will be used for the multiprocessing
#' }
#' @return spatial data (Seurat object) with connected component location described as integers in metadata (meta.data)
#' @export
topological_similarity_celltype_pair <- function(sp_object, celltype_x='',celltype_y='',
                                                 feat_pairs=data.frame(),
                                                 use_lr_db=F, lr_db_species='human',
                                                 min_size=20, fwhm=2.5, thres_per=30,
                                                 conda.env.name='STopover',
                                                 fov_colname = 'fov', cell_id_colname='cell_ID',
                                                 celltype_colname='celltype', transcript_colname='target',
                                                 sc_norm_total=1e3,
                                                 assay='Spatial', slot='data',lognorm=F,
                                                 J_result_name='result', num_workers=NULL){
  spatial_type <- class(sp_object@images$image)[1]
  # if (dim(feat_pairs)[2]!=2){stop("There should be two columns in 'feat_pairs'")}
  # Install and load environment
  install_load_env(conda.env.name)
  ann <- reticulate::import('anndata', convert = FALSE)
  scipy <- reticulate::import('scipy', convert = FALSE)
  STopover <- reticulate::import('STopover', convert = FALSE)

  if (spatial_type=='visium'){
    # Calculate connected components for feature pairs
    sp_object_xy <- topological_similarity(sp_object,
                                           feat_pairs=feat_pairs,
                                           use_lr_db = use_lr_db, lr_db_species = lr_db_species,
                                           min_size=min_size, fwhm=fwhm, thres_per=thres_per,
                                           conda.env.name=conda.env.name,
                                           assay=assay, slot=slot, lognorm=lognorm,
                                           J_result_name=J_result_name, num_workers=num_workers)
    # Calculate connected components for celltype pairs
    sp_object_celltype <- topological_similarity(sp_object,
                                                 feat_pairs=data.frame("X"=celltype_x, "Y"=celltype_y),
                                                 min_size=min_size, fwhm=fwhm, thres_per=thres_per,
                                                 conda.env.name=conda.env.name,
                                                 assay=assay, slot=slot, lognorm=lognorm,
                                                 J_result_name=J_result_name, num_workers=num_workers)
    df_over = (sp_object_celltype[[paste0('Comb_CC_',celltype_x)]] != 0) * (sp_object_celltype[[paste0('Comb_CC_',celltype_y)]] != 0)
    colnames(df_over) = "cell_over"
    sp_object_xy <- Seurat::AddMetaData(sp_object_xy, df_over)
    Seurat::Idents(sp_object_xy) <- "cell_over"
    sp_object_xy = subset(sp_object_xy, idents="cell_over")
  } else {
    # Rename the feature pairs
    if (use_lr_db) {
      feat_pairs <- return_celltalkdb(lr_db_species=lr_db_species, conda.env.name = conda.env.name)
      feat_pairs <- feat_pairs[,c('ligand_gene_symbol','receptor_gene_symbol')]
      use_lr_db <- F
      print(paste0("Calculating topological similarity between genes in ",celltype_x," and ",celltype_y))
      print("Using CellTalkDB ligand-receptor dataset")
    } else {
      colnames(feat_pairs) <- c('ligand_gene_symbol','receptor_gene_symbol')
    }
    # Modify the column name of the feature pair dataframe
    feat_pairs[['ligand_gene_symbol']] <- paste0(celltype_x,'_',feat_pairs[['ligand_gene_symbol']])
    feat_pairs[['receptor_gene_symbol']] <- paste0(celltype_y,'_',feat_pairs[['receptor_gene_symbol']])

    # Create combined Seurat object for two cell type specific count matrices
    sp_object_list <- celltype_specific_object(sp_object, c(celltype_x, celltype_y),
                                               conda.env.name=conda.env.name,
                                               fov_colname = fov_colname,
                                               cell_id_colname=cell_id_colname,
                                               celltype_colname=celltype_colname,
                                               transcript_colname=transcript_colname,
                                               sc_norm_total=sc_norm_total,
                                               return_mode='anndata')
    comb_var_names <- c(paste0(celltype_x,'_',reticulate::py_to_r(sp_object_list[[1]]$var_names$values)),
                        paste0(celltype_y,'_',reticulate::py_to_r(sp_object_list[[2]]$var_names$values)))
    adata_xy <- ann$AnnData(X=scipy$sparse$hstack(c(sp_object_list[[1]]$X, sp_object_list[[2]]$X))$tocsr(),
                            obs=sp_object_list[[1]]$obs)
    adata_xy$var_names <- comb_var_names
    adata_xy <- STopover$STopover_cosmx(adata_xy, sc_celltype_colname=celltype_colname,
                                        sc_norm_total=sc_norm_total,
                                        min_size=min_size, fwhm=fwhm, thres_per=thres_per)

    # Calculate topological similarites between the pairs from the two cell types
    if (is.null(num_workers)){
      adata_xy$topological_similarity(feat_pairs=feat_pairs, use_lr_db=use_lr_db, lr_db_species=lr_db_species,
                                      group_name='batch', group_list=NULL, J_result_name=J_result_name)
    } else {
      adata_xy$topological_similarity(feat_pairs=feat_pairs, use_lr_db=use_lr_db, lr_db_species=lr_db_species,
                                      group_name='batch', group_list=NULL, J_result_name=J_result_name, num_workers=num_workers)
    }
    sparse_mtx <- reticulate::py_to_r(adata_xy$X$T)
    colnames(sparse_mtx) <- reticulate::py_to_r(adata_xy$obs_names$values)
    rownames(sparse_mtx) <- reticulate::py_to_r(adata_xy$var_names$values)
    sp_object_xy <- Seurat::CreateSeuratObject(counts = sparse_mtx, assay = "Spatial")
    sp_object_xy[['nCount_Spatial']] <- NULL
    # Assign coordinates of the cell
    df_coord = data.frame(row=reticulate::py_to_r(adata_xy$obs['array_row']),
                          col=reticulate::py_to_r(adata_xy$obs['array_col']),
                          stringsAsFactors=F)
    rownames(df_coord) = reticulate::py_to_r(adata_xy$obs_names$values)
    sp_object_xy@images$image = methods::new(Class = 'SlideSeq', assay = "Spatial", key = "image_", coordinates = df_coord)


    # Saving CC location data into the metadata of spatial Seurat object
    cc_loc_df <- reticulate::py_to_r(adata_xy$obs)
    cc_loc_names <- grep(colnames(cc_loc_df), pattern = "Comb_CC_", value=T)
    cc_loc_df <- cc_loc_df[,cc_loc_names]
    sp_object_xy <- Seurat::AddMetaData(sp_object_xy, cc_loc_df)
    # Saving jaccard index result into the @misc of spatial Seurat object
    sp_object_xy@misc <- reticulate::py_to_r(adata_xy$uns[[paste(c('J',J_result_name,0),collapse="_")]])
  }

  return(sp_object_xy)
}
