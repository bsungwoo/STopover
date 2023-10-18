#' Visualizing imageST data on the grid
#' @description  Visualizing imageST data with ggplot2
#' @param sp_object spatial data (Seurat object) to be used
#' @param feature feature to visualize on the grid
#' @param plot_title title of the plot (default = feature)
#' @param title_fontsize fontsize of the plot title (default = 12)
#' @param color_dis if the feature is discrete value, provide vector of colors to distinguish the values (default = c("#A2E1CA","#FBBC05","#4285F4","#34A853"))
#' @param color_cont if the feature is a continuous variable, provide color palette to use from RColorBrewer::brewer.pal.info (default = "RdPu")
#' @param vmax if the feature is a continuous variable, provide the maximum of the colorbar (default = NULL)
#' @param vmin if the feature is a continuous variable, provide the minimum of the colorbar (default = NULL)
#' @param legend_loc location of the legend (default= 'right')
#' @param legend_fontsize fontsize of the legend title (default = 12)
#' @export
#' @return ggplot object
vis_spatial_imageST <- function(sp_object, feature, plot_title=feature, title_fontsize=12,
                              color_dis=c("#A2E1CA","#FBBC05","#4285F4","#34A853"),
                              color_cont="RdPu",vmin=NULL, vmax=NULL, legend_loc='right',
                              legend_fontsize=12) {
  if (!identical(setdiff(c("array_col","array_row",feature),
                         c(colnames(sp_object@meta.data),rownames(sp_object))), character(0))){
    stop("'array_col', 'array_row', or 'feature' not found among column names of metadata or gene names")
  }
  df <- Seurat::FetchData(sp_object, vars = c("array_col","array_row",feature))
  if (is.null(levels(df[[feature]]))){
    if (!color_cont %in% rownames(RColorBrewer::brewer.pal.info)){
      stop(paste0("'color_cont' should be among: ",
                  paste(rownames(RColorBrewer::brewer.pal.info), collapse = ", ")))
    }
    # Fix the minmax value
    if (!is.null(vmax)) {
      if (vmax < max(df[[feature]])) df[df[feature]>vmax,][feature] <- vmax
      if (is.null(vmin)) {
        vmin <- min(df[[feature]])
      } else{
        if (vmin > min(df[[feature]])) df[df[feature]<vmin,][feature] <- vmin
      }
    } else if (!is.null(vmin)) {
      if (vmin > min(df[[feature]])) df[df[feature]<vmin,][feature] <- vmin
      vmax <- max(df[[feature]])
    }
    # Draw plot
    eval(parse(text=paste0("
    p <- ggplot2::ggplot(df, ggplot2::aes(x = array_col, y = array_row, fill =`",
                           feature,"`)) + ggplot2::theme_minimal() +
      ggplot2::geom_tile() + ggplot2::ggtitle(plot_title) +
      ggplot2::coord_fixed() + ggplot2::scale_fill_distiller(palette = color_cont, ",
                           "type='seq', limits=c(vmin,vmax))")))
  } else {
    eval(parse(text=paste0("
    p <- ggplot2::ggplot(df, ggplot2::aes(x = array_col, y = array_row, fill =`",
                           feature,"`)) + ggplot2::theme_minimal() +
      ggplot2::geom_tile() + ggplot2::ggtitle(plot_title) +
      ggplot2::coord_fixed() + ggplot2::scale_fill_manual(values = color_dis)")))
  }
  p <- p +
    ggplot2::theme(panel.border = ggplot2::element_blank(),
                   panel.grid.major = ggplot2::element_blank(),
                   panel.grid.minor = ggplot2::element_blank(),
                   plot.title=ggplot2::element_text(size=title_fontsize,hjust=0.5),
                   axis.title.x=ggplot2::element_blank(),
                   axis.title.y=ggplot2::element_blank(),
                   axis.text.x=ggplot2::element_blank(),
                   axis.text.y=ggplot2::element_blank(),
                   axis.ticks.x = ggplot2::element_blank(),
                   axis.ticks.y = ggplot2::element_blank(),
                   legend.position=legend_loc,
                   legend.text=ggplot2::element_text(size=legend_fontsize,hjust=0.5))
  return(p)
}

#' Visualizing all connected components of feature x and y
#' @description Visualizing connected component locations of feature x and feature y on the tissue and return plots or save plots if designated
#' @param sp_object spatial data (Seurat object) to be used
#' @param feat_name_x name of the feature x (default = '')
#' @param feat_name_y name of the feature y (default = '')
#' @param celltype_x cell type corresponding to the first feature if the cell type specific data is provided (default = NULL)
#' @param celltype_y cell type corresponding to the second feature if the cell type specific data is provided (default = NULL)
#' @param conda.env.name name of the conda environment to use for STopover analysis (default = 'STopover')
#' @param dot_size size of the spot/grid visualized on the tissue (default =  1.8)
#' @param alpha_img transparency of the background image (not available for imageST data) (default = 0.8)
#' @param alpha transparency of the colored spot (default = 0.8)
#' @param vis_jaccard whether to visualize jaccard index on right corner of the plot (default = T)
#' @param subset_by_slide whether to select the certain slides for the visualization (if there are multiple slides in sp_object) (default = F)
#' @param slide_names name of the slides to select (among 'names(sp_object@images)') (default = names(sp_object@images))
#' @param slide_titles title of the slides to visualize in the plot (default = NULL)
#' @param slide_ncol number of images to visualize in the column (default = 2)
#' @param title_fontsize fontsize of the figure title (default = 15)
#' @param legend_loc location of the figure legend (default = 'right')
#' @param legend_fontsize fontsize of the figure legend (default = 10)
#' @param add_title_text the text to add in the figure title (default = '')
#' @param crop_image whether to crop the image (default = F)
#' @param return_plot whether to return the plot as list (default = F)
#' @param save whether to save the image (default = F)
#' @param save_path path to save the image (default = '.')
#' @param save_name_add the text to add in the file name (default = '')
#' @param dpi dpi to save the image (default = 100)
#' @param fig_width figure width in ggsave (default = 5)
#' @param fig_height figure height in ggsave (default = 5)
#' @return list of plots
#' @export
vis_all_connected <- function(sp_object, feat_name_x='', feat_name_y='',
                              celltype_x = NULL, celltype_y = NULL,
                              conda.env.name = 'STopover',
                              dot_size=1.8, alpha_img=0.8, alpha=0.8, vis_jaccard=T,
                              subset_by_slide=F, slide_names=names(sp_object@images), slide_titles=NULL,
                              slide_ncol=2, # For multiple slides
                              title_fontsize=15, legend_loc='right', legend_fontsize=10,
                              add_title_text='', crop_image=F, return_plot=F,
                              save=F, save_path='.', save_name_add='', dpi=100,
                              fig_width=4, fig_height=4){
  # Check the data type
  spatial_type <- ifelse(grepl(tolower(class(sp_object@images[[1]])[1]),pattern="visium"),"visium","imageST")
  cat(paste0("The provided object is considered a ",spatial_type," dataset\n"))
  # Convert the feature name if cell type specific data is provided
  if (!is.null(celltype_x) & !is.null(celltype_y) & spatial_type=='imageST'){
    feat_name_x <- paste0(celltype_x,"_",feat_name_x)
    feat_name_y <- paste0(celltype_y,"_",feat_name_y)
  }
  if (!is.null(celltype_x) & !is.null(celltype_y) & spatial_type=='visium'){
    feat_name_x <- paste0(celltype_x,"_",celltype_y,"_",feat_name_x)
    feat_name_y <- paste0(celltype_x,"_",celltype_y,"_",feat_name_y)
  }
  # Aggregate all connected components and save overlapping regions separately
  sp_object[['Over']] = factor(((1 * ((sp_object[[paste0('Comb_CC_',feat_name_x)]] != 0) &
                                        (sp_object[[paste0('Comb_CC_',feat_name_y)]] == 0))) +
                                  (2 * ((sp_object[[paste0('Comb_CC_',feat_name_x)]] == 0) &
                                          (sp_object[[paste0('Comb_CC_',feat_name_y)]] != 0))) +
                                  (3 * ((sp_object[[paste0('Comb_CC_',feat_name_x)]] != 0) &
                                          (sp_object[[paste0('Comb_CC_',feat_name_y)]] != 0)))))
  # Assign colors and labels to the spots/grids
  feature_map <- c("0"="Others","1"=feat_name_x,"2"=feat_name_y,"3"="Over")
  levels(sp_object@meta.data[['Over']]) <- feature_map[as.character(levels(sp_object@meta.data[['Over']]))]
  color.map <- c("#A2E1CA","#FBBC05","#4285F4","#34A853")
  names(color.map) <- c("Others", feat_name_x, feat_name_y, "Over")
  if (spatial_type=='visium') {
    color.map <- color.map[as.character(setdiff(levels(sp_object@meta.data[['Over']]),"Others"))]
  } else if (spatial_type=='imageST') {
    color.map <- color.map[as.character(levels(sp_object@meta.data[['Over']]))]
  }

  # Define the 'batch' column according to the slide names: names(sp_object@images)
  if (!'batch' %in% colnames(sp_object@meta.data)){
    df_image_all <- data.frame()
    for (slide in names(sp_object@images)){
      df_image <- data.frame(row.names = rownames(sp_object@images[[slide]]@coordinates))
      df_image[['batch']] <- slide
      df_image_all <- rbind(df_image_all, df_image)
    }
    sp_object <- Seurat::AddMetaData(sp_object, df_image_all)
  }

  # Subset the slide by the image name: names(sp_object@images)
  slide_list <- 1:length(names(sp_object@images))
  names(slide_list) <- names(sp_object@images)
  if (subset_by_slide){
    if (!identical(setdiff(slide_names, names(slide_list)), character(0))){
      stop("'slide_names' should be among names(sp_object@images)")
    }
    Seurat::Idents(sp_object) <- "batch"
    sp_object <- subset(sp_object, idents = slide_names)
    # Rearrange the names according to the sequence saved in names(sp_object@images)
    slide_names <- names(slide_list)[slide_list[slide_names]]
    sp_object@images <- sp_object@images[slide_names]
  }

  # Subset the object to highlight the connected component locations
  sp_object_mod <- sp_object
  Seurat::Idents(sp_object_mod) <- "Over"
  if (spatial_type=='visium'){
    sp_object_mod <- subset(sp_object_mod, idents = names(color.map))
  }
  # Draw spatial cluster plot for connected component locations
  if (spatial_type=='imageST'){
    p <- list(vis_spatial_imageST(sp_object_mod, "Over", color_dis=color.map))
  } else{
    p <- Seurat::SpatialPlot(sp_object_mod, alpha=alpha, image.alpha = alpha_img,
                             label=F, crop=crop_image, cols = color.map,
                             pt.size.factor=dot_size,
                             combine=FALSE)
  }

  # Install and load conda environment
  install_load_env(conda.env.name)
  ## Import python modules
  STopover <- reticulate::import('STopover', convert = FALSE)

  # Check the slide title and check the format of the data
  if (is.null(slide_titles)) {
    if (spatial_type=='visium') {slide_titles <- paste0(slide_names, '\n')}
    else {slide_titles <- ''}
  } else {
    if (length(slide_titles)!=length(slide_names)){
      stop("'slide_titles' should be a vector or list with a same length as slide_names")
    }
    else {
      for (idx in 1:length(slide_titles)){if (slide_titles[idx]!=''){slide_titles[idx] <- paste0(slide_titles[idx],'\n')}}
    }
  }
  # Generate plots
  for (i in 1:length(p)){
    if (spatial_type=='visium'){
      Seurat::Idents(sp_object) <- "batch"
      sp_object_mod <- subset(sp_object, idents = slide_names[i])
    } else {
      sp_object_mod <- sp_object
    }
    comb_cc_loc <- c(paste0("Comb_CC_",feat_name_x), paste0("Comb_CC_",feat_name_y))

    # Convert CC location factor information into numeric values
    for (cc_element in comb_cc_loc){
      sp_object_mod@meta.data[[cc_element]] <- as.numeric(as.character(sp_object_mod@meta.data[[cc_element]]))
    }
    if (vis_jaccard){
      adata_sp <- convert_to_anndata(sp_object_mod, features = comb_cc_loc)
      J_comp <- reticulate::py_to_r(STopover$jaccard_and_connected_loc_(adata_sp, feat_name_x=feat_name_x, feat_name_y=feat_name_y, J_index=T))
    }

    p[[i]] <- p[[i]] + ggplot2::ggtitle(paste0(slide_titles[[i]],
                                               feat_name_x,' & ',feat_name_y, add_title_text)) +
      ggplot2::theme(plot.title=ggplot2::element_text(size=title_fontsize,hjust=0.5),
                     legend.title=ggplot2::element_blank(),
                     legend.text=ggplot2::element_text(size=legend_fontsize,hjust=0.5))
    if (vis_jaccard) {p[[i]] <- p[[i]] + ggplot2::annotate("text", x=Inf, y=Inf,
                                                           label=paste0("J_comp: ",
                                                                        format(round(J_comp,3), nsmall = 3, digits=3)),
                                                           size=legend_fontsize/3, hjust=1, vjust=1)}
  }
  # Draw plot
  if (spatial_type=='visium'){
    if (length(p)<slide_ncol) {slide_ncol <- length(p)}
    slide_nrow <- ((length(p)-1)%/%slide_ncol)+1
    p_mod <- patchwork::wrap_plots(p, ncol=slide_ncol) +
      patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = legend_loc)
  } else {
    p_mod <- patchwork::wrap_plots(p, ncol=1)
    slide_nrow <- 1; slide_ncol <- 1
  }
  plot(p_mod)

  if (save){ggplot2::ggsave(file.path(save_path,
                                      paste0(paste(c(spatial_type,'loc_CCxy',feat_name_x,feat_name_y), collapse="_"),
                                             save_name_add, '.png')),
                            height=fig_height*slide_nrow, width=fig_width*slide_ncol, dpi=dpi, bg = "white",
                            units = "cm", limitsize=F)}
  if (return_plot){return(p_mod)}
}


#' Visualizing top connected components with the highest local Jaccard index
#' @description Visualizing top connected component locations of feature x and feature y having the highest local Jaccard index and return plots or save plots if designated
#' @param sp_object spatial data (Seurat object) to be used
#' @param feat_name_x name of the feature x (default = '')
#' @param feat_name_y name of the feature y (default = '')
#' @param celltype_x cell type corresponding to the first feature if the cell type specific data is provided (default = NULL)
#' @param celltype_y cell type corresponding to the second feature if the cell type specific data is provided (default = NULL)
#' @param top_n the number of the top connected component pairs with the highest Jaccard similarity index
#' @param conda.env.name name of the conda environment to use for STopover analysis (default = 'STopover')
#' @param dot_size size of the spot/grid visualized on the tissue (default =  1.8)
#' @param alpha_img transparency of the background image (not available for imageST data) (default = 0.8)
#' @param alpha transparency of the colored spot (default = 0.8)
#' @param vis_jaccard whether to visualize jaccard index on right corner of the plot (default = T)
#' @param slide_name name of one slide to select (among 'names(sp_object@images)') (default = names(sp_object@images))
#' @param slide_title title of the selected slide to visualize in the plot (default = NULL)
#' @param slide_ncol number of images to visualize in the column (default = 2)
#' @param title_fontsize fontsize of the figure title (default = 15)
#' @param legend_loc location of the figure legend (default = 'right')
#' @param legend_fontsize fontsize of the figure legend (default = 10)
#' @param add_title_text the text to add in the figure title (default = '')
#' @param crop_image whether to crop the image (default = F)
#' @param return_plot whether to return the plot as list (default = F)
#' @param return_sp_object whether to return the sp_object containing top n connected component location saved in metadata (default = F)
#' @param save whether to save the image (default = F)
#' @param save_path path to save the image (default = '.')
#' @param save_name_add the text to add in the file name (default = '')
#' @param dpi dpi to save the image (default = 100)
#' @param fig_width figure width in ggsave (default = 5)
#' @param fig_height figure height in ggsave (default = 5)
#' @return list of plots or Seurat object containing top n connected component location
#' @export
vis_jaccard_top_n_pair <- function(sp_object, feat_name_x='', feat_name_y='',
                                   celltype_x=NULL, celltype_y=NULL,
                                   top_n=2,
                                   conda.env.name = 'STopover',
                                   dot_size=1.8, alpha_img=0.8, alpha=0.8, vis_jaccard=T,
                                   slide_name=names(sp_object@images)[1], slide_title=NULL,
                                   slide_ncol=2, # For multiple slides
                                   title_fontsize=15, legend_loc = 'right',
                                   legend_fontsize=10, add_title_text='',
                                   crop_image=F, return_plot=F, return_sp_object=F,
                                   save=F, save_path = '.', save_name_add='', dpi=100,
                                   fig_width=5, fig_height=5){
  if (length(slide_name) > 1){stop("'slide_name' should be one element of names(sp_object@images)")}
  if (!slide_name %in% names(sp_object@images)){stop("'slide_name' should be among names(sp_object@images)")}
  # Check the data type
  spatial_type <- ifelse(grepl(tolower(class(sp_object@images[[1]])[1]),pattern="visium"),"visium","imageST")
  cat(paste0("The provided object is considered a ",spatial_type," dataset\n"))
  # Convert the feature name if cell type specific data is provided
  if (!is.null(celltype_x) & !is.null(celltype_y) & spatial_type=='imageST'){
    feat_name_x <- paste0(celltype_x,"_",feat_name_x)
    feat_name_y <- paste0(celltype_y,"_",feat_name_y)
  }
  if (!is.null(celltype_x) & !is.null(celltype_y) & spatial_type=='visium'){
    feat_name_x <- paste0(celltype_x,"_",celltype_y,"_",feat_name_x)
    feat_name_y <- paste0(celltype_x,"_",celltype_y,"_",feat_name_y)
  }

  # Define the 'batch' column according to the slide names: names(sp_object@images)
  if (!'batch' %in% colnames(sp_object@meta.data)){
    df_image_all <- data.frame()
    for (slide in names(sp_object@images)){
      df_image <- data.frame(row.names = rownames(sp_object@images[[slide]]@coordinates))
      df_image[['batch']] <- slide
      df_image_all <- rbind(df_image_all, df_image)
    }
    sp_object <- Seurat::AddMetaData(sp_object, df_image_all)
  }

  # Subset the slide by the image name: names(sp_object@images)
  if (spatial_type == 'visium'){
    Seurat::Idents(sp_object) <- "batch"
    sp_object <- subset(sp_object, idents = slide_name)
    sp_object@images <- sp_object@images[slide_name]
  }

  # Install and load conda environment
  install_load_env(conda.env.name)
  ## Import python modules
  STopover <- reticulate::import('STopover', convert = FALSE)
  comb_cc_loc <- c(paste0("Comb_CC_",feat_name_x), paste0("Comb_CC_",feat_name_y))
  # Convert CC location factor information into numeric values
  for (cc_element in comb_cc_loc){
    sp_object@meta.data[[cc_element]] <- as.numeric(as.character(sp_object@meta.data[[cc_element]]))
  }
  adata_sp <- convert_to_anndata(sp_object, features = comb_cc_loc)
  J_top_result <- STopover$jaccard_top_n_connected_loc_(adata_sp, feat_name_x=feat_name_x, feat_name_y=feat_name_y, top_n = as.integer(top_n))
  adata_sp_mod <- J_top_result[[0]]; J_top <- reticulate::py_to_r(J_top_result[[1]])

  # Generate plots
  p <- list()
  if (spatial_type=='imageST'){alpha_img <- 0}
  for (i in 1:top_n){
    cc_loc_xy <- reticulate::py_to_r(adata_sp_mod$obs[[paste(c("CCxy_top",i,feat_name_x,feat_name_y),collapse = "_")]]$astype('int'))
    sp_object[[paste0('CCxy_top_',i)]] <- factor(cc_loc_xy)

    # Assign colors and labels to the spots/grids
    color.map <- c("0"="#A2E1CA","1"="#FBBC05","2"="#4285F4","3"="#34A853")
    if (spatial_type=='visium') {
      color.map <- color.map[as.character(setdiff(levels(sp_object@meta.data[[paste0('CCxy_top_',i)]]),0))]
    } else if (spatial_type=='imageST') {
      color.map <- color.map[as.character(levels(sp_object@meta.data[[paste0('CCxy_top_',i)]]))]
    }
    feature_map <- c("0"="Others","1"=feat_name_x,"2"=feat_name_y,"3"="Over")
    levels(sp_object@meta.data[[paste0('CCxy_top_',i)]]) <- feature_map[as.character(levels(sp_object@meta.data[[paste0('CCxy_top_',i)]]))]

    # Subset the object to highlight the top i connected component locations
    sp_object_mod <- sp_object
    Seurat::Idents(sp_object_mod) <- paste0('CCxy_top_',i)
    if (spatial_type=='visium'){sp_object_mod <- subset(sp_object_mod, idents = feature_map[names(color.map)])}

    # Draw spatial cluster plot for connected component locations
    if (spatial_type == 'visium'){
      p[[i]] <- Seurat::SpatialPlot(sp_object_mod, alpha=alpha, image.alpha = alpha_img,
                                    label=F, crop=crop_image,
                                    pt.size.factor=dot_size,
                                    combine=FALSE)[[1]]
    } else {
      p[[i]] <- vis_spatial_imageST(sp_object_mod, paste0('CCxy_top_',i), color_dis=color.map)
    }
    p[[i]] <- p[[i]] + ggplot2::ggtitle(paste0(ifelse(is.null(slide_title), paste0(slide_name,'\n'),
                                                      ifelse(slide_title=='','',paste0(slide_title,'\n'))),
                                               feat_name_x,' & ',feat_name_y, add_title_text,": ",
                                               "top ",i," CCxy")) +
      ggplot2::scale_fill_manual(values = as.character(color.map)) +
      ggplot2::theme(plot.title=ggplot2::element_text(size=title_fontsize,hjust=0.5),
                     legend.title=ggplot2::element_blank(), legend.position='right',
                     legend.text=ggplot2::element_text(size=legend_fontsize,hjust=0.5))
    if (vis_jaccard) {p[[i]] <- p[[i]] + ggplot2::annotate("text", x=Inf, y=Inf,
                                                           label=paste0("J_local: ",
                                                                        format(round(J_top[i],3), nsmall = 3, digits=3)),
                                                           size=legend_fontsize/3, hjust=1, vjust=1)}
  }

  # Draw plot
  if (length(p)<slide_ncol) {slide_ncol <- length(p)}
  slide_nrow <- ((length(p)-1)%/%slide_ncol)+1
  p_mod <- patchwork::wrap_plots(p, ncol=slide_ncol) +
    patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = legend_loc)
  plot(p_mod)

  if (save){ggplot2::ggsave(file.path(save_path,
                                      paste0(paste(c(spatial_type,slide_name,'J_top',top_n,feat_name_x,feat_name_y),collapse="_"),
                                             save_name_add,'.png')),
                            height=fig_height*slide_nrow, width=fig_width*slide_ncol, dpi=dpi, bg = "white",
                            units = "cm", limitsize=F)}
  if (return_plot){
    if (return_sp_object) {
      try(Seurat::Idents(sp_object) <- "seurat_clusters")
      return(list(p_mod, sp_object))
    }
    else return (p_mod)
  }
}


#' Visualizing upregulated LR interactions and their corresponding GO terms using a heatmap
#' @description Spatial LR interaction profiles are estimated by STopover and differentially upregulated LR interactions in 'comp_group' compared to 'ref_group' are selected (threshold: Jcomp>J_comp_cutoff & fold change of Jcomp>J_comp_cutoff). The enrichment analysis is performed for the genes in LR pairs and the top 10 enriched GO terms with the lowest adjusted p-values were selected (threshold: p.adjust<padjust_cutoff). LR pairs in which either ligand or receptor gene intersected with the genes composing the top GO terms were listed. The upregulated functional terms, the intersecting LR pairs, and fold change of Jcomp are visualized in a heatmap.
#' @param sp_object spatial data (Seurat object) to be used
#' @param ref_group vector containing slide name(s) corresponding to the group that are considered as a baseline (should be among names(sp_object@images))
#' @param comp_group vector containing slide name(s) corresponding to the group in which upregulated LR paris are tested (should be among names(sp_object@images))
#' @param logFC_cutoff log fold change cutoff of J_comp for the differentially upregulated LR pairs in 'comp_group' compared to 'ref_group' (default = 1)
#' @param J_comp_cutoff J_comp cutoff in the comp_group (default = 0.2)
#' @param go_species species of the given dataset (default = "human")
#' @param db_name: name of the ligand-receptor database to use: either 'CellTalk', 'CellChat', or 'Omnipath' (default = 'CellTalk')
#' @param ontology_cat category of GO terms to show in the heatmap (default ="BP")
#' @param padjust_method method for adjusting p-values during overrepresentation tests to identify significant GO terms (default ="BH")
#' @param padjust_cutoff adjusted p-value cutoff to select the GO terms (default = 0.05)
#' @param top_n the top GO terms to show in the heatmap (default=10)
#' @param heatmap_max the maximum value to show in the colorbar (default=10)
#' @param title_fontsize the title fontsize in the heatmap (default = 12)
#' @param xaxis_title_fontsize the x axis label size in the heatmap (default = 12)
#' @param yaxis_title_fontsize the y axis label size in the heatmap (default = 12)
#' @param angle_col the angle of the column label in the heatmap (refer to the pheatmap 'angle_col') (default = 45)
#' @param colorbar_palette the vector of colors to be used in the colormap (select among the grDevices::hcl.pals())
#' @param legend_loc location of the figure legend (default = 'right')
#' @param save_plot whether to save the heatmap (default = F)
#' @param save_path the path to save the heatmap (default = '.')
#' @param save_name the name of the heatmap to save (default = 'heatmap_GO_LR_int.png')
#' @param fig_width the width of the heatmap to save (default = 21)
#' @param fig_height the height of the heatmap to save (default = 10)
#' @param dpi the dpi of the heatmap to save (default = 150)
#' @param return_results return a dataframe showing the list of top upregulated LR interaction and a heatmap
#' @return a list of dataframe showing the list of top differentially increased LR interaction in comp_group compared to ref_group and a heatmap
#' @export
vis_diff_inc_lr_pairs <- function(sp_object, ref_group, comp_group,
                                  logFC_cutoff=1, J_comp_cutoff=0.2,
                                  go_species=c("human","mouse"),
                                  db_name='CellTalk',
                                  ontology_cat=c("BP","CC","MF","all"),
                                  padjust_method = "BH", padjust_cutoff=0.05,
                                  top_n = 10, heatmap_max=10,
                                  title_fontsize=12,
                                  xaxis_title_fontsize=10,
                                  yaxis_title_fontsize=10,
                                  angle_col=45,
                                  colorbar_palette=rownames(RColorBrewer::brewer.pal.info),
                                  legend_loc='right',legend_fontsize=10,
                                  save_plot=F, save_path='.',
                                  save_name='heatmap_GO_LR_int',
                                  fig_width=21, fig_height=10, dpi=150,
                                  return_results=T){
  # Check the feasibility of the given inputs
  if (!identical(setdiff(ref_group, names(sp_object@images)),character(0))){stop("'ref_group' should be among names(sp_object@images)")}
  if (!identical(setdiff(comp_group, names(sp_object@images)),character(0))){stop("'comp_group' should be among names(sp_object@images)")}
  go_species <- match.arg(go_species)
  ontology_cat <- match.arg(ontology_cat)
  colorbar_palette <-  match.arg(colorbar_palette)
  ## Define group match vector
  ref_group_match <- rep("ref", length(ref_group))
  comp_group_match <- rep("comp", length(comp_group))
  names(ref_group_match) <- ref_group
  names(comp_group_match) <- comp_group
  group_match <- c(ref_group_match, comp_group_match)
  ## Extract LR database for the given species
  ref_df_ <- STopover::return_lr_db(lr_db_species = go_species, db_name=db_name)[c('lr_pair','ligand_gene_symbol','receptor_gene_symbol')]
  ref_df <- do.call("rbind", replicate(length(names(sp_object@images)), ref_df_, simplify = FALSE))
  ref_df <- cbind(data.frame(Slide =  as.character(rep(names(sp_object@images), each = dim(ref_df_)[1]))),
                  ref_df)
  colnames(ref_df) <- c("batch","lr_pair","Feat_1","Feat_2")

  ## Comparison between ref_group and comp_group
  df_top_diff_ref <- sp_object@misc %>% data.frame(.) %>%
    dplyr::mutate(lr_pair = paste(Feat_1,Feat_2,sep= "_")) %>%
    dplyr::left_join(ref_df, ., by=c("lr_pair","Feat_1","Feat_2","batch")) %>%
    dplyr::mutate(J_comp = ifelse(is.na(J_comp), 0, J_comp)) %>%
    dplyr::mutate(Group = group_match[batch]) %>%
    dplyr::group_by(Group, lr_pair, Feat_1, Feat_2) %>%
    dplyr::summarise(Mean = mean(J_comp), .groups="keep")

  df_top_diff_comp <- df_top_diff_ref %>%
    dplyr::group_by(Group) %>%
    dplyr::mutate(logFC = log2(Mean / dplyr::filter(., Group == "ref") %>% dplyr::pull(Mean))) %>%
    dplyr::filter(Group=="comp") %>%
    dplyr::filter(logFC>logFC_cutoff, Mean>J_comp_cutoff) %>%
    dplyr::arrange(dplyr::desc(logFC))

  require(clusterProfiler)
  feat_interest <- unique(c(df_top_diff_comp %>% dplyr::pull("Feat_1"),
                            df_top_diff_comp %>% dplyr::pull("Feat_2")))

  if (!identical(feat_interest, character(0))){
    if (go_species=="human"){
      require(org.Hs.eg.db)
      sym2ent <- AnnotationDbi::mapIds(org.Hs.eg.db, feat_interest, "ENTREZID", "SYMBOL")
      GO_result <- clusterProfiler::enrichGO(gene          = sym2ent,
                                             OrgDb         = org.Hs.eg.db,
                                             ont           = ontology_cat,
                                             pAdjustMethod = padjust_method,
                                             pvalueCutoff  = padjust_cutoff,
                                             qvalueCutoff  = 0.2, readable = TRUE)
    } else if (go_species=="mouse"){
      require(org.Mm.eg.db)
      sym2ent <- AnnotationDbi::mapIds(org.Mm.eg.db, feat_interest, "ENTREZID", "SYMBOL")
      GO_result <- clusterProfiler::enrichGO(gene          = sym2ent,
                                             OrgDb         = org.Mm.eg.db,
                                             ont           = ontology_cat,
                                             pAdjustMethod = padjust_method,
                                             pvalueCutoff  = padjust_cutoff,
                                             qvalueCutoff  = 0.2, readable = TRUE)
    }
    df <- GO_result@result %>% dplyr::filter(p.adjust < padjust_cutoff)
    df[['GeneRatio']] <- sapply(df[['GeneRatio']], function(x) eval(parse(text=x)))
    df <- df %>% dplyr::slice(1:top_n) %>% dplyr::arrange(dplyr::desc(GeneRatio))
    lr_pair_match <- data.frame()
    for (idx in 1:length(df$geneID)){
      if (grepl(df$geneID[idx],pattern="/")) {gene_list <- strsplit(df$geneID[idx], split="/")[[1]]}
      else {gene_list <- df$geneID[idx]}
      df_lr_tmp <- data.frame()
      for (gene in gene_list){
        lr_list_tmp <- df_top_diff_comp %>% dplyr::filter(Feat_1==gene|Feat_2==gene) %>% dplyr::pull(lr_pair)
        df_lr_tmp <- data.frame(GO_terms = df[idx, "Description"], lr_pair = lr_list_tmp)
        lr_pair_match <- rbind(lr_pair_match, df_lr_tmp)
      }
    }
    lr_pair_match_ <- lr_pair_match %>% dplyr::distinct(.) %>%
      dplyr::left_join(., df_top_diff_comp, by="lr_pair") %>%
      dplyr::select(GO_terms, lr_pair, logFC) %>%
      dplyr::arrange(dplyr::desc(logFC)) %>%
      dplyr::mutate(logFC = ifelse(logFC>heatmap_max, heatmap_max, logFC))
    # Make pivot table with the dataset
    df_summ <- lr_pair_match_ %>% tidyr::pivot_wider(names_from = lr_pair, values_from = logFC)
    lr_pair_match_[['lr_pair']] <- factor(lr_pair_match_[['lr_pair']], levels = colnames(df_summ)[-1])
    lr_pair_match_[['GO_terms']] <- factor(lr_pair_match_[['GO_terms']], levels = rev(df_summ$GO_terms))
    # Draw the plot
    out <- ggplot2::ggplot(lr_pair_match_, ggplot2::aes(x = lr_pair, y = GO_terms, fill = logFC)) +
      ggplot2::geom_tile(width = 0.95, height = 0.95) +
      ggplot2::coord_fixed() +
      ggplot2::scale_fill_distiller(palette = colorbar_palette, type="seq", limits=c(0,heatmap_max)) +
      ggplot2::scale_y_discrete(labels = function(x) stringr::str_wrap(x,  width = 40)) +
      ggplot2::theme(panel.border = ggplot2::element_blank(),
                     panel.grid.major = ggplot2::element_blank(),
                     panel.grid.minor = ggplot2::element_blank(),
                     panel.background = ggplot2::element_rect(fill = "grey",
                                                              colour = "grey"),
                     plot.title=ggplot2::element_text(size=title_fontsize,hjust=0.5),
                     axis.title.x=ggplot2::element_blank(),
                     axis.title.y=ggplot2::element_blank(),
                     axis.text.x=ggplot2::element_text(size=xaxis_title_fontsize,angle=angle_col,hjust=1),
                     axis.text.y=ggplot2::element_text(size=yaxis_title_fontsize),
                     legend.position=legend_loc,
                     legend.text=ggplot2::element_text(size=legend_fontsize,hjust=0.5))
    if (save_plot) {
      plot(out)
      ggplot2::ggsave(file.path(save_path, paste0(paste(c(save_name,ref_group,comp_group),collapse="_"),'.png')),
                      height=fig_height, width=fig_width, dpi=dpi, bg = "white",
                      units = "cm", limitsize=F)
    }
    if (return_results){
      return(list(df_top_diff_comp, out))
    }
  } else {
    cat("None of the features were remained after filtering\n")
  }
}
