#' Visualizing all connected components of feature x and y
#' @description Visualizing connected component locations of feature x and feature y on the tissue and return plots or save plots if designated
#' @param sp_object spatial data (Seurat object) to be used
#' @param feat_name_x name of the feature x (default = '')
#' @param feat_name_y name of the feature y (default = '')
#' @param celltype_x cell type corresponding to the first feature if the cell type specific data is provided (default = NULL)
#' @param celltype_y cell type corresponding to the second feature if the cell type specific data is provided (default = NULL)
#' @param conda.env.name name of the conda environment to use for STopover analysis (default = 'STopover')
#' @param dot_size size of the spot/grid visualized on the tissue (default =  1.8)
#' @param alpha_img transparency of the background image (not available for cosmx data) (default = 0.8)
#' @param alpha transparency of the colored spot (default = 0.8)
#' @param vis_jaccard whether to visualize jaccard index on right corner of the plot (default = T)
#' @param subset_by_slide whether to select the certain slides for the visualization (if there are multiple slides in sp_object) (default = F)
#' @param slide_names name of the slides to select (among 'names(sp_object@images)') (default = names(sp_object@images))
#' @param slide_titles title of the slides to visualize in the plot (default = NULL)
#' @param slide_ncol number of images to visualize in the column (default = 2)
#' @param title_fontsize fontsize of the figure title (default = 15)
#' @param legend_fontsize fontsize of the figure legend (default = 10)
#' @param add_title_text the text to add in the figure title (default = ')
#' @param crop_image whether to crop the image (default = T)
#' @param return_plot whether to return the plot as list (default = F)
#' @param save whether to save the image (default = F)
#' @param save_path path to save the image (default = '.')
#' @param save_name_add the text to add in the file name (default = ')
#' @param dpi dpi to save the image (default = 100)
#' @param fig_width figure width in ggsave (default = 5)
#' @param fig_height figure height in ggsave (default = 5)
#' @return list of plots
#' @export
vis_all_connected <- function(sp_object, feat_name_x='', feat_name_y='',
                              celltype_x = NULL, celltype_y = NULL,
                              conda.env.name = 'STopover',
                              dot_size=1.8, alpha_img=0.8, alpha=0.8, vis_jaccard=T,
                              subset_by_slide=F, slide_names=names(sp_object@images),slide_titles=NULL,
                              slide_ncol=2, # For multiple slides
                              title_fontsize=15, legend_fontsize=10,
                              add_title_text='', crop_image=T, return_plot=F,
                              save=F, save_path='.', save_name_add='', dpi=100,
                              fig_width=4, fig_height=4){
  # Check the data type
  spatial_type <- ifelse(class(sp_object@images$image)[1]=="SlideSeq","cosmx","visium")
  # Convert the feature name if cell type specific data is provided
  if (!is.null(celltype_x) & spatial_type=='cosmx'){feat_name_x <- paste0(celltype_x,"_",feat_name_x)}
  if (!is.null(celltype_y) & spatial_type=='cosmx'){feat_name_y <- paste0(celltype_y,"_",feat_name_y)}
  # Aggregate all connected components and save overlapping regions separately
  sp_object[['Over']] = factor(((1 * ((sp_object[[paste0('Comb_CC_',feat_name_x)]] != 0) &
                                        (sp_object[[paste0('Comb_CC_',feat_name_y)]] == 0))) +
                                  (2 * ((sp_object[[paste0('Comb_CC_',feat_name_x)]] == 0) &
                                          (sp_object[[paste0('Comb_CC_',feat_name_y)]] != 0))) +
                                  (3 * ((sp_object[[paste0('Comb_CC_',feat_name_x)]] != 0) &
                                          (sp_object[[paste0('Comb_CC_',feat_name_y)]] != 0)))))

  # Define the 'batch' column according to the slide names: names(sp_object@images)
  df_image_all <- data.frame()
  for (slide in names(sp_object@images)){
    df_image <- data.frame(row.names = rownames(sp_object@images[[slide]]@coordinates))
    df_image[['batch']] <- slide
    df_image_all <- rbind(df_image_all, df_image)
  }
  sp_object <- Seurat::AddMetaData(sp_object, df_image_all)

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
    sp_object_mod <- subset(sp_object_mod, idents = c(feat_name_x, feat_name_y, "Over"))
  }
  # Draw spatial cluster plot for connected component locations
  if (spatial_type=='cosmx'){alpha_img <- 0}
  p <- Seurat::SpatialPlot(sp_object_mod, alpha=alpha, image.alpha = alpha_img,
                           label=F, crop=crop_image,
                           pt.size.factor=dot_size,
                           combine=FALSE)

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
    Seurat::Idents(sp_object) <- "batch"
    sp_object_mod <- subset(sp_object, idents = slide_names[i])
    comb_cc_loc <- c(paste0("Comb_CC_",feat_name_x), paste0("Comb_CC_",feat_name_y))

    # Assign colors and labels to the spots/grids
    sp_object_mod@meta.data[['Over']] <- factor(sp_object_mod@meta.data[['Over']])
    color.map <- c("0"="#A2E1CA","1"="#FBBC05","2"="#4285F4","3"="#34A853")
    if (spatial_type=='visium') {
      color.map <- color.map[as.character(setdiff(levels(sp_object_mod@meta.data[['Over']]),0))]
    } else if (spatial_type=='cosmx') {
      color.map <- color.map[as.character(levels(sp_object_mod@meta.data[['Over']]))]
    }
    feature_map <- c("0"="Others","1"=feat_name_x,"2"=feat_name_y,"3"="Over")
    levels(sp_object_mod@meta.data[['Over']]) <- feature_map[as.character(levels(sp_object_mod@meta.data[['Over']]))]

    # Convert CC location factor information into numeric values
    for (cc_element in comb_cc_loc){
      sp_object_mod@meta.data[[cc_element]] <- as.numeric(as.character(sp_object_mod@meta.data[[cc_element]]))
    }
    adata_sp <- convert_to_anndata(sp_object_mod, features = comb_cc_loc)
    J_comp <- reticulate::py_to_r(STopover$jaccard_and_connected_loc_(adata_sp, feat_name_x=feat_name_x, feat_name_y=feat_name_y, J_index=T))

    p[[i]] <- p[[i]] + ggplot2::ggtitle(paste0(slide_titles[[i]],
                                               feat_name_x,' & ',feat_name_y, add_title_text)) +
      ggplot2::scale_fill_manual(values = as.character(color.map)) +
      ggplot2::theme(plot.title=ggplot2::element_text(size=title_fontsize,hjust=0.5),
                     legend.title=ggplot2::element_blank(), legend.position='right',
                     legend.text=ggplot2::element_text(size=legend_fontsize,hjust=0.5))
    if (vis_jaccard) {p[[i]] <- p[[i]] + ggplot2::annotate("text", x=Inf, y=Inf,
                                                           label=paste0("J_comp: ",
                                                                        format(J_comp, nsmall = 3, digits=3)),
                                                           size=legend_fontsize/3, hjust=1, vjust=1)}
  }
  # Draw plot
  if (spatial_type=='visium'){
    if (length(p)<slide_ncol) {slide_ncol <- length(p)}
    slide_nrow <- ((length(p)-1)%/%slide_ncol)+1
    p_mod <- patchwork::wrap_plots(p, ncol=slide_ncol) + patchwork::plot_layout(guides = "collect")
  } else {
    p_mod <- patchwork::wrap_plots(p, ncol=1)
    slide_nrow <- 1; slide_ncol <- 1
  }
  plot(p_mod)

  if (save){ggplot2::ggsave(file.path(save_path,
                                      paste0(paste(c(spatial_type,'loc_CCxy',feat_name_x,feat_name_y),collapse="_"),
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
#' @param alpha_img transparency of the background image (not available for cosmx data) (default = 0.8)
#' @param alpha transparency of the colored spot (default = 0.8)
#' @param vis_jaccard whether to visualize jaccard index on right corner of the plot (default = T)
#' @param slide_name name of one slide to select (among 'names(sp_object@images)') (default = names(sp_object@images))
#' @param slide_title title of the selected slide to visualize in the plot (default = NULL)
#' @param slide_ncol number of images to visualize in the column (default = 2)
#' @param title_fontsize fontsize of the figure title (default = 15)
#' @param legend_fontsize fontsize of the figure legend (default = 10)
#' @param add_title_text the text to add in the figure title (default = ')
#' @param crop_image whether to crop the image (default = T)
#' @param return_plot whether to return the plot as list (default = F)
#' @param return_sp_object whether to return the sp_object containing top n connected component location saved in metadata (default = F)
#' @param save whether to save the image (default = F)
#' @param save_path path to save the image (default = '.')
#' @param save_name_add the text to add in the file name (default = ')
#' @param dpi dpi to save the image (default = 100)
#' @param fig_width figure width in ggsave (default = 5)
#' @param fig_height figure height in ggsave (default = 5)
#' @return list of plots or Seurat object containing top n connected component location
#' @export
vis_jaccard_top_n_pair <- function(sp_object, feat_name_x='', feat_name_y='',
                                   celltype_x=NULL, celltype_y=NULL,
                                   top_n=2,
                                   conda.env.name = 'STopover',
                                   dot_size=1.3, alpha_img=0.8, alpha=0.8, vis_jaccard=T,
                                   slide_name=names(sp_object@images)[1],
                                   slide_ncol=2, # For multiple slides
                                   title_fontsize=15, legend_fontsize=10, slide_title=NULL,
                                   add_title_text='', crop_image=T, return_plot=F, return_sp_object=F,
                                   save=F, save_path = '.', save_name_add='', dpi=100,
                                   fig_width=5, fig_height=5){
  if (length(slide_name) > 1){stop("'slide_name' should be one element of names(sp_object@images)")}
  if (!slide_name %in% names(sp_object@images)){stop("'slide_name' should be among names(sp_object@images)")}
  # Check the data type
  spatial_type <- ifelse(class(sp_object@images$image)[1]=="SlideSeq","cosmx","visium")
  # Convert the feature name if cell type specific data is provided
  if (!is.null(celltype_x) & spatial_type=='cosmx'){feat_name_x <- paste0(celltype_x,"_",feat_name_x)}
  if (!is.null(celltype_y) & spatial_type=='cosmx'){feat_name_y <- paste0(celltype_y,"_",feat_name_y)}

  # Define the 'batch' column according to the slide names: names(sp_object@images)
  df_image_all <- data.frame()
  for (slide in names(sp_object@images)){
    df_image <- data.frame(row.names = rownames(sp_object@images[[slide]]@coordinates))
    df_image[['batch']] <- slide
    df_image_all <- rbind(df_image_all, df_image)
  }
  sp_object <- Seurat::AddMetaData(sp_object, df_image_all)

  # Subset the slide by the image name: names(sp_object@images)
  Seurat::Idents(sp_object) <- "batch"
  sp_object <- subset(sp_object, idents = slide_name)
  sp_object@images <- sp_object@images[slide_name]

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
  if (spatial_type=='cosmx'){alpha_img <- 0}
  for (i in 1:top_n){
    cc_loc_xy <- reticulate::py_to_r(adata_sp_mod$obs[[paste(c("CCxy_top",i,feat_name_x,feat_name_y),collapse = "_")]]$astype('int'))
    sp_object[[paste0('CCxy_top_',i)]] <- factor(cc_loc_xy)

    # Subset the object to highlight the top i connected component locations
    sp_object_mod <- sp_object
    Seurat::Idents(sp_object_mod) <- paste0('CCxy_top_',i)
    if (spatial_type=='visium'){sp_object_mod <- subset(sp_object_mod, idents = c(feat_name_x, feat_name_y, "Over"))}

    # Assign colors and labels to the spots/grids
    color.map <- c("0"="#A2E1CA","1"="#FBBC05","2"="#4285F4","3"="#34A853")
    if (spatial_type=='visium') {
      color.map <- color.map[as.character(setdiff(levels(sp_object_mod@meta.data[[paste0('CCxy_top_',i)]]),0))]
    } else if (spatial_type=='cosmx') {
      color.map <- color.map[as.character(levels(sp_object_mod@meta.data[[paste0('CCxy_top_',i)]]))]
    }
    feature_map <- c("0"="Others","1"=feat_name_x,"2"=feat_name_y,"3"="Over")
    levels(sp_object_mod@meta.data[[paste0('CCxy_top_',i)]]) <- feature_map[as.character(levels(sp_object_mod@meta.data[[paste0('CCxy_top_',i)]]))]

    # Draw spatial cluster plot for connected component locations
    p[[i]] <- Seurat::SpatialDimPlot(sp_object_mod, alpha=alpha, image.alpha = alpha_img,
                                     label=F, crop=crop_image,
                                     pt.size.factor=dot_size,
                                     combine=FALSE)[[1]]
    p[[i]] <- p[[i]] + ggplot2::ggtitle(paste0(ifelse(is.null(slide_title), paste0(slide_name,'\n'),
                                                      ifelse(slide_title=='','',paste0(slide_title,'\n'))),
                                               feat_name_x,' & ',feat_name_y, add_title_text,": ",
                                               "top ",i," CCxy")) +
      ggplot2::scale_fill_manual(values = as.character(color.map)) +
      ggplot2::theme(plot.title=ggplot2::element_text(size=title_fontsize,hjust=0.5),
                     legend.title=ggplot2::element_blank(), legend.position='right',
                     legend.text=ggplot2::element_text(size=legend_fontsize,hjust=0.5))
    if (vis_jaccard) {p[[i]] <- p[[i]] + ggplot2::annotate("text", x=Inf, y=Inf,
                                                           label=paste0("J_comp: ",
                                                                        format(J_top[i], nsmall = 3, digits=3)),
                                                           size=legend_fontsize/3, hjust=1, vjust=1)}
  }

  # Draw plot
  if (length(p)<slide_ncol) {slide_ncol <- length(p)}
  slide_nrow <- ((length(p)-1)%/%slide_ncol)+1
  p_mod <- patchwork::wrap_plots(p, ncol=slide_ncol) + patchwork::plot_layout(guides = "collect")
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
