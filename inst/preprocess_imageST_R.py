import os
import argparse
from STopover import STopover_cosmx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sp_load_path', type=str, default='.')
    parser.add_argument('--sc_load_path', type=str, default='sc_adata.h5ad')
    parser.add_argument('--sc_celltype_colname', type=str, default='celltype')
    parser.add_argument('--ST_type', type=str, default='cosmx')
    parser.add_argument('--grid_method', type=str, default='transcript')
    parser.add_argument('--annot_method', type=str, default='ingest')
    parser.add_argument('--sc_norm_total', type=float, default=1e3)
    parser.add_argument('--min_counts', type=float, default=50)
    parser.add_argument('--min_genes', type=float, default=0)
    parser.add_argument('--tx_file_name', type=str, default='tx_file.csv')
    parser.add_argument('--cell_exprmat_file_name', type=str, default='exprMat_file.csv')
    parser.add_argument('--cell_metadata_file_name', type=str, default='metadata_file.csv')
    parser.add_argument('--fov_colname', type=str, default='fov')
    parser.add_argument('--cell_id_colname', type=str, default='cell_ID')
    parser.add_argument('--tx_xcoord_colname', type=str, default='x_global_px')
    parser.add_argument('--tx_ycoord_colname', type=str, default='y_global_px')
    parser.add_argument('--transcript_colname', type=str, default='target')
    parser.add_argument('--meta_xcoord_colname', type=str, default='CenterX_global_px')
    parser.add_argument('--meta_ycoord_colname', type=str, default='CenterY_global_px')
    parser.add_argument('--x_bins', type=int, default=100)
    parser.add_argument('--y_bins', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='.')
    args = parser.parse_args()
    
    adata_sp_all = STopover_imageST(sp_load_path=args.sp_load_path, 
                                    sc_adata=args.sc_load_path,
                                    sc_celltype_colname=args.sc_celltype_colname,
                                    ST_type=args.ST_type,
                                    grid_method=args.grid_method,
                                    annot_method=args.annot_method,
                                    sc_norm_total=args.sc_norm_total,
                                    min_counts=args.min_counts,
                                    min_genes=args.min_genes,
                                    tx_file_name=args.tx_file_name,
                                    cell_exprmat_file_name=args.cell_exprmat_file_name,
                                    cell_metadata_file_name=args.cell_metadata_file_name,
                                    fov_colname = args.fov_colname, 
                                    cell_id_colname=args.cell_id_colname,
                                    tx_xcoord_colname=args.tx_xcoord_colname,
                                    tx_ycoord_colname=args.tx_ycoord_colname,
                                    transcript_colname=args.transcript_colname,
                                    meta_xcoord_colname=args.meta_xcoord_colname,
                                    meta_ycoord_colname=args.meta_ycoord_colname,
                                    x_bins=args.x_bins, 
                                    y_bins=args.y_bins, 
                                    save_path=args.save_path)
                                   
    adata_sp_all.save_connected_loc_data(save_format='h5ad', 
                                         filename = os.path.join(args.save_path, "preprocess_"+str(args.ST_type)))
