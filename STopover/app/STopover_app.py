import sys, os
import pandas as pd
import scanpy as sc
import functools

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..STopover_class import *
from .STopover_ui import Ui_Dialog

class StreamOutput(QtCore.QObject):
    '''
    Class to emit text
    '''
    text_written = QtCore.pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))


class STopoverApp(QMainWindow, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.comboBox_feat_x.setMaxVisibleItems(10)
        self.comboBox_feat_y.setMaxVisibleItems(10)
        self.comboBox_ref.setMaxVisibleItems(10)

        self.pushButton_visdir.clicked.connect(lambda: self.open_files('visium', 'dir'))
        self.pushButton_visfile.clicked.connect(lambda: self.open_files('visium', 'file'))
        self.pushButton_cosmxdir.clicked.connect(lambda: self.open_files('cosmx', 'dir'))
        self.pushButton_cosmxfile.clicked.connect(lambda: self.open_files('cosmx', 'file'))
        self.pushButton_save_dir.clicked.connect(lambda: self.open_files('save', 'dir'))
        self.pushButton_ref.clicked.connect(lambda: self.open_files('single-cell', 'file'))
        self.pushButton_start1.clicked.connect(self.create_stopover)
        self.pushButton_start2.clicked.connect(self.run_stopover)
        self.pushButton_csv.clicked.connect(lambda: self.open_files('csv', 'file'))
        self.pushButton_vis.clicked.connect(self.draw_plot)
        
        self.pushButton_refresh.clicked.connect(self.refresh)
        self.pushButton_save_ann.clicked.connect(self.save_anndata)
        self.pushButton_save_df.clicked.connect(self.save_cc_loc_df)

        self.data_type = 'visium'
        self.load_path = './'
        self.save_path = './'
        self.stopover_class = AnnData()
        self.df_feat = pd.DataFrame([])
        self.sc_adata = None

        sys.stdout = StreamOutput()
        sys.stdout.text_written.connect(self.normal_output_written)

        self.fig = plt.Figure((10,10))
        self.canvas = FigureCanvas(self.fig)

    # https://stackoverflow.com/questions/8356336/how-to-capture-output-of-pythons-interpreter-and-show-in-a-text-widget
    def closeEvent(self, event: QtGui.QCloseEvent):
        sys.stdout = sys.__stdout__
        event.accept()
        
    def normal_output_written(self, text):
        cursor = self.textEdit_output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit_output.setTextCursor(cursor)
        self.textEdit_output.ensureCursorVisible()

    def show_error_message(self, msg=""):
        QMessageBox.critical(self, "Error", msg)

    def loading_movie(func):
        @functools.wraps(func)
        def decorated_func(self):
            # Start loading image
            movie = QtGui.QMovie("Spinner_loading.io.gif")
            self.label_log.setMovie(movie)
            movie.start()
            output = func(self)
            movie.stop()
            return output
        return decorated_func

    def draw_on_canvas(func):
        @functools.wraps(func)
        def decorated_func(self, *args, **kwargs):
            self.fig.clear()
            sc.set_figure_params(figsize=(10, 10), facecolor='white', frameon=False)
            output = func(self, *args, **kwargs)
            self.canvas.draw()
            return output
        return decorated_func
    
    def open_files(self, data_type='visium', open_type='file'):
        assert data_type in ['visium', 'cosmx', 'csv', 'single-cell', 'save']
        assert open_type in ['dir', 'file']
        
        if data_type in ['visium', 'cosmx']:
            if open_type=='dir':
                file_name = [str(QFileDialog.getExistingDirectory(self, "Select folder"))]
                self.textEdit_load1.setText(data_type+'_'+open_type+':')
                self.textEdit_load2.setText(file_name[0])
                self.load_path = file_name[0]
                self.data_type = data_type
            else:
                file_name = QFileDialog.getOpenFileName(self, "Open file", './', 'H5ad file(*.h5ad)')
                self.textEdit_load1.setText(data_type+'_'+open_type+':')
                self.textEdit_load2.setText(file_name[0])
                self.load_path = file_name[0]
                self.data_type = data_type
        elif data_type == 'csv':
            file_name = QFileDialog.getOpenFileName(self, "Load feature pairs",'./',
                                                          'CSV file(*.csv);; TSV file(*.tsv);; TXT file(*.txt)')
            if file_name[0] != '':
                if file_name[0].endswith('.csv'): delim = ','
                elif file_name[0].endswith('.tsv'): delim = '/t'
                else: delim = ' '
                self.df_feat = pd.read_csv(file_name[0], usecols=[0, 1], delimiter=delim)
                self.textEdit_csv.setText(self.df_feat.to_string())
                self.checkBox.setChecked(False)
        elif data_type == 'single-cell':
            file_name = QFileDialog.getOpenFileName(self, "Open file", './', 'H5ad file(*.h5ad)')
            self.textEdit_ref.setText(file_name[0])
        else:
            file_name = [str(QFileDialog.getExistingDirectory(self, "Select folder"))]
            self.textEdit_load3.setText(file_name[0])
            self.save_path = file_name[0]        

    
    def refresh(self):
        self.comboBox_feat_x.clear()
        self.comboBox_feat_y.clear()
        self.comboBox_ref.clear()
        feature_list = [feat.split("Comb_CC_")[1] for feat in self.stopover_class.obs.columns if feat.startswith('Comb_CC_')]
        self.comboBox_feat_x.addItems(feature_list)
        self.comboBox_feat_y.addItems(feature_list)
        if (self.textEdit_ref.toPlainText() != "") and (self.textEdit_ref.toPlainText() is not None):
            self.sc_adata = sc.read_h5ad(self.textEdit_ref.toPlainText())
            self.comboBox_ref.addItems(list(self.sc_adata.obs.columns))

    @loading_movie
    def create_stopover(self):
        print("Creating STopover object")
        if self.data_type == 'visium':
            not_found_element = []
            if not self.load_path.endswith('.h5ad'):
                required_list = ['filtered_feature_bc_matrix.h5',
                                './spatial/tissue_positions_list.csv','./spatial/scalefactors_json.json',
                                './spatial/tissue_hires_image.png', './spatial/tissue_lowres_image.png']

                for file in required_list:
                    if not os.path.exists(os.path.join(self.load_path, file)): not_found_element.append(file)

            if len(not_found_element) > 0:
                self.show_error_message('%s not found in the given directory'% ', '.join(not_found_element))
            else: 
                try: 
                    self.stopover_class = STopover_visium(sp_load_path=self.load_path, lognorm=False, 
                                                          min_size=self.doubleSpinBox_min_size.value(), 
                                                          fwhm=self.doubleSpinBox_fwhm.value(), 
                                                          thres_per=self.doubleSpinBox_per_thres.value(), 
                                                          save_path=self.save_path)
                    print("STopover object created for "+self.data_type)
                except: self.show_error_message("Error in running STopover for "+self.data_type)

        elif self.data_type == 'cosmx':
            if (self.textEdit_ref.toPlainText() == "") or (self.textEdit_ref.toPlainText() is None): self.sc_adata = None
            else: self.sc_adata = sc.read_h5ad(self.textEdit_ref.toPlainText())
            
            not_found_element = []
            if not self.load_path.endswith('.h5ad'):
                if self.load_path == '': self.load_path = '.'
                file_list = os.listdir(self.load_path)
                tx_file_name = [i for i in file_list if i.endswith('tx_file.csv')]
                cell_exprmat_file_name = [i for i in file_list if i.endswith('exprMat_file.csv')]
                cell_metadata_file_name = [i for i in file_list if i.endswith('metadata_file.csv')]

                if len(tx_file_name)==0: not_found_element.append('Transcript file (~tx_file.csv)')
                if len(cell_exprmat_file_name)==0: not_found_element.append('Cell-level expression matrix (~exprMat_file.csv)')
                if len(cell_metadata_file_name)==0: not_found_element.append('Cell-level metadata file (~metadata_file.csv)')
            else:
                tx_file_name, cell_exprmat_file_name, cell_metadata_file_name = [None], [None], [None]

            if len(not_found_element) > 0:
                self.show_error_message('%s not found in the given directory'% ', '.join(not_found_element))
            else:
                try:
                    self.stopover_class = STopover_cosmx(sp_load_path=self.load_path,
                                                         sc_adata=self.sc_adata, sc_celltype_colname = self.comboBox_spec.currentText(),
                                                         tx_file_name = tx_file_name[0], cell_exprmat_file_name=cell_exprmat_file_name[0], 
                                                         cell_metadata_file_name=cell_metadata_file_name[0], 
                                                         min_size=self.doubleSpinBox_min_size.value(), 
                                                         fwhm=self.doubleSpinBox_fwhm.value(), 
                                                         thres_per=self.doubleSpinBox_per_thres.value(), 
                                                         save_path=self.save_path)
                    print("STopover object created for "+self.data_type)
                except: self.show_error_message("Error in running STopover for "+self.data_type)

    @loading_movie
    def run_stopover(self):
        print("Running STopover")
        if self.stopover_class.shape[0]>0:
            try: self.stopover_class.topological_similarity(feat_pairs=self.df_feat, 
                                                            use_lr_db=self.checkBox.isChecked(), 
                                                            lr_db_species=self.comboBox_spec.currentText())
            except: self.show_error_message("None of the first or second features in the pairs are found")

    @loading_movie
    @draw_on_canvas
    def draw_plot(self):
        J_result_key = sorted([key for key in self.stopover_class.uns.keys() if key.startswith('J_result')])
        if len(J_result_key) > 0:
            # Visualize the J_result in the window
            df_j = self.stopover_class.uns[J_result_key[-1]].loc[:,['Feat_1','Feat_2','J_comp']]
            self.textEdit_J_result.setText(df_j.to_string())

        if self.comboBox_vis_type.currentText() == "CC location":
            print("Drawing CC location plot")
            try: self.stopover_class.vis_all_connected(feat_name_x=self.comboBox_feat_x.currentText(),
                                                       feat_name_y=self.comboBox_feat_y.currentText())
            except: self.show_error_message("Error in drawing locations of all CCs")
        elif self.comboBox_vis_type.currentText() == "Spatial plot":
            print("Drawing spatial plot for feature x")
            if self.data_type=='cosmx':
                try: self.stopover_class.vis_spatial_cosmx(feat_name=self.comboBox_feat_x.currentText())
                except: self.show_error_message("Error in drawing spatial plot for feature x")
            elif self.data_type=='visium':
                try: sc.pl.spatial(self.stopover_class, color=self.comboBox_feat_x.currentText())
                except: self.show_error_message("Error in drawing spatial plot for feature x")
        elif self.comboBox_vis_type.currentText() == "Top n CC location":
            print("Drawing locations for top 4 overlapping CCx - CCy pairs")
            try: self.stopover_class.vis_jaccard_top_n_pair(feat_name_x=self.comboBox_feat_x.currentText(),
                                                            feat_name_y=self.comboBox_feat_y.currentText(), 
                                                            top_n=self.spinBox_top_n.value())
            except: self.show_error_message("Error in drawing locations for top "+str(self.spinBox_top_n.value())+" overlapping CCx - CCy pairs")

    @loading_movie
    def save_anndata(self):
        print("Saving result anndata (.h5ad)")
        if self.stopover_class.shape[0]>0:
            try: self.stopover_class.save_connected_loc_data(save_format='h5ad', filename = 'cc_location')
            except: self.show_error_message("Error in saving STopover object anndata")
    
    @loading_movie
    def save_cc_loc_df(self):
        print("Saving result dataframe (.csv)")
        if self.stopover_class.shape[0]>0:
            try: self.stopover_class.save_connected_loc_data(save_format='csv', filename = 'cc_location')
            except: self.show_error_message("Error in saving jaccard similarity result dataframe")

def main():
    app = QApplication(sys.argv)
    mywindow = STopoverApp()
    mywindow.show()
    app.exec_()

if __name__ == "__main__": 
    main()