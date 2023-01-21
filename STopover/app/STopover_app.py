import sys, os
import pandas as pd
import scanpy as sc
import pkg_resources

from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog, QApplication
from PyQt5.QtCore import QObject, Qt, QThreadPool, QRunnable, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QTextCursor, QMovie
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..STopover_class import *
from .STopover_ui import Ui_Dialog
from multiprocessing import freeze_support

class StreamOutput(QObject):
    '''
    Class to emit text
    '''
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)

class WorkerThread(QRunnable):
    '''
    Define QThread wrapper for the function
    Reference: https://gist.github.com/Orizzu/c95c170bdc88f458aa527986a2c616f6
    '''
    def __init__(self, function, *args, **kwargs):
        super(WorkerThread, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
    @pyqtSlot()    
    def run(self):
        output2 = "None"
        try: output1 = self.function(*self.args, **self.kwargs) # Export result output
        except: 
            output1 = list(self.kwargs.values())[-1] # Export message
            output2 = list(self.kwargs.keys())[-1] # Export message type
        self.signals.result.emit((output1, output2))
        self.signals.finished.emit()
       

class STopoverApp(QMainWindow, Ui_Dialog):
    '''
    Main class to run PyQt app
    '''
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.threadpool = QThreadPool()
        self.thread_startnum = 1 # Total thread number that process started
        self.thread_endnum = 1 # Total thread number that process ended

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
        self.pushButton_save_df_loc.clicked.connect(self.save_cc_loc_df)
        self.pushButton_save_df_comp.clicked.connect(self.save_j_comp_result)
        self.pushButton_apply_name.clicked.connect(self.apply_name)

        self.data_type = 'visium'
        self.load_path = './'
        self.save_path = './'
        self.stopover_class = AnnData()
        self.df_feat = pd.DataFrame([])
        self.df_j_comp = pd.DataFrame([])
        self.sc_adata = None
        self.name_prefix = ''
        # Spinner loading gif generated from https://loading.io/
        self.load_gif = pkg_resources.resource_filename(__name__, 'image/Spinner_loading.io.gif')

        sys.stdout = StreamOutput()
        sys.stdout.text_written.connect(self.normal_output_written)

        plt.ion() # Turn on ion mode
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
    
    def run_thread(func):
        def decorated_func(self, *args, **kwargs):
            # Load gif file for the file loader (generated from https://loading.io/)
            if self.thread_startnum == self.thread_endnum:
                # Run movie only when the process is not overlapped
                self.movie = QMovie(self.load_gif)
                self.label_log.setMovie(self.movie)
                self.movie.start()
            self.thread_startnum += 1
            worker = WorkerThread(func, self, *args, **kwargs)
            worker.signals.result.connect(self.thread_result)
            worker.signals.finished.connect(self.thread_end)
            self.threadpool.start(worker) # start thread
        return decorated_func
    
    def thread_result(self, output):
        if isinstance(output[0], STopover_cosmx) or isinstance(output[0], STopover_visium): 
            self.stopover_class = output[0]
        elif isinstance(output[0], str):
            if output[1] == "error_output": self.show_error_message(output[0])
            elif output[1] == "info_output": self.show_info_message(output[0])
    
    def thread_end(self): 
        self.thread_endnum += 1
        if self.thread_startnum == self.thread_endnum: self.movie.stop()
    
    @run_thread
    def STopover_visium_(self, sp_load_path, lognorm, min_size, fwhm, thres_per, save_path, print_output="Finished", error_output=""):
        object = STopover_visium(sp_load_path=sp_load_path, lognorm=lognorm, 
                                 min_size=min_size, fwhm=fwhm, thres_per=thres_per, save_path=save_path)
        print(print_output)
        return object

    @run_thread
    def STopover_cosmx_(self, sp_load_path, sc_adata, sc_celltype_colname, tx_file_name, cell_exprmat_file_name, 
                        cell_metadata_file_name, min_size, fwhm, thres_per, save_path, print_output="Finished", error_output=""):
        object = STopover_cosmx(sp_load_path=sp_load_path, sc_adata=sc_adata, 
                                sc_celltype_colname=sc_celltype_colname, tx_file_name=tx_file_name, 
                                cell_exprmat_file_name=cell_exprmat_file_name, 
                                cell_metadata_file_name=cell_metadata_file_name,
                                min_size=min_size, fwhm=fwhm, thres_per=thres_per, save_path=save_path)
        print(print_output)
        return object
    
    @run_thread
    def topological_sim_(self, object, feat_pairs, use_lr_db, lr_db_species, error_output=""):
        return object.topological_similarity(feat_pairs=feat_pairs, use_lr_db=use_lr_db, lr_db_species=lr_db_species)

    @run_thread
    def save_connected_loc_data_(self, object, save_format, filename, print_output="Finished", error_output=""):
        object.save_connected_loc_data(save_format=save_format, filename=filename)
        print(print_output)

    @run_thread
    def save_j_comp_result_(self, j_result, save_path, filename, print_output="Finished", error_output=""):
        j_result.to_csv(os.path.join(save_path, filename+'_J_comp_feats_df.csv'), sep = ',', header=True, index=True)
        print(print_output)
          
    # https://stackoverflow.com/questions/8356336/how-to-capture-output-of-pythons-interpreter-and-show-in-a-text-widget
    def closeEvent(self, event: QCloseEvent):
        sys.stdout = sys.__stdout__
        event.accept()

    def normal_output_written(self, text):
        cursor = self.textEdit_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.textEdit_output.setTextCursor(cursor)
        self.textEdit_output.ensureCursorVisible()

    def show_error_message(self, msg=""):
        QMessageBox.critical(self, "Error", msg)

    def show_info_message(self, msg=""):
        QMessageBox.information(self, "Information", msg)

    def draw_on_canvas(func):
        def decorated_func(self):
            self.fig.clear()
            sc.set_figure_params(facecolor='white', frameon=False)
            output = func(self)
            self.canvas.draw()
            return output
        return decorated_func
    
    def open_files(self, data_type='visium', open_type='file'):
        assert data_type in ['visium', 'cosmx', 'csv', 'single-cell', 'save']
        assert open_type in ['dir', 'file']
        
        if data_type in ['visium', 'cosmx']:
            if open_type=='dir':
                file_name = [str(QFileDialog.getExistingDirectory(self, "Select folder"))]
                if file_name[0] != '':
                    self.textEdit_load1.setText(data_type+'_'+open_type+':')
                    self.textEdit_load2.setText(file_name[0])
                    self.load_path = file_name[0]
                    self.data_type = data_type
            else:
                file_name = QFileDialog.getOpenFileName(self, "Open file", './', 'H5ad file(*.h5ad)')
                if file_name[0] != '':
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
            if file_name[0] != '': self.textEdit_ref.setText(file_name[0])
        else:
            file_name = [str(QFileDialog.getExistingDirectory(self, "Select folder"))]
            if file_name[0] != '':
                self.textEdit_load3.setText(file_name[0])
                self.save_path = file_name[0]
    
    def extract_j_comp_result(self, info_output=""):
        J_result_key = sorted([key for key in self.stopover_class.uns.keys() if key.startswith('J_result')])
        if len(J_result_key) > 0:
            df_j = self.stopover_class.uns[J_result_key[-1]].loc[:,['Feat_1','Feat_2','J_comp']].sort_values(by='J_comp',ascending=False)
            self.df_j_comp = df_j
        else: self.show_info_message(info_output)

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
        self.extract_j_comp_result(info_output="J_comp table cannot be updated\n Possibly the data is missing")
        self.textEdit_J_result.setText(self.df_j_comp.to_string())
        print("Info updated")

    def create_stopover(self):
        print("Creating STopover object")
        if self.data_type == 'visium':
            not_found_element = []
            if not self.load_path.endswith('.h5ad'):
                required_list = ['filtered_feature_bc_matrix.h5',
                                './spatial/tissue_positions_list.csv','./spatial/scalefactors_json.json',
                                './spatial/tissue_hires_image.png', './spatial/tissue_lowres_image.png']
                log_norm = True
                for file in required_list:
                    if not os.path.exists(os.path.join(self.load_path, file)): not_found_element.append(file)
            else: log_norm = False

            if len(not_found_element) > 0:
                self.show_error_message('%s not found in the given directory'% ', '.join(not_found_element))
            else: 
                self.STopover_visium_(sp_load_path=self.load_path, lognorm=log_norm, 
                                      min_size=self.doubleSpinBox_min_size.value(), 
                                      fwhm=self.doubleSpinBox_fwhm.value(), 
                                      thres_per=self.doubleSpinBox_per_thres.value(), 
                                      save_path=self.save_path, 
                                      print_output="STopover object created for "+self.data_type,
                                      error_output="Error in running STopover for "+self.data_type)

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
                self.STopover_cosmx_(sp_load_path=self.load_path,
                                         sc_adata=self.sc_adata, sc_celltype_colname = self.comboBox_ref.currentText(),
                                         tx_file_name = tx_file_name[0], cell_exprmat_file_name=cell_exprmat_file_name[0], 
                                         cell_metadata_file_name=cell_metadata_file_name[0], 
                                         min_size=self.doubleSpinBox_min_size.value(), 
                                         fwhm=self.doubleSpinBox_fwhm.value(), 
                                         thres_per=self.doubleSpinBox_per_thres.value(), 
                                         save_path=self.save_path, print_output="STopover object created for "+self.data_type,
                                         error_output = "Error in running STopover for "+self.data_type)

    def run_stopover(self):
        print("Running STopover")
        if self.stopover_class.shape[0]>0:
            self.topological_sim_(self.stopover_class, feat_pairs=self.df_feat, 
                                  use_lr_db=self.checkBox.isChecked(), 
                                  lr_db_species=self.comboBox_spec.currentText(),
                                  error_output="Failed to calculate topological similarity\n Possibly none of the first or second features in the pairs are found")
        else: self.show_error_message("Data files are not loaded yet")

    @draw_on_canvas
    def draw_plot(self):
        if self.comboBox_vis_type.currentText() == "CC location":
            print("Drawing CC location plot")
            try: self.stopover_class.vis_all_connected(feat_name_x=self.comboBox_feat_x.currentText(),
                                                       feat_name_y=self.comboBox_feat_y.currentText(), fig_size=(6,6))
            except: self.show_error_message("Error in drawing locations of all CCs")
        elif self.comboBox_vis_type.currentText() == "Spatial plot":
            print("Drawing spatial plot for feature x")
            if self.data_type=='cosmx':
                try: self.stopover_class.vis_spatial_cosmx(feat_name=self.comboBox_feat_x.currentText(), fig_size=(6,6))
                except: self.show_error_message("Error in drawing spatial plot for feature x")
            elif self.data_type=='visium':
                try:
                    f, ax = plt.subplots(1, 1, tight_layout=True)
                    sc.pl.spatial(self.stopover_class, color=self.comboBox_feat_x.currentText(), ax=ax)
                except: self.show_error_message("Error in drawing spatial plot for feature x")
        elif self.comboBox_vis_type.currentText() == "Top n CC location":
            print("Drawing locations for top "+str(self.spinBox_top_n.value())+" overlapping CCx - CCy pairs")
            try: self.stopover_class.vis_jaccard_top_n_pair(feat_name_x=self.comboBox_feat_x.currentText(),
                                                            feat_name_y=self.comboBox_feat_y.currentText(), 
                                                            top_n=self.spinBox_top_n.value(), fig_size=(6,6))
            except: self.show_error_message("Error in drawing locations for top "+str(self.spinBox_top_n.value())+" overlapping CCx - CCy pairs")
    
    def apply_name(self):
        self.name_prefix = self.textEdit_prefix_name.toPlainText()
        print("%s was saved as prefix" % self.name_prefix)

    def save_anndata(self):
        print("Saving result anndata (.h5ad)")
        self.save_connected_loc_data_(self.stopover_class, save_format='h5ad', filename = self.name_prefix+'_cc_location',
                                      error_output="Error in saving STopover object anndata")

    def save_cc_loc_df(self):
        print("Saving CC location dataframe (.csv)")
        self.save_connected_loc_data_(self.stopover_class, save_format='csv', filename = self.name_prefix+'_cc_location',
                                      error_output="Error in saving jaccard similarity result dataframe")
    
    def save_j_comp_result(self):
        print("Saving J_comp results dataframe (.csv)")
        self.save_j_comp_result_(self.df_j_comp, save_path=self.save_path, filename=self.name_prefix,
                                 error_output="Error in saving J_comp result dataframe")


def main():
    app = QApplication(sys.argv)
    mywindow = STopoverApp()
    mywindow.show()
    app.exec_()


if __name__ == "__main__": 
    freeze_support()
    main()
