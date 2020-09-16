import sys
import os
import numpy as np
import glob
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import PyQt5.QtWidgets as qw


def extract_value(column):
    value_series = column[column.notnull()]
    value_list = value_series.tolist()
    return value_list


def extract_value_each_csv(csv_file):
    each_file = pd.read_csv(csv_file, engine='python', header=10, error_bad_lines=True, index_col=0)
    each_file_values = each_file.apply(lambda cols: extract_value(column=cols))
    value_lists = [x for l in each_file_values.tolist() for x in l]
    return value_lists


class ROIAnalyzer(qw.QMainWindow):
    b_width, w_itv = 20, 32
    h_itv, v_itv = 30, 280  # horizontal interval, vertical interval
    lbl_wdw = 120
    is_analysis = True

    g1, g2 = 10, 10+3*h_itv
    g3, g4 = 110, 30
    m1 = 30

    total_lists, total_all_values = None, None
    select_all_values = None
    v_files, v_values = None, None
    v_mean, v_std = None, None
    v_median, v_mode = None, None
    v_min, v_max = None, None
    v_skew, v_kurt = None, None
    v_entropy, v_per = None, None
    adjust_value = None  # n% percentile
    default_n = 'n'  # default n% percentile

    all_min, all_max = None, None
    min_default, max_default = None, None
    is_hist = False
    is_line = False
    p_line = None

    per_all_value = True  # Whether to use all values when calculating percentiles

    set_wd = None

    def __init__(self):
        super().__init__()

        self.dir_desc = qw.QLabel(self)
        self.dir_desc.setGeometry(10, 10, 110, 30)
        self.dir_desc.move(30, self.b_width)
        self.dir_desc.setText('Select Folder:')

        self.dir_name = qw.QLineEdit(self)
        self.dir_name.setGeometry(100, 10+self.h_itv, 780, 30)
        self.dir_name.move(150, self.b_width)
        self.dir_name.setText(self.set_wd)
        self.dir_name.setReadOnly(True)

        self.dir_browse = qw.QPushButton('Browse', self)
        self.dir_browse.move(940, self.b_width)
        self.dir_browse.clicked.connect(self.set_directory)
        self.dir_browse.setShortcut('Ctrl+O')

        self.info_browse = qw.QPushButton('Info', self)
        self.info_browse.move(1180, self.b_width)
        self.info_browse.clicked.connect(self.view_info)
        self.info_browse.setShortcut('Ctrl+I')

        self.basic_stats = qw.QLabel(self)
        self.basic_stats.setGeometry(10, 10+2*self.h_itv, 110,  30)
        self.basic_stats.move(30, self.b_width+2*self.w_itv)
        self.basic_stats.setText('Basic Statistics')

        self.n_files, self.n_files_window = \
            self.show_label(desc='# of Files: ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1, m2=self.b_width+3*self.w_itv,
                            wm1=self.m1+self.lbl_wdw)

        self.n_values, self.n_values_window = \
            self.show_label(desc='# of Pixels: ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1+self.v_itv, m2=self.b_width+3*self.w_itv,
                            wm1=self.m1+self.lbl_wdw+self.v_itv)

        self.n_mean, self.n_mean_window = \
            self.show_label(desc='Mean: ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1, m2=self.b_width+4*self.w_itv,
                            wm1=self.m1+self.lbl_wdw)

        self.n_std, self.n_std_window = \
            self.show_label(desc='Standard Deviation: ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1+self.v_itv, m2=self.b_width+4*self.w_itv,
                            wm1=self.m1+self.lbl_wdw+self.v_itv)

        self.n_median, self.n_median_window = \
            self.show_label(desc='Median: ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1, m2=self.b_width+5*self.w_itv,
                            wm1=self.m1+self.lbl_wdw)

        self.n_mode, self.n_mode_window = \
            self.show_label(desc='Mode (Count): ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1+self.v_itv, m2=self.b_width+5*self.w_itv,
                            wm1=self.m1+self.lbl_wdw+self.v_itv)

        self.n_min, self.n_min_window = \
            self.show_label(desc='Minimum: ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1, m2=self.b_width+6*self.w_itv,
                            wm1=self.m1+self.lbl_wdw)

        self.n_max, self.n_max_window = \
            self.show_label(desc='Maximum: ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1+self.v_itv, m2=self.b_width+6*self.w_itv,
                            wm1=self.m1+self.lbl_wdw+self.v_itv)

        self.n_skew, self.n_skew_window = \
            self.show_label(desc='Skewness: ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1, m2=self.b_width+7*self.w_itv,
                            wm1=self.m1+self.lbl_wdw)

        self.n_kurt, self.n_kurt_window = \
            self.show_label(desc='Kurtosis: ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1+self.v_itv, m2=self.b_width+7*self.w_itv,
                            wm1=self.m1+self.lbl_wdw+self.v_itv)

        self.n_entropy, self.n_entropy_window = \
            self.show_label(desc='Entropy: ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1, m2=self.b_width+8*self.w_itv,
                            wm1=self.m1+self.lbl_wdw)

        self.n_per, self.n_per_window, self.n_per_button = \
            self.show_label(desc='Percentile: ',
                            g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4,
                            m1=self.m1+self.v_itv, m2=self.b_width+8*self.w_itv,
                            wm1=self.m1+self.lbl_wdw+self.v_itv, add_button=True)

        self.n_min_range, self.n_min_range_window = \
            self.show_label(desc='Min Range: ',
                            g1=self.g1, g2=self.g2, g3=self.g3-25, g4=self.g4,
                            m1=self.m1, m2=self.b_width + 10 * self.w_itv,
                            wm1=self.m1+75, read_only=False)

        self.n_max_range, self.n_max_range_window = \
            self.show_label(desc='Max Range: ',
                            g1=self.g1, g2=self.g2, g3=self.g3-25, g4=self.g4,
                            m1=self.m1 + 175, m2=self.b_width + 10 * self.w_itv,
                            wm1=self.m1+75+175, read_only=False)

        self.remove_noise = qw.QPushButton('Calculate', self)
        self.remove_noise.setGeometry(self.g1, self.g2, 160, self.g4)
        self.remove_noise.move(380, self.b_width + 10*self.w_itv)
        self.remove_noise.clicked.connect(self.cal_remove_noise)
        self.remove_noise.setShortcut('Alt+c')

        self.pixel_value, self.pixel_value_window = \
            self.show_label(desc='Pixel value: ', g1=self.g1, g2=self.g2, g3=self.g3-25, g4=self.g4,
                            m1=self.m1, m2=self.b_width + 11 * self.w_itv,
                            wm1=self.m1+75, read_only=False)

        self.pixel_prop, self.pixel_prop_window = \
            self.show_label(desc='Pixel prop: ', g1=self.g1, g2=self.g2, g3=self.g3-25, g4=self.g4,
                            m1=self.m1 + 175, m2=self.b_width + 11 * self.w_itv,
                            wm1=self.m1 + 75 + 175, read_only=True)

        self.pixel_prop_bt = qw.QPushButton('Calculate pixel ratio', self)
        self.pixel_prop_bt.setGeometry(self.g1, self.g2, 160, self.g4)
        self.pixel_prop_bt.move(380, self.b_width + 11*self.w_itv)
        self.pixel_prop_bt.clicked.connect(self.cal_pixel_prop)
        self.pixel_prop_bt.setShortcut('Alt+r')

        self.fig = plt.Figure(figsize=(7.2, 3.6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.move(560, 90)
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.xaxis.label.set_fontsize(2)
        self.ax.yaxis.label.set_fontsize(2)
        self.fig.tight_layout()
        self.canvas.setParent(self)

    def cal_pixel_prop(self):
        try:
            ref_pixel = float(self.pixel_value_window.text())
        except:
            self.popup_box('Error!', 'Please enter a number between min range and max range.')
            return
        min_r = float(self.n_min_range_window.text())
        max_r = float(self.n_max_range_window.text())

        if ref_pixel < min_r or ref_pixel > max_r:
            self.popup_box('Error!', 'Please enter a number between min range and max range.')

        else:
            pixel_all_values = list(x for x in self.total_all_values if min_r <= x <= max_r)

            if len(pixel_all_values) > 0:
                greater_equal_values = list(x for x in pixel_all_values if x >= ref_pixel)

                greater_equal_ratio = len(greater_equal_values) / len(pixel_all_values)
                greater_equal_percent = greater_equal_ratio*100
                self.pixel_prop_window.setText('%.2f' % greater_equal_percent + '%')

                if self.is_line:
                    self.p_line.remove()

                self.p_line = self.ax.axvline(ref_pixel, color='deepskyblue')
                self.is_line = True
                self.canvas.draw()

            else:
                self.popup_box('Error!', 'The value corresponding to the condition does not exist.')

    def cal_remove_noise(self):
        try:
            min_r = float(self.n_min_range_window.text())
            max_r = float(self.n_max_range_window.text())
        except:
            self.popup_box('Error!', 'Enter a number in the min range and max range')
            return

        if min_r > max_r:
            self.popup_box('Error!', 'Minimum value is greater than maximum value.')
        else:
            self.select_all_values = list(x for x in self.total_all_values if min_r <= x <= max_r)

            if len(self.select_all_values) > 0:
                self.show_results(self.csv_list, self.select_all_values)

                if min_r <= self.all_min and max_r >= self.all_max:
                    self.draw_hist(self.select_all_values, color='indigo')
                else:
                    self.draw_hist(self.select_all_values, color='darkseagreen')

                self.n_per_button.setText(self.default_n + '%')
                self.n_per_window.setText('')  # default setting

                if self.per_all_value:
                    self.n_per_button.clicked.disconnect()
                    self.n_per_button.clicked.connect(lambda: self.cal_percentile(self.select_all_values))
                    self.n_per_button.setShortcut('Alt+n')
                    self.per_all_value = False  # selected pixel values mode

            else:
                self.popup_box('Error!', 'The value corresponding to the condition does not exist.')

    def show_label(self, desc, g1, g2, g3, g4, m1, m2, wm1, add_button=False, read_only=True):
        n_label = qw.QLabel(self)
        n_label.setGeometry(g1, g2, g3, g4)
        n_label.move(m1, m2)
        n_label.setText(desc)

        n_label_window = qw.QLineEdit(self)
        n_label_window.setGeometry(g1, g2, g3, g4)
        n_label_window.move(wm1, m2)
        n_label_window.setReadOnly(read_only)

        if add_button:
            n_label_button = qw.QPushButton(self.default_n + '%', self)
            n_label_button.setGeometry(g1, g2, 50, 30)
            n_label_button.move(m1+65, m2)
            n_label_button.clicked.connect(lambda: self.cal_percentile(self.total_all_values))
            n_label_button.setShortcut('Alt+n')

            return n_label, n_label_window, n_label_button
        else:
            return n_label, n_label_window

    def cal_percentile(self, pixel_values):
        if self.total_all_values is None:
            self.popup_box("Error!", "Please select a folder first")
        else:
            adjust, ok = qw.QInputDialog.getDouble(self, 'Adjust value', 'Enter n% percentile',
                                                   decimals=2)
            if ok:
                if adjust < 0 or adjust > 100:
                    self.popup_box('Error!', 'Percentiles must be in the range [0, 100].')
                else:
                    self.adjust_value = '%.2f' % adjust + '%'
                    self.n_per_button.setText(self.adjust_value)
                    self.v_per = '%d' % np.percentile(pixel_values, adjust)
                    self.n_per_window.setText(self.v_per)
                    self.n_per_button.setShortcut('Alt+n')
            else:
                return

    def set_directory(self):
        self.set_wd = qw.QFileDialog.getExistingDirectory(self, 'Open Folder', self.set_wd)

        if self.set_wd == '':
            return
        else:
            self.dir_name.setText(self.set_wd)
            if os.path.exists(self.set_wd):
                self.csv_list = glob.glob(os.path.join(self.set_wd+'/*.csv'))

                if len(self.csv_list) > 0:
                    self.total_lists = list(map(extract_value_each_csv, self.csv_list))
                    self.total_all_values = [x for y in self.total_lists for x in y]

                    if len(self.total_all_values) > 0:
                        self.show_results(self.csv_list, self.total_all_values)  # Calculate statistics
                        self.draw_hist(self.total_all_values)  # Draw histogram

                        self.all_min, self.all_max = np.min(self.total_all_values), np.max(self.total_all_values)

                        if self.per_all_value is False:
                            self.n_per_button.clicked.disconnect()
                            self.n_per_button.clicked.connect(lambda: self.cal_percentile(self.total_all_values))
                            self.n_per_button.setShortcut('Alt+n')
                            self.per_all_value = True  # all pixel values mode

                    else:
                        self.popup_box('Error!', 'The value corresponding to the condition does not exist.')
                else:
                    self.popup_box("Error!", "Select the folder that contains .csv file.")

    def show_results(self, csv_list, pixel_values):
        # 1 Calculate number of files, number of pixel values
        self.v_files = '%d' % len(csv_list)
        self.v_values = '%d' % len(pixel_values)

        # 2 Calculate mean & standard deviation
        self.v_mean = '%.4f' % np.mean(pixel_values)

        if len(pixel_values) == 1:
            self.v_std = 'Not countable'
        else:
            self.v_std = '%.4f' % np.std(pixel_values)

        # 3 Calculate median % mode
        if np.isnan(np.median(pixel_values)):
            self.v_median = 'Not countable'
        else:
            self.v_median = '%d' % np.median(pixel_values)
        v_mode = scipy.stats.mode(pixel_values)
        self.v_mode = '%d' % v_mode.mode
        v_mode_count = '%d' % v_mode.count
        v_mode_summary = self.v_mode + ' (' + v_mode_count + ')'

        # 4 Calculate minimum % maximum
        self.v_min = '%d' % np.min(pixel_values)
        self.v_max = '%d' % np.max(pixel_values)

        # 5 calculate skewness & kurtosis
        self.v_skew = '%.4f' % scipy.stats.skew(np.array(pixel_values))
        self.v_kurt = '%.4f' % scipy.stats.kurtosis(pixel_values)

        # 6 calculate entropy\
        _, value_counts = np.unique(pixel_values, return_counts=True)
        value_pmf = value_counts / len(pixel_values)

        self.v_entropy = '%.4f' % -np.sum(value_pmf * np.log(value_pmf))

        # Show all variables
        self.n_files_window.setText(self.v_files)
        self.n_values_window.setText(self.v_values)

        self.n_mean_window.setText(self.v_mean)
        self.n_std_window.setText(self.v_std)

        self.n_median_window.setText(self.v_median)
        self.n_mode_window.setText(v_mode_summary)

        self.n_min_window.setText(self.v_min)
        self.n_max_window.setText(self.v_max)

        self.n_skew_window.setText(self.v_skew)
        self.n_kurt_window.setText(self.v_kurt)

        self.n_entropy_window.setText(self.v_entropy)
        self.n_per_button.setText(self.default_n + '%')  # default setting
        self.n_per_window.setText('')  # default setting

        # Initialize min, max range
        self.n_min_range_window.setText(self.v_min)
        self.n_max_range_window.setText(self.v_max)
        self.pixel_value_window.setText(self.v_min)
        self.pixel_prop_window.setText('')

        self.is_line = False

    def draw_hist(self, pixel_values, color='indigo'):
        if self.is_hist:
            self.ax.clear()

        pixel_min, pixel_max = int(np.min(pixel_values)), int(np.max(pixel_values))
        pixel_range = pixel_max - pixel_min
        bin_number = 20

        if pixel_range < 100:
            bin_number = int(pixel_range / 5) + 1
        # if n_itv > 0:
        self.ax.hist(pixel_values, color=color, bins=bin_number)
        self.ax.tick_params(axis='y', labelsize=8)

        x_axis_label = np.linspace(pixel_min, pixel_max, bin_number, dtype=int)
        self.ax.set_xticks(x_axis_label)
        self.ax.set_xticklabels(labels=x_axis_label, fontsize=8)
        self.ax.set_title('Histogram of Texture', size=10)

        self.canvas.draw()
        self.is_hist = True
        # else:
        #     self.popup_box('Error!', 'Too narrow pixel range entered.')

    def popup_box(self, popup_title, popup_message):
        popup_message = qw.QMessageBox.about(self, popup_title, popup_message)

    def view_info(self):
        self.popup_box('Information', 'ROI Analyzer 1.1 \nDeveloped by Dongjun Choi\n\n'
                                      '- Select Folder: Ctrl+O\n'
                                      '- View Information: Ctrl+I\n'
                                      '- Remove pixel noise: Alt+C\n'
                                      '- Calculate percentile: Alt+N\n'
                                      '- Calculate reverse percentile: Alt+R')

    def run_app(self):
        self.setGeometry(300, 300, 1300, 460)
        self.setWindowTitle('ROI Analyzer v1.1')
        self.show()


if __name__ == '__main__':
    sys._excepthook = sys.excepthook

    def exception_hook(exctype, value, traceback):
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)

    sys.excepthook = exception_hook

    app = qw.QApplication(sys.argv)
    roi_analyzer = ROIAnalyzer()
    roi_analyzer.run_app()
    sys.exit(app.exec_())