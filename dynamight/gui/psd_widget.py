from __future__ import annotations
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QLabel, QComboBox,
    #QTextEdit, QRadioButton,
    QWidget, QVBoxLayout, QGridLayout,
    #QStatusBar, QMenu, QTreeView,
    QPushButton, QCheckBox,
    #QDockWidget, QLineEdit,
    QDoubleSpinBox,
)
#from qtpy import QtGui
#import qtpy.QtCore as QtCore
from pyNastran.gui.utils.qt.pydialog import QFloatEdit# , PyDialog, set_combo_box_text
from dynamight.core.time import TimeSeries
from dynamight.core.psd import PowerSpectralDensity
if TYPE_CHECKING:
    from dynamight.gui.gui import MainWindow

#PSD_WINDOWS = ['barthann', 'bartlett', 'blackman', 'blackmanharris', 'bohman', 'boxcar', 'chebwin', 'cosine', 'dpss', 'exponential', 'flattop', 'gaussian', 'general_cosine', 'general_gaussian', 'general_hamming', 'hamming', 'hann', 'kaiser', 'kaiser_bessel_derived', 'lanczos', 'nuttall', 'parzen', 'taylor', 'triang', 'tukey',]
PSD_WINDOWS = ['hann', 'hamming', 'boxcar', 'bartlett', 'blackman', 'cosine', 'flattop', 'kaiser',]

class PsdWidget(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__()

        self.parent = parent

        grid = QGridLayout(parent)
        window_size_label = QLabel('window_size (sec)')

        #self.text_edit = QLineEdit('-0.1 0.5; 4. 5.;')
        self.window_size_edit = QFloatEdit('1.0')

        window_label = QLabel('Window:')
        self.window_pulldown = QComboBox(parent)
        self.window_pulldown.addItems(PSD_WINDOWS)

        overlap_label = QLabel('Overlap:')
        self.overlap_spinner = QDoubleSpinBox(parent)
        self.overlap_spinner.setValue(0.5)
        self.overlap_spinner.setDecimals(3)
        self.overlap_spinner.setMinimum(0.0)
        self.overlap_spinner.setMaximum(1.0)

        irow = 0
        grid.addWidget(window_size_label, irow, 0)
        grid.addWidget(self.window_size_edit, irow, 1)
        irow += 1

        grid.addWidget(window_label, irow, 0)
        grid.addWidget(self.window_pulldown, irow, 1)
        irow += 1

        grid.addWidget(overlap_label, irow, 0)
        grid.addWidget(self.overlap_spinner, irow, 1)
        irow += 1

        #grid.addWidget(log_mean, irow, 0)
        #grid.addWidget(self.calculate_log_mean_check, irow, 1)
        irow += 1

        self.calculate_log_mean = QCheckBox(text='Calculate Log Mean')
        self.calculate_log_mean.setChecked(True)
        self.apply_button = QPushButton('Apply')

        vbox = QVBoxLayout(parent)
        vbox.addLayout(grid)
        vbox.addWidget(self.calculate_log_mean)
        vbox.addWidget(self.apply_button)

        self.setLayout(vbox)
        self.apply_button.clicked.connect(self.on_apply)

    def on_apply(self):
        window_size = float(self.window_size_edit.text())
        window = self.window_pulldown.currentText()
        overlap = self.overlap_spinner.value()
        calculate_log_mean = self.calculate_log_mean.isChecked()

        #sline = text.split(';')
        #trims = []  # inclusive
        #for slinei in sline:
            #slinei = slinei.strip()
            #if len(slinei) == 0:
                #continue
            #mini, maxi = slinei.split(' ')
            #min_float = float(mini)
            #max_float = float(maxi)
            #trim = [min_float, max_float]
            #trims.append(trim)
        self.parent.log_info(f'window_size={window_size}; window={window!r}')
        self.parent.on_analyze_psd(window, window_size, overlap)
