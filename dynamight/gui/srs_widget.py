from __future__ import annotations
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QLabel, # QTextEdit, QRadioButton, QComboBox, QTabBar,
    QWidget, QVBoxLayout, QGridLayout, # QHBoxLayout, QTabWidget, QAction, QMenuBar,
    #QStatusBar, QMenu, QTreeView,
    QPushButton, QCheckBox, # QDockWidget, QLineEdit, QDoubleSpinBox
)
from qtpy import QtGui
import qtpy.QtCore as QtCore
from pyNastran.gui.utils.qt.pydialog import QFloatEdit
if TYPE_CHECKING:
    from dynamight.gui.gui import MainWindow

class SrsWidget(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__()

        self.parent = parent

        grid = QGridLayout(parent)

        fmin_label = QLabel('Freq Min (Hz):')
        self.fmin_edit = QFloatEdit('10.0')

        fmax_label = QLabel('Freq Max (Hz):')
        self.fmax_edit = QFloatEdit('2000.')

        q_label = QLabel('Q')
        self.q_edit = QFloatEdit('10.0')

        irow = 0
        grid.addWidget(fmin_label, irow, 0)
        grid.addWidget(self.fmin_edit, irow, 1)
        irow += 1

        grid.addWidget(fmax_label, irow, 0)
        grid.addWidget(self.fmax_edit, irow, 1)
        irow += 1

        grid.addWidget(q_label, irow, 0)
        grid.addWidget(self.q_edit, irow, 1)
        irow += 1

        self.apply_button = QPushButton('Apply')

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
        fmin = float(self.fmin_edit.text())
        fmax = float(self.fmax_edit.text())
        Q = float(self.q_edit.text())
        calculate_log_mean = self.calculate_log_mean.isChecked()
        self.parent.log_info(f'Q={Q}')
        self.parent.on_analyze_srs(fmin=fmin, fmax=fmax, Q=Q,
                                   calculate_log_mean=calculate_log_mean)
