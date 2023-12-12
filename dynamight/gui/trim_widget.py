from __future__ import annotations
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QGridLayout, # QHBoxLayout,
    QPushButton, QLineEdit,
)
#from qtpy import QtGui
#import qtpy.QtCore as QtCore
if TYPE_CHECKING:
    from dynamight.gui.gui import MainWindow


class TrimWidget(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__()

        self.parent = parent

        grid = QGridLayout(parent)
        trim_label = QLabel('Trim:')

        #self.text_edit = QTextEdit('-0.1 0.5; 4. 5.;')
        self.text_edit = QLineEdit('0.6 4.; 5 6;')  #  inclusive
        grid.addWidget(trim_label, 0, 0)
        grid.addWidget(self.text_edit, 0, 1)

        self.apply_button = QPushButton('Apply')

        vbox = QVBoxLayout(parent)
        vbox.addLayout(grid)
        vbox.addWidget(self.apply_button)
        self.setLayout(vbox)
        self.apply_button.clicked.connect(self.on_apply)

    def on_apply(self):
        text = self.text_edit.text()
        sline = text.split(';')
        trims = []
        for slinei in sline:
            slinei = slinei.strip()
            if len(slinei) == 0:
                continue
            mini, maxi = slinei.split(' ')
            min_float = float(mini)
            max_float = float(maxi)
            trim = [min_float, max_float]
            trims.append(trim)
        self.parent.log_info(f'trims = {str(trims)}')
        self.parent.on_trim_data(trims)
