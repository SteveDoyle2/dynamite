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


class EquationWidget(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__()

        self.parent = parent

        grid = QGridLayout(parent)
        equation_label = QLabel('Equation:')

        #self.text_edit = QTextEdit('-0.1 0.5; 4. 5.;')
        self.text_edit = QLineEdit('0.6 4.; 5 6;')  #  inclusive
        grid.addWidget(equation_label, 0, 0)
        grid.addWidget(self.text_edit, 0, 1)

        self.apply_button = QPushButton('Apply')

        vbox = QVBoxLayout(parent)
        vbox.addLayout(grid)
        vbox.addWidget(self.apply_button)
        self.setLayout(vbox)
        self.apply_button.clicked.connect(self.on_apply)

    def on_apply(self):
        equation = self.text_edit.text()
        self.parent.log_info(f'equation = {str(equation)}')
        self.parent.on_equation(equation)
