#import sys
#import os
#from typing import Callable

from qtpy.QtWidgets import (
    #QApplication, QLabel, QTextEdit, QRadioButton, QComboBox, QTabBar, QMainWindow,
    QWidget, QVBoxLayout, # QGridLayout, QHBoxLayout, QTabWidget, QAction, QMenuBar,
    #QStatusBar, QMenu,
    QTreeView, QPushButton,
    #QDockWidget,
)
from qtpy import QtGui
#import qtpy.QtCore as QtCore
#from pyNastran.gui.menus.menus import (
    #Sidebar,
    #ApplicationLogWidget,
    #PythonConsoleWidget,
#)

class Sidebar2(QWidget):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent

        data = [
            #[u'Geometry', None, [
                #(u'NodeID', 0, []),
                #(u'ElementID', 1, []),
                #(u'PropertyID', 2, []),
                #(u'MaterialID', 3, []),
                #(u'E', 4, []),
                #(u'Element Checks', None, [
                    #(u'ElementDim', 5, []),
                    #(u'Min Edge Length', 6, []),
                    #(u'Min Interior Angle', 7, []),
                    #(u'Max Interior Angle', 8, [])],
                #),],
            #],
        ]


        self.result_method_window = ResultsWindow(self, 'Files', data)
        #self.result_method_window.setVisible(False)
        #else:
            #self.result_method_window = None

        #self.show_pulldown = False
        #if self.show_pulldown:
            ##combo_options = ['a1', 'a2', 'a3']
            #self.pulldown = QComboBox()
            #self.pulldown.addItems(choices)
            #self.pulldown.activated[str].connect(self.on_pulldown)

        self.apply_button = QPushButton('Apply', self)

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.result_method_window)
        vbox.addWidget(self.apply_button)

        self.setup_connections()
        self.setLayout(vbox)

    def setup_connections(self) -> None:
        self.apply_button.clicked.connect(self.on_apply)

    def on_apply(self) -> None:
        window = self.result_method_window
        tree_view = window.tree_view
        #model = window.model
        indexs = tree_view.selectedIndexes()

        index_to_map_dict = {}
        for i, index0 in enumerate(indexs):
            data = index0.data()
            datas = [data]
            rows = [index0.row()]

            index = index0.parent()
            while 1:
                data = index.data()
                if data is None:
                    break
                row = index.row()
                datas.append(data)
                rows.append(row)
                index = index.parent()

            datas2 = list(reversed(datas))
            rows2 = list(reversed(rows))
            index_to_map_dict[i] = (datas2, rows2)
        asdf

class ResultsWindow(QWidget):
    def __init__(self, parent: Sidebar2, name: str, data: list, *args, **kwargs):
        super().__init__()
        assert isinstance(name, str), name
        assert isinstance(data, list), data
        self.parent = parent

        self.model = QtGui.QStandardItemModel()
        self.model.setHorizontalHeaderLabels([self.tr(name)])
        is_single = self.addItems(self.model, data)

        self.tree_view = QTreeView(self)
        self.tree_view.setModel(self.model)

        self.tree_view.setSelectionMode(QTreeView.MultiSelection)
        #self.tree_view.setSelectionMode(self.tree_view.selectionMode().MultiSelection)
        #self.treeView.set_single(is_single)
        self.tree_view.expandAll()

        vbox = QVBoxLayout()
        vbox.addWidget(self.tree_view)
        self.setLayout(vbox)

    def set_form(self, form: list):
        self.tree_view.clearSelection()
        self.addItems(self.model, form, level=0, count_check=False)
        self.tree_view.expandAll()

    def addItems(self, model, elements, level=0, count_check=False):
        nelements = len(elements)
        redo = False
        #print(elements[0])
        try:
            #if len(elements):
                #assert len(elements[0]) == 3, 'len=%s elements[0]=%s\nelements=\n%s\n' % (
                    #len(elements[0]), elements[0], elements)
            for element in elements:
                #if isinstance(element, str):
                    #print('elements = %r' % str(elements))

                #print('element = %r' % str(element))
                if not len(element) == 3:
                    print('element = %r' % str(element))
                try:
                    text, i, children = element
                except ValueError:
                    #  [
                    #     ('Point Data', None, [
                    #         ('NodeID', 0, []),
                    #         ('Displacement T_XYZ_subcase=1', 1, []),
                    #         ('Displacement R_XYZ_subcase=1', 2, []),
                    #     ])
                    #  ]
                    #
                    # should be:
                    #   ('Point Data', None, [
                    #       ('NodeID', 0, []),
                    #       ('Displacement T_XYZ_subcase=1', 1, []),
                    #       ('Displacement R_XYZ_subcase=1', 2, []),
                    #   ])
                    print('failed element = ', element)
                    raise
                nchildren = len(children)
                #print('text=%r' % text)
                item = QtGui.QStandardItem(text)
                model.appendRow(item)

                # TODO: count_check and ???
                if nelements == 1 and nchildren == 0 and level == 0:
                    #self.result_data_window.setEnabled(False)
                    item.setEnabled(False)
                    #print(dir(self.treeView))
                    #self.treeView.setCurrentItem(self, 0)
                    #item.mousePressEvent(None)
                    redo = True
                #else:
                    #pass
                    #print('item=%s count_check=%s nelements=%s nchildren=%s' % (
                        #text, count_check, nelements, nchildren))
                if children:
                    assert isinstance(children, list), children
                    self.addItems(item, children, level + 1, count_check=count_check)
                    #print('*children = %s' % children)
            is_single = redo
            return is_single
        except ValueError:
            print()
            print(f'elements = {elements}')
            print(f'element = {element}')
            print(f'len(element) = {len(element)}')
            print(f'len(elements)={len(elements)}')
            for elem in elements:
                print('  e = %s' % str(elem))
            raise
        #if redo:
        #    data = [
        #        ('A', []),
        #        ('B', []),
        #    ]
        #    self.update_data(data)
