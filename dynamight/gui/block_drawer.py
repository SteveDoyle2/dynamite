import sys
from typing import Any

# kills the program when you hit Cntl+C from the command line
# doesn't save the current state as presumably there's been an error
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import numpy as np
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (
    QWidget, QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsItem, QMainWindow, QTreeView, QListView, QVBoxLayout, QHBoxLayout, QPushButton)
from PyQt5.QtCore import Qt, QRect, QRectF


class MyDrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Block Drawing")
        self.setGeometry(100, 100, 400, 300)

    def paintEvent(self, event):
        painter = QPainter(self)

        # Set pen for outlines
        painter.setPen(QPen(Qt.black, 2, Qt.SolidLine))

        # Set brush for fill
        painter.setBrush(QBrush(QColor(255, 255, 0), Qt.SolidPattern))  # Yellow solid fill

        # Draw a single rectangle
        painter.drawRect(50, 50, 100, 80)

        # Change brush and draw another rectangle
        painter.setBrush(QBrush(QColor(0, 100, 200), Qt.Dense2Pattern))  # Blue patterned fill
        painter.drawRect(200, 150, 120, 60)

        # Draw multiple rectangles using QRect objects
        rects_to_draw = [
            QRect(30, 200, 70, 40),
            QRect(120, 220, 90, 50)
        ]
        painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        painter.drawRects(rects_to_draw)
        painter.end()


# 1. Create a custom block item
class DraggableBlock(QGraphicsRectItem):
    def __init__(self, block_id: int, row_id: int,
                 x: int, y: int, width: int, height: int,
                 color='pink', text: str=''):
        self.block_id = block_id
        self.row_id = row_id
        self.text = text

        super().__init__(x, y, width, height)

        # Set styling properties
        if isinstance(color, str):
            qcolor = QColor(color)
        elif isinstance(color, tuple):
            assert isinstance(color[0], int), color
            qcolor = QColor.fromRgb(*color)
        else:
            raise RuntimeError(f'Unsupported color type: {type(color)}; color={color}')

        qbrush = QBrush(qcolor)
        self.setBrush(qbrush)
        # Enable the item to be selected and movable
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        # self.text().setText(text)


    # 2. Add event handling for mouse clicks
    def mousePressEvent(self, event):
        print("Block pressed!")
        # Call the parent's method to enable dragging
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        print(f"Block released at new position: ({self.x()}, {self.y()})")
        super().mouseReleaseEvent(event)


def main2():
    app = QApplication([])
    widget = MyDrawingWidget()
    widget.show()
    app.exec_()


class Time:
    def __init__(self, idi: int):
        self.id = idi
        self.color = (255, 0, 0)
        self.dt = 0.1
        self.num = 101
        self.ttotal = 1.0

    def run(self):
        return np.linspace(0., self.dt, self.ttotal + self.dt, num=101)

class SineWave:
    def __init__(self, idi: int):
        self.id = idi
        self.time = np.array([])
        self.color = (0, 0, 255)

    def set_time(self, time: np.ndarray):
        self.time = time

    def run(self):
        return np.sin(self.time)


block_name_to_block = {
    'time': Time,
    'sine': SineWave,
}
mapper = {
    'Dynamics': {
        'Time': 'time',
        'Sine Wave': 'sine',
        # 'Square Wave': ('wave2', SquareWave),
    },
    # 'Aero': {
    #     'KEAS': 'KEASi',
    # },
}
class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.block_id = 0
        self.setWindowTitle("PyQt Block Drawing")

        self.model, self.tree_view, self.block_id_to_block = get_model('Blocks Diagram Builder', mapper)

        # Create the QGraphicsScene
        scene = QGraphicsScene(parent=self)
        scene.setSceneRect(0, 0, 800, 600)  # Define the scene size
        self.scene = scene

        # Create the QGraphicsView and connect it to the scene
        view = QGraphicsView(scene)
        view.setWindowTitle("Interactive Draggable Block with PyQt")
        view.resize(800, 600)

        self.view = view
        # view.show()
        # widget = QWidget(self)
        # widget.setLayout(view)
        # self.setCentralWidget(widget)
        # self.setCentralWidget(self.view)

        self.apply_button = QPushButton('Apply')
        self.apply_button.clicked.connect(self.on_apply)

        vbox = QVBoxLayout()
        vbox.addWidget(self.tree_view)
        vbox.addWidget(self.apply_button)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox)
        hbox.addWidget(view)
        # self.setLayout(vbox)

        widget = QWidget()
        widget.setLayout(hbox)
        self.setCentralWidget(widget)
        # self.setLayout(hbox)
        self.show()

    def on_apply(self):
        current = self.tree_view.currentIndex()
        print(f'current = {current}; row={current.row()}')
        indices = self.tree_view.selectedIndexes()
        # print(indices)
        index = indices[0]
        # print(index)
        row_id = index.row()
        # index.
        print(f'row_id = {row_id}')
        print(f'block_id_to_block = {self.block_id_to_block}')
        block_cls = self.block_id_to_block[row_id]
        print(f'block_cls = {block_cls}')
        blocki = block_cls(self.block_id)
        # self.model.get_state()

        # Create and add the DraggableBlock item to the scene
        color = blocki.color
        print(f'color = {color}')
        block = DraggableBlock(self.block_id, row_id, 100, 100, 80, 50,
                               color=color)
        self.scene.addItem(block)

        self.block_id += 1


def main():
    app = QApplication(sys.argv)
    mw = MyMainWindow()
    sys.exit(app.exec_())


class TreeViewExample(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt QTreeView Example")
        self.setGeometry(100, 100, 400, 300)

        # mapper = {
        #     'Dynamics': {
        #         'Sine Wave': 'wave1',
        #         'Square Wave': 'wave2', },
        #     'Aero': {
        #         'KEAS': 'KEASi',
        #     },
        # }
        self.model, self.tree_view = get_model('Blocks Diagram Builder', mapper)
        # self.model = model
        # self.tree_view = tree_view
        # Set the central widget
        self.setCentralWidget(self.tree_view)

from functools import partial
def fill_qstandarditem_from_subdict(parent_item: QStandardItem, mapper: dict,
                                    block_id_to_block_name: dict[int, str],
                                    id: int) -> int:
    for name, subdict in mapper.items():
        assert isinstance(name, str), (name, type(name))
        parent_item2 = QStandardItem(name)
        parent_item.appendRow(parent_item2)
        if isinstance(subdict, dict):
            fill_qstandarditem_from_subdict(parent_item2, subdict, block_id_to_block_name, id=id)
        elif isinstance(subdict, str):
            # (right_click_msg, callback_func, validate) = right_click_action
            # action = right_click_menu.addAction(right_click_msg)
            # true_false_callback = true_callback if validate else false_callback
            # trigger_func = partial(true_false_callback, callback_func)
            # action.triggered.connect(trigger_func)

            print(name, subdict)
            block_id_to_block_name[id] = subdict
            # if 0:
            #     trigger_func = partial(on_custom_context_menu, id)
            #     right_click_msg = 'MyRightClick'
            #     action = parent_item2.addAction(right_click_msg)
            #     action.triggered.connect(trigger_func)
            #     parent_item2.addAction(action)
            # parent_item2.customContextMenuRequested.connect(func)
            id += 1
            # child = QStandardItem(subdict)
            # parent_item2.appendRow(child)
        # elif isinstance(subdict, tuple):
        #     pass
        else:
            raise NotImplementedError((subdict, type(subdict)))
        # id += 1
    print(f'block_id_to_block_name = {block_id_to_block_name}')
    return id

def on_custom_context_menu(id: int):
    print(f'idi = {id}')

def get_model(title: str, mapper: dict[str, Any]) -> tuple[QStandardItemModel, QTreeView]:
    # Create the model
    model = QStandardItemModel()
    model.setHorizontalHeaderLabels([title])

    block_id_to_block_name = {}
    idi = 0
    for name, subdict in sorted(mapper.items()):
        assert isinstance(name, str), (name, type(name))
        parent_item = QStandardItem(name)
        if isinstance(subdict, dict):
            fill_qstandarditem_from_subdict(parent_item, subdict, block_id_to_block_name, id=idi)
        else:
            raise NotImplementedError((subdict, type(subdict)))
        model.appendRow(parent_item)
    print(f'model_block_id_to_block_name = {block_id_to_block_name}')

    block_id_to_block = {}
    for idi, name in block_id_to_block_name.items():
        block = block_name_to_block[name]
        print(idi, name, block)
        block_id_to_block[idi] = block
    print(f'block_id_to_block = {block_id_to_block}')

    # Create the QTreeView
    tree_view = QTreeView()
    tree_view.setModel(model)

    # Expand all items by default
    tree_view.expandAll()

    return model, tree_view, block_id_to_block


def main_treeview():
    app = QApplication(sys.argv)
    window = TreeViewExample()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    # main_treeview()
    main()
