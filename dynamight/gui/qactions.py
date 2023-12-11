import os
from typing import Callable, Optional

from qtpy.QtWidgets import (
    #QApplication, QLabel, QTextEdit, QRadioButton, QComboBox, QTabBar, QMainWindow,
    #QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QTabWidget,
    QAction, QMenuBar,
    #QStatusBar,
    QMenu,
    # QTreeView, QPushButton, QDockWidget, QLineEdit, QDoubleSpinBox
)
#from qtpy import QtGui
#import qtpy.QtCore as QtCore


def build_actions_dict(parent,
                       actions_data_dict: dict[str, tuple[str, str, str,
                                                          str, str, Callable, Optional[bool]],
                                               ],
                       ) -> dict[str, QAction]:
    actions_dict = {}
    for key, data in actions_data_dict.items():
        (txt, icon, shortcut, tip, func, checked) = data
        #ico = None
        #action = QAction(ico, txt, parent, checkable=False)
        action = QAction(txt, parent)
        if icon and os.path.exists(icon):
            new_icon = QIcon(':file-new.svg')
            action.setIcon(new_icon)
        if shortcut:
            action.setShortcut(shortcut)
        if tip:
            action.setStatusTip(tip)
        if func:
            action.triggered.connect(func)
        if checked is not None:
            action.setCheckable(True)
            action.setChecked(checked)
        actions_dict[key] = action

    #actions_dict['toolbar'] = parent.toolbar.toggleViewAction()
    actions_dict['res_dock'] = parent.res_dock_widget.toggleViewAction()
    actions_dict['log_dock'] = parent.log_dock_widget.toggleViewAction()
    actions_dict['analyze_dock'] = parent.analyze_dock_widget.toggleViewAction()

    #actions_dict['toolbar'].setStatusTip('Show/Hide application toolbar')
    actions_dict['res_dock'].setStatusTip('Show/Hide Results dock')
    actions_dict['log_dock'].setStatusTip('Show/Hide Log dock')
    actions_dict['analyze_dock'].setStatusTip('Show/Hide Analyze dock')
    return actions_dict

def build_menu_bar(parent,
                   menu_bar: QMenuBar,
                   menu_bar_dict: dict[str, list[str]],
                   actions_dict: dict[str, QAction]) -> None:
    """
    menu_bar_dict = {
        '&File': ['open', 'save', 'save_as', '',
                  ('&Base', ['a',  'b', 'c']),
                  'exit'],
        '&Help': ['about'],
    }
    """
    for menu_name, action_names in menu_bar_dict.items():
        menu = menu_bar.addMenu(menu_name)
        add_actions(parent, menu, action_names, actions_dict)

def add_actions(parent,
                menu: QMenu,
                action_names,
                actions_dict: dict[str, QAction]) -> None:
    for action_name in action_names:
        assert action_name != '-', action_name
        if action_name == '':
            menu.addSeparator()
            continue
        if isinstance(action_name, str):
            try:
                action = actions_dict[action_name]
            except KeyError:
                action = QAction(action_name, parent)
                action.setStatusTip('Not added to actions_dict')
                action.setEnabled(False)
            menu.addAction(action)
        elif isinstance(action_name, tuple):
            assert len(action_name) == 2, action_name
            base, action_names2 = action_name
            menu2 = menu.addMenu(base)
            add_actions(parent, menu2, action_names2, actions_dict)
        else:  # pragma: no cover
            raise NotImplementedError(action_name)
