import sys
from typing import Callable

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

from qtpy.QtWidgets import (
    QApplication, QLabel, QTextEdit, QRadioButton, QComboBox, QTabBar, QMainWindow,
    QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QTabWidget, QAction, QMenuBar,
    QStatusBar, QMenu, QTreeView, QPushButton, QDockWidget,
)
from qtpy import QtGui
import qtpy.QtCore as QtCore
from cpylog import SimpleLogger
from cpylog.html_utils import str_to_html
from pyNastran.gui.menus.menus import (
    Sidebar,
    ApplicationLogWidget,
    #PythonConsoleWidget,
)

class ResultsWindow(QWidget):
    def __init__(self, parent, name: str, data: list, *args, **kwargs):
        super().__init__()
        assert isinstance(name, str), name
        assert isinstance(data, list), data
        self.parent = parent

        self.model = QtGui.QStandardItemModel()
        self.model.setHorizontalHeaderLabels([self.tr(name)])
        is_single = self.addItems(self.model, data)

        self.treeView = QTreeView(self)
        self.treeView.setModel(self.model)
        #self.treeView.set_single(is_single)

        vbox = QVBoxLayout()
        vbox.addWidget(self.treeView)
        self.setLayout(vbox)

    def addItems(self, parent, elements, level=0, count_check=False):
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
                parent.appendRow(item)

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

class Sidebar2(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data = [
            [u'Geometry', None, [
                (u'NodeID', 0, []),
                (u'ElementID', 1, []),
                (u'PropertyID', 2, []),
                (u'MaterialID', 3, []),
                (u'E', 4, []),
                (u'Element Checks', None, [
                    (u'ElementDim', 5, []),
                    (u'Min Edge Length', 6, []),
                    (u'Min Interior Angle', 7, []),
                    (u'Max Interior Angle', 8, [])],
                ),],
            ],
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

        self.setLayout(vbox)

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        """
        https://stackoverflow.com/questions/33806598/dynamic-plotting-with-matplotlib-and-pyqt-freezing-window
        https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html
        """
        super().__init__()

        self.html_logging = True
        self.execute_python = False

        self.settings = QtCore.QSettings()
        self.settings.show_debug = True
        self.settings.show_info = True
        self.settings.show_command = True
        self.settings.show_warning = True
        self.settings.show_error = True
        self.performance_mode = False

        # ----------------------------------------------------------------------------------
        tabs = QTabWidget(self)
        self.tab1 = QWidget(self)
        self.tab2 = QWidget(self)
        self.tab3 = QWidget(self)

        tabs.addTab(self.tab1, 'Time Domain')
        tabs.addTab(self.tab2, 'Freq Domain')
        #tabs.addTab(self.tab3, 'Tab 3')

        #window = QWidget()
        self.setWindowTitle("PyQt App")

        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)

        self.res_widget = Sidebar2(self)

        self.res_dock_widget = QDockWidget('Results', self)
        self.res_dock_widget.setObjectName('results_obj')
        self.res_dock_widget.setWidget(self.res_widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.res_dock_widget)

        self.create_log_python_docks()
        self.create_menu_actions()

        x = y = 300
        width = 500
        height = 500
        self.setGeometry(x, y, width, height)
        #helloMsg = QLabel("<h1>Hello, World!</h1>", parent=window)
        #window =
        #layout = QHBoxLayout(self)
        #layout.addWidget(tabs)
        #self.setLayout(layout)
        #self.setFixedSize(tabs.sizeHint())

        #helloMsg.move(60, 15)
        self.create_static_canvas(self.tab1)
        self.setCentralWidget(tabs)

        self.log_info('cats')
        self.show()

    def create_menu_actions(self):
        menu_bar = QMenuBar()

        recent_files = []

        menu_window = []
        if self.html_logging:
            #self.actions['log_dock_widget'] = self.log_dock_widget.toggleViewAction()
            #self.actions['log_dock_widget'].setStatusTip("Show/Hide application log")
            #menu_view += ['', 'show_info', 'show_debug', 'show_command', 'show_warning', 'show_error']
            menu_window += ['res_dock', 'log_dock']

        #self.actions['toolbar'] = self.toolbar.toggleViewAction()
        #self.actions['reswidget'] = self.res_dock.toggleViewAction()
        #self.actions['log_dock'] = self.res_dock.toggleViewAction()

        #self.actions['toolbar'].setStatusTip('Show/Hide application toolbar')
        #self.actions['reswidget'].setStatusTip('Show/Hide results selection')
        #self.actions['log_dock'].setStatusTip('Show/Hide log_dock')
        # ---------------------------------------------------------------------------------
        menu_bar_dict = {
            '&File': ['open', 'save', 'save_as', '',
                      'load_time_csv',
                      ('&Recent Files', recent_files),
                      ('&Base', ['a',  'b', 'c']),
                      'exit'],
        }
        if len(menu_window):
            menu_bar_dict['&Window'] = menu_window
        menu_bar_dict['&Help'] = ['about']

        actions_data_dict = {
            # (name, txt, icon, shortcut, tip, func, checkable)
            'load_time_csv': ('Load Time CSV', '', '', '', 'loads a time domain csv', self.on_load_time_csv, False),
            'open': ('Open', '', '', 'Ctrl+O', 'opens the thingy', None, False),
            'save': ('Save', '', '', 'Ctrl+S', 'saves the thingy', None, False),
            #'save': ('Save', '', 'Ctrl+S', 'saves the thingy', None, False),
            'exit': ('Exit', '', '', 'Ctrl+Q', 'exits the thingy', None, False),

            'about': ('About', '', '', '', 'about the program', None, False),
            'log_dock' : ('Lock Dock', '', '', '', 'Show/Hide log_dock', self.log_dock_widget.toggleViewAction, True),
            'res_dock' : ('Results Dock', '', '', '', 'Show/Hide results dock', self.res_dock_widget.toggleViewAction, True),
        }
        actions_dict = build_actions_dict(self, actions_data_dict)


        build_menu_bar(self, menu_bar, menu_bar_dict, actions_dict)
        self.setMenuBar(menu_bar)

    def on_load_time_csv(self):
        pass

    #------------------------------------------------------------------------
    # plotting
    def create_static_canvas(self, widget: QWidget) -> None:
        layout = QVBoxLayout(self)
        fig = Figure(figsize=(5, 3))
        static_canvas = FigureCanvas(fig)
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        layout.addWidget(NavigationToolbar(static_canvas, self))
        layout.addWidget(static_canvas)

        self._static_ax = static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._static_ax.plot(t, np.tan(t), '.-')
        self._static_ax.grid(True)

        widget.setLayout(layout)
    #------------------------------------------------------------------------
    # Logging
    def create_log_python_docks(self):
        """
        Creates the
         - HTML Log dock
         - Python Console dock
        """
        #=========== Logging widget ===================
        self.start_logging()
        if self.html_logging is True:
            self.log_dock_widget = ApplicationLogWidget(self)
            self.log_widget = self.log_dock_widget.log_widget
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.log_dock_widget)
        else:
            self.log_widget = self.log

        if self.execute_python:
            self.python_dock_widget = PythonConsoleWidget(self)
            self.python_dock_widget.setObjectName('python_console')
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.python_dock_widget)

    def start_logging(self):
        #if self.log is not None:
            #return
        if self.html_logging:
            log = SimpleLogger(
                level='debug', encoding='utf-8',
                log_func=lambda w, x, y, z: self._logg_msg(w, x, y, z))
            # logging needs synchronizing, so the messages from different
            # threads would not be interleave
            self.log_mutex = QtCore.QReadWriteLock()
        else:
            log = SimpleLogger(
                level='debug', encoding='utf-8',
                #log_func=lambda x, y: print(x, y)  # no colorama
            )
        self.log = log

    def _logg_msg(self, log_type: str, filename: str, lineno: int, msg: str) -> None:
        """
        Add message to log widget trying to choose right color for it.

        Parameters
        ----------
        log_type : str
            {DEBUG, INFO, ERROR, COMMAND, WARNING} or prepend 'GUI '
        filename : str
            the active file
        lineno : int
            line number
        msg : str
            message to be displayed
        """
        if not self.html_logging:
            # standard logger
            name = '%-8s' % (log_type + ':')
            filename_n = '%s:%s' % (filename, lineno)
            msg2 = ' %-28s %s\n' % (filename_n, msg)
            print(name, msg2)
            return

        if 'DEBUG' in log_type and not self.settings.show_debug:
            return
        elif 'INFO' in log_type and not self.settings.show_info:
            return
        elif 'COMMAND' in log_type and not self.settings.show_command:
            return
        elif 'WARNING' in log_type and not self.settings.show_warning:
            return
        elif 'ERROR' in log_type and not self.settings.show_error:
            return

        if log_type in ['GUI ERROR', 'GUI COMMAND', 'GUI DEBUG', 'GUI INFO', 'GUI WARNING']:
            log_type = log_type[4:] # drop the GUI

        html_msg = str_to_html(log_type, filename, lineno, msg)

        if self.performance_mode or self.log_widget is None:
            self._log_messages.append(html_msg)
        else:
            self._log_msg(html_msg)

    def _log_msg(self, msg: str) -> None:
        """prints an HTML log message"""
        self.log_mutex.lockForWrite()
        text_cursor = self.log_widget.textCursor()
        end = text_cursor.End
        text_cursor.movePosition(end)
        text_cursor.insertHtml(msg)
        self.log_widget.ensureCursorVisible() # new message will be visible
        self.log_mutex.unlock()

    def log_info(self, msg: str) -> None:
        """ Helper function: log a message msg with a 'INFO:' prefix """
        if msg is None:
            msg = 'msg is None; must be a string'
            return self.log.simple_msg(msg, 'GUI ERROR')
        return self.log.simple_msg(msg, 'GUI INFO')

    def log_debug(self, msg: str) -> None:
        """ Helper function: log a message msg with a 'DEBUG:' prefix """
        if msg is None:
            msg = 'msg is None; must be a string'
            return self.log.simple_msg(msg, 'GUI ERROR')
        return self.log.simple_msg(msg, 'GUI DEBUG')

    def log_command(self, msg: str) -> None:
        """ Helper function: log a message msg with a 'COMMAND:' prefix """
        if msg is None:
            msg = 'msg is None; must be a string'
            return self.log.simple_msg(msg, 'GUI ERROR')
        return self.log.simple_msg(msg, 'GUI COMMAND')

    def log_error(self, msg: str) -> None:
        """ Helper function: log a message msg with a 'GUI ERROR:' prefix """
        if msg is None:
            msg = 'msg is None; must be a string'
            return self.log.simple_msg(msg, 'GUI ERROR')
        return self.log.simple_msg(msg, 'GUI ERROR')

    def log_warning(self, msg: str) -> None:
        """ Helper function: log a message msg with a 'WARNING:' prefix """
        if msg is None:
            msg = 'msg is None; must be a string'
            return self.log.simple_msg(msg, 'GUI ERROR')
        return self.log.simple_msg(msg, 'GUI WARNING')


def build_actions_dict(parent,
                       actions_data_dict: dict[str, tuple[str, str, str,
                                                          str, str, Callable, bool]],
                       ) -> dict[str, QAction]:
    actions_dict = {}
    for key, data in actions_data_dict.items():
        (name, txt, icon, shortcut, tip, func, checkable) = data
        #ico = None
        txt = name
        #action = QAction(ico, txt, parent, checkable=False)
        action = QAction(txt, parent)
        if icon:
            new_icon = QIcon(':file-new.svg')
            action.setIcon(new_icon)
        if shortcut:
            action.setShortcut(shortcut)
        if tip:
            action.setStatusTip(tip)
        if func:
            action.triggered.connect(func)
        if checkable:
            action.setCheckable(checkable)
        actions_dict[key] = action
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

if __name__ == '__main__':
    # 2. Create an instance of QApplication
    app = QApplication([])

    Window = MainWindow()

    # 4. Show your application's GUI
    #window.show()

    # 5. Run your application's event loop
    sys.exit(app.exec())
