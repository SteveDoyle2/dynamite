"""
https://vibrationdata.wordpress.com/2014/04/02/python-signal-analysis-package-gui/
"""
import getpass
#import os
import sys
from functools import partial
#from typing import Callable, Optional

import numpy as np

import matplotlib
#print(matplotlib.__version__)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

if 1:
    import PySide6
    from PySide6.QtWidgets import (
        QApplication, QMainWindow,
        QLabel, QTextEdit, #QRadioButton,
        QComboBox, #QTabBar,
        QWidget, QVBoxLayout, QGridLayout, QHBoxLayout,
        QTabWidget, # QAction,
        QMenuBar,
        QStatusBar, # QMenu, QTreeView, QPushButton,
        QDockWidget, # QLineEdit, QDoubleSpinBox
    )
    from PySide6 import QtGui
    import PySide6.QtCore as QtCore
else:
    from qtpy.QtWidgets import (
        QApplication, QMainWindow,
        #QLabel, QTextEdit, QRadioButton, QComboBox, QTabBar,
        QWidget, QVBoxLayout, # QGridLayout, QHBoxLayout,
        QTabWidget, # QAction,
        QMenuBar,
        QStatusBar, # QMenu, QTreeView, QPushButton,
        QDockWidget, # QLineEdit, QDoubleSpinBox
    )
    from qtpy import QtGui
    import qtpy.QtCore as QtCore

from cpylog import SimpleLogger
from cpylog.html_utils import str_to_html
from pyNastran.gui.menus.menus import (
    ApplicationLogWidget,
    #PythonConsoleWidget,
)
#from pyNastran.gui.menus.about.about import AboutWindow

import dynamight
from dynamight.gui.about_window import AboutWindow
from pyNastran.gui.utils.qt.pydialog import PyDialog, QFloatEdit
from pyNastran.gui.utils.qt.dialogs import open_file_dialog
from pyNastran.gui.utils.qt.qcombobox import set_combo_box_text

from dynamight.gui.qactions import build_actions_dict, build_menu_bar
from dynamight.gui.sidebar import Sidebar2
from dynamight.gui.trim_widget import TrimWidget
from dynamight.gui.psd_widget import PsdWidget
from dynamight.gui.srs_widget import SrsWidget
from dynamight.gui.equation_widget import EquationWidget
from dynamight.core.time import TimeSeries
from dynamight.core.psd import PowerSpectralDensity
from dynamight.core.srs import ShockResponseSpectra


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
        time = np.linspace(0, 10, 501)
        response = np.tan(time)
        self.xy_data: dict[int, tuple[np.ndarray, np.ndarray]] = {
            1: (time, response),
        }
        self.trims: list[tuple[float, float]] = []
        self.psd_data: dict[int, PowerSpectralDensity] = {}
        self.srs_data: dict[int, ShockResponseSpectra] = {}

        tabs = QTabWidget(self)
        self.tab_time = QWidget(self)
        self.tab_psd = QWidget(self)
        #self.tab_vrs = QWidget(self)
        self.tab_srs = QWidget(self)
        self.tab_equation = QWidget(self)

        tabs.addTab(self.tab_equation, 'Equation')

        tabs.addTab(self.tab_time, 'Time Domain')
        tabs.addTab(self.tab_psd, 'PSD-Freq Domain')
        #tabs.addTab(self.tab_vrs, 'VRS-Freq Domain')
        tabs.addTab(self.tab_srs, 'SRS-Freq Domain')
        #self.tab_freq.setDisabled(True)
        self.tab_equation.hide()
        self.tab_psd.hide()
        self.tab_srs.hide()

        self.setWindowTitle('dynamight')

        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)

        self.res_widget = Sidebar2(self)
        self.res_dock_widget = QDockWidget('Results', self)
        self.res_dock_widget.setObjectName('results_obj')
        self.res_dock_widget.setWidget(self.res_widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.res_dock_widget)

        self.is_psd = False
        self.is_srs = False
        self.analyze_widgets = {
            'equation': EquationWidget(self),
            'trim': TrimWidget(self),
            'psd': PsdWidget(self),
            'srs': SrsWidget(self),
            #'vrs': VrsWidget(self),
        }
        self.analyze_dock_widget = QDockWidget('Analyze', self)
        self.analyze_dock_widget.hide()
        #self.analyze_dock_widget.setObjectName('results_obj')
        #self.analyze_dock_widget.setWidget(self.menu_widget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.analyze_dock_widget)

        self.create_log_python_docks()
        self.create_menu_actions()

        x = y = 300
        width = 1000
        height = 1000
        self.setGeometry(x, y, width, height)
        #helloMsg = QLabel("<h1>Hello, World!</h1>", parent=window)

        #layout = QHBoxLayout(self)
        #layout.addWidget(tabs)
        #self.setLayout(layout)
        #self.setFixedSize(tabs.sizeHint())

        #helloMsg.move(60, 15)
        self._equation_ax = self.create_static_canvas(self.tab_equation, domain='equation')
        self._time_ax = self.create_static_canvas(self.tab_time, domain='time')
        self._psd_ax = self.create_static_canvas(self.tab_psd, domain='freq')
        self._srs_ax = self.create_static_canvas(self.tab_srs, domain='freq')
        self._srs_ax.set_ylabel('SRS Acceleration (g); Q=10')
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
            menu_window += [
                'res_dock', 'log_dock', 'analyze_dock', '',
                'show_debug', 'show_info', 'show_warning', 'show_error', 'show_command', ]
        # ---------------------------------------------------------------------------------
        analyze = [
            'equation',
            'trim', 'filter', '',
            'psd', 'set_psd_limits', '',
            'vrs', '',
            'srs','set_srs_limits', '',
            'pseudo-velocity',
        ]
        menu_bar_dict = {
            '&File': ['open', 'save', 'save_as', '',
                      'load_time_csv',
                      ('&Recent Files', recent_files),
                      #('&Base', ['a',  'b', 'c']),
                      'exit'],
            '&Analyze': analyze,
        }
        if len(menu_window):
            menu_bar_dict['&Window'] = menu_window
        menu_bar_dict['&Help'] = ['about']

        checked = True
        no_check = None
        actions_data_dict = {
            # (name, icon, shortcut, tip, func, checkable)
            'load_time_csv': ('Load Time CSV', '', '', 'loads a time domain csv', self.on_load_time_csv, no_check),
            'open': ('Open...', '', 'Ctrl+O', 'opens the thingy', None, no_check),
            'save': ('Save...', '', 'Ctrl+S', 'saves the thingy', None, no_check),
            #'save_as': ('Save As...', 'Ctrl+S', 'saves the thingy', None, no_check),
            'exit': ('Exit...', '', 'Ctrl+Q', 'exits the thingy', None, no_check),

            'about'        : ('About...',     '', '', 'about the program', self.on_about, no_check),
            'equation_dock': ('Equation Dock','', '', 'Show/Hide equation_dock', None, checked),
            'log_dock'     : ('Log Dock',     '', '', 'Show/Hide log_dock', None, checked),
            'res_dock'     : ('Results Dock', '', '', 'Show/Hide results dock', None, checked),
            'analyze_dock' : ('Analyze Dock', '', '', 'Show/Hide analyze dock', None, checked),

            'show_info'   : ('Show INFO',    'show_info.png',    '', 'Show "INFO" messages', self.on_show_info, checked),
            'show_debug'  : ('Show DEBUG',   'show_debug.png',   '', 'Show "DEBUG" messages', self.on_show_debug, checked),
            'show_command': ('Show COMMAND', 'show_command.png', '', 'Show "COMMAND" messages', self.on_show_command, checked),
            'show_warning': ('Show WARNING', 'show_warning.png', '', 'Show "WARJNING" messages', self.on_show_warning, checked),
            'show_error'  : ('Show ERROR',   'show_error.png',   '', 'Show "ERROR" messages', self.on_show_error, checked),

            # analyze
            'equation' : ('Equation', '', 'Ctrl+E', 'Make an equation',                                        partial(self._show_sidebar, 'equation'), no_check),
            'trim'  : ('Trim Time', '', '', 'Trim a time series',                                              partial(self._show_sidebar, 'trim'), no_check),
            'psd'  : ('PSD',        '', '', 'Calculates a PSD (power spectral density) of the shown data',     partial(self._show_sidebar, 'psd'), no_check),
            'vrs'  : ('VRS',        '', '', 'Calculates a VRS (vibration response spectra) of the shown data', partial(self._show_sidebar, 'vrs'), no_check),
            'srs'  : ('SRS',        '', '', 'Calculates a SRS (shock response spectra) of the shown data',     partial(self._show_sidebar, 'srs'), no_check),
        }
        actions_dict = build_actions_dict(self, actions_data_dict)

        #actions_dict['show_info'].setChecked(self.settings.show_info)
        #actions_dict['show_debug'].setChecked(self.settings.show_debug)
        #actions_dict['show_command'].setChecked(self.settings.show_command)
        #actions_dict['show_warning'].setChecked(self.settings.show_warning)
        #actions_dict['show_error'].setChecked(self.settings.show_error)

        build_menu_bar(self, menu_bar, menu_bar_dict, actions_dict)
        self.setMenuBar(menu_bar)
        self.cases: dict[int, int] = {}
        self.form: list[tuple] = []

    def get_form_cases(self) -> tuple[int, list[tuple], dict[int, int]]:
        i = len(self.cases)
        return i, self.form, self.cases

    def set_form_cases(self, form: list[tuple], cases: dict[int, int]) -> None:
        self.cases = cases
        self.res_widget.result_method_window.set_form(form)

    def on_load_time_csv(self):
        """opens a file dialog"""
        if 0:
            #filetypes = 'Comma Separated Value (*.csv);;Space/Tab Separated Value (*.dat);;All files (*)'
            default_filename = ''
            file_types = 'Delimited Text (*.txt; *.dat; *.csv)'
            csv_filename, filt = open_file_dialog(self, 'Select a CSV File', default_filename, file_types)
        else:
            csv_filename = r'C:\NASA\m4\formats\git\dynamight\dynamight\gui\spike.csv'

        time_series = TimeSeries.load_from_csv_filename(csv_filename)
        i, form, cases = self.get_form_cases()

        columns = []
        for ilabel, label in enumerate(time_series.label):
            columns.append((label, i, []))
            cases[i] = (i, ilabel, label, time_series)
            i += 1
        formi = (csv_filename, None, columns)
        form.append(formi)

        self.set_form_cases(form, cases)

    @property
    def colors(self):
        ncurves = len(self.xy_data)
        return [f'C{i}' for i in range(ncurves)]

    def on_analyze_psd(self, window: str, window_size_sec: float, overlap: float):
        #colors = self.colors
        iresponse = 0
        self.psd_data = {}
        for key, (time, response) in self.xy_data.items():
            label = str(key)
            series = TimeSeries(time, response, label=label)
            res = series.to_psd_welch(sided=1, window=window, window_size_sec=window_size_sec, overlap=overlap)
            self.psd_data[iresponse] = res
        set_tab(self.tab_psd)

        self.on_plot_psd(self.psd_data)
        self.log_command(f'self.on_analyze_psd(window={window!r}, window_size_sec={window_size_sec}, overlap={overlap})')

    def on_analyze_vrs(self, Q: float):
        pass

    def on_analyze_srs(self,
                       fmin: float=10.,
                       fmax: float=2000.,
                       Q: float=10.):
        #colors = self.colors
        iresponse = 0
        self.psd_data = {}
        for key, (time, response) in self.xy_data.items():
            label = str(key)
            series = TimeSeries(time, response, label=label)
            srs = series.to_srs(fmin=fmin, fmax=fmax,
                                Q=Q,
                                #noctave: int=6,
                                calc_accel_srs=True,
                                calc_rel_disp_srs=False,
                                )
            self.srs_data[iresponse] = srs

        self.on_plot_srs(self.srs_data)
        set_tab(self.tab_srs)
        self.log_command(f'self.on_analyze_srs(fmin={fmin}, fmax={fmax}, Q={Q})')

    def on_analyze_waterfall_psd(self):
        pass
    def on_analyze_waterfall_vrs(self):
        pass
    def on_analyze_waterfall_srs(self):
        pass
    #------------------------------------------------------------------------
    # analyze setup

    def _show_sidebar(self, name: str) -> None:
        keys = list(self.analyze_widgets.keys())
        if name not in keys:
            self.log_error(f'name={name!r} is not in {keys}')
            return
        for name_widget, widget in self.analyze_widgets.items():
            if name == name_widget:
                widget.show()
                self.analyze_dock_widget.setWidget(widget)
                self.analyze_dock_widget.show()
            else:
                widget.hide()

    #------------------------------------------------------------------------
    # plotting
    def on_equation(self, equation: str) -> None:
        if len(equation) == 0:
            return
        ax = self._equation_ax
        ax.clear()
        ax.set_title('$'+ equation +'$')
        fig = ax.get_figure()
        fig.canvas.draw()
        self.log_command(f'self.on_equation({equation})')

    def on_trim_data(self, trims: list[tuple[float, float]]):
        if len(trims) == 0:
            return
        ax = self._time_ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.clear()

        #ncurves = len(self.xy_data)
        colors = self.colors
        iresponse = 0
        for key, (x, y) in self.xy_data.items():
            color = colors[iresponse]
            ax.plot(x, y, linestyle='-', color='grey')
            for (mini, maxi) in trims:
                bools = (mini < x) & (x < maxi)
                i = np.where(bools)
                ax.plot(x[i], y[i], color=color, linewidth=3)
            iresponse += 1
        ax.grid(True)

        domain = 'time'
        if domain == 'time':
            xlabel = 'Time (sec)'
            ylabel = 'Response (g)'
        elif domain == 'freq':
            xlabel = 'Frequency (Hz)'
            ylabel = 'PSD ($g^2$/Hz)'
        else:
            raise RuntimeError(domain)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        fig = ax.get_figure()
        fig.canvas.draw()
        self.log_command(f'self.on_trim_data({trims})')
        self.trims = trims

    def on_plot_psd(self, psd_data: dict[int, PowerSpectralDensity]):
        # self.trims
        ax = self._psd_ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.clear()

        #ncurves = len(self.xy_data)
        colors = self.colors
        iresponse = 0
        for key, psd_series in self.psd_data.items():
            #color = colors[iresponse]
            psd_series.plot(ax=ax)
            #ax.plot(x, y, linestyle='-', color='grey')
            #for (mini, maxi) in trims:
                #bools = (mini < x) & (x < maxi)
                #i = np.where(bools)
                #ax.plot(x[i], y[i], color=color, linewidth=3)
            iresponse += 1

        fig = ax.get_figure()
        fig.canvas.draw()
        #if self.is_psd:
            #ax.set_xlim(xlim)
            #ax.set_ylim(ylim)
        self.is_psd = True
        #self.trims = trims
        return

    def on_plot_srs(self, srs_data: dict[int, ShockResponseSpectra]):
        # self.trims
        ax = self._srs_ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.clear()

        #ncurves = len(self.xy_data)
        #colors = self.colors
        iresponse = 0
        for key, srs_series in self.srs_data.items():
            #color = colors[iresponse]
            srs_series.plot_srs_accel(ax=ax)
            #ax.plot(x, y, linestyle='-', color='grey')
            #for (mini, maxi) in trims:
                #bools = (mini < x) & (x < maxi)
                #i = np.where(bools)
                #ax.plot(x[i], y[i], color=color, linewidth=3)
            iresponse += 1

        fig = ax.get_figure()
        fig.canvas.draw()
        #if self.is_srs:
            #ax.set_xlim(xlim)
            #ax.set_ylim(ylim)
        self.is_srs = True
        #self.trims = trims
        return

    def create_static_canvas(self, widget: QWidget,
                             domain: str='time') -> plt.Axes:
        """
        domains: time, psd
        """
        layout = QVBoxLayout(self)
        widget.setLayout(layout)

        fig = Figure(figsize=(5, 3))
        static_canvas = FigureCanvas(fig)

        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.

        if 0:
            nav_bar = NavigationToolbar(static_canvas)
        else:
            nav_bar = NavigationToolbar(static_canvas, self)
        layout.addWidget(nav_bar)
        layout.addWidget(static_canvas)

        ax = static_canvas.figure.subplots()

        if domain == 'time':
            iresponse = 1
            time, response = self.xy_data[iresponse]
            ax.plot(time, response, marker='.', linestyle='-', linewidth=3)

        ax.grid(True)
        if domain == 'time':
            xlabel = 'Time (sec)'
            ylabel = 'Response (g)'
        elif domain == 'freq':
            xlabel = 'Frequency (Hz)'
            ylabel = 'PSD ($g^2$/Hz)'
        elif domain == 'equation':
            xlabel = ''
            ylabel = ''
        else:
            raise RuntimeError(domain)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax
    #------------------------------------------------------------------------
    # Logging
    # basic interaction
    def on_show_debug(self) -> None:
        """sets a flag for showing/hiding DEBUG messages"""
        self.settings.show_debug = not self.settings.show_debug

    def on_show_info(self) -> None:
        """sets a flag for showing/hiding INFO messages"""
        self.settings.show_info = not self.settings.show_info

    def on_show_command(self) -> None:
        """sets a flag for showing/hiding COMMAND messages"""
        self.settings.show_command = not self.settings.show_command

    def on_show_warning(self) -> None:
        """sets a flag for showing/hiding WARNING messages"""
        self.settings.show_warning = not self.settings.show_warning

    def on_show_error(self) -> None:
        """sets a flag for showing/hiding ERROR messages"""
        self.settings.show_error = not self.settings.show_error

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

    #------------------------------------
    #minor windows
    def on_about(self) -> None:
        data = {'font_size': 8,}
        win = AboutWindow(data, win_parent=self, show_tol=True)
        win.show()
    def open_website(self) -> None:
        pass
    def _check_for_latest_version(self) -> None:
        pass

def set_tab(tab: QWidget) -> None:
    tab.setEnabled(True)
    tab.activateWindow()
    tab.setFocus()


import getpass
class Login(QGridLayout):
    def __init__(self):
        super().__init__()
        method_label = QLabel('Method:')
        method_pulldown = QComboBox()
        method_pulldown.addItems(['csv', 'nominal'])
        set_combo_box_text(method_pulldown, 'csv')

        csv_label = QLabel('CSV File:')
        csv_filename = QTextEdit()

        user_label = QLabel('User:')
        user_name = getpass.getuser()
        user = QTextEdit(user_name)
        #-------------------------
        irow = 0
        self.addWidget(method_label, irow, 0)
        self.addWidget(method_pulldown, irow, 1)
        irow += 1

        self.addWidget(csv_label, irow, 1)
        self.addWidget(csv_filename, irow, 2)
        irow += 1

        self.addWidget(user_label, irow, 1)
        self.addWidget(user, irow, 2)
        irow += 1


def main():
    import os
    dirname = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(dirname)

    # 2. Create an instance of QApplication
    app = QApplication([])
    box = Login()


    Window = MainWindow()

    # 4. Show your application's GUI
    #window.show()

    # 5. Run your application's event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
