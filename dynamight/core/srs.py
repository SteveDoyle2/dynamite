from itertools import count
from typing import Optional
import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from dynamight.core.load_utils import _update_label, _response_squeeze
from dynamight.core.plot_utils import _set_grid, _adjust_axes_limit


class ShockResponseSpectra:
    def __init__(self, frequency: np.ndarray,
                 Q: float,
                 label: Optional[list[str]],
                 srs_min: Optional[np.ndarray]=None,
                 srs_max: Optional[np.ndarray]=None,
                 rel_disp_min: Optional[np.ndarray]=None,
                 rel_disp_max: Optional[np.ndarray]=None,
                 accel_scale: float=386.088,
                 displacement_units: str='in'):
        """
        acceleration has units of g's
        pseudo-displacement needs a scale factor of g to go from g*s^2 to in
        """
        #if isinstance(response, (tuple, list)):
            #response = np.array(response).T
        if srs_min is not None and srs_min.ndim == 1:
            srs_min = srs_min.reshape(len(srs_min), 1)
        if srs_max is not None and srs_max.ndim == 1:
            srs_max = srs_max.reshape(len(srs_max), 1)

        if rel_disp_min is not None and rel_disp_min.ndim == 1:
            rel_disp_min = rel_disp_min.reshape(len(rel_disp_min), 1) * accel_scale
        if rel_disp_max is not None and rel_disp_max.ndim == 1:
            rel_disp_max = rel_disp_max.reshape(len(rel_disp_max), 1) * accel_scale
        self.frequency = frequency
        self.srs_min = srs_min
        self.srs_max = srs_max
        self.rel_disp_min = rel_disp_min
        self.rel_disp_max = rel_disp_max
        self.Q = Q
        self.label = _update_label(label)
        self.displacement_units = displacement_units

    def plot_srs_accel(self, y_units: str='g',
                       ifig: int=0,
                       ax: Optional[plt.Axes]=None,
                       linestyle='-',
                       short_ylabel: bool=False,
                       show: bool=True):
        fig, ax = get_fig_ax(ax=ax, ifig=ifig)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(f'SRS Acceleration (g); Q={self.Q:g}')
        _plot(ax, self.frequency,
              self.srs_max, self.srs_min,
              linestyle, self.label, show=show)

    def plot_pseudo_displacement_srs(self, ifig: int=0,
                                     ax: Optional[plt.Axes]=None,
                                     linestyle='-',
                                     short_ylabel: bool=False,
                                     show: bool=True):
        fig, ax = get_fig_ax(ax=ax, ifig=ifig)
        if short_ylabel:
            ylabel = f'Pseudo-Displacement SRS ({self.displacement_units}); Q={self.Q:g}'
        else:
            ylabel = f'SD ({self.displacement_units}/$s^2$)'
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(ylabel)
        _plot(ax, self.frequency,
              self.rel_disp_max, self.rel_disp_min,
              linestyle, self.label, show=show)

    def plot_pseudo_velocity_srs(self, ifig: int=0,
                                 ax: Optional[plt.Axes]=None,
                                 linestyle='-',
                                 short_ylabel: bool=False,
                                 show: bool=True):
        fig, ax = get_fig_ax(ax=ax, ifig=ifig)
        if short_ylabel:
            ylabel = f'Pseudo-Velocity SRS ({self.displacement_units}/s); Q={self.Q:g}'
        else:
            ylabel = f'PSV ({self.displacement_units}/$s^2$)'

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(ylabel)
        omega = 2 * np.pi * self.frequency
        _plot(ax, self.frequency,
              self.rel_disp_max*omega,
              self.rel_disp_min*omega,
              linestyle, self.label, show=show)

    def plot_pseudo_accel_srs(self, ifig: int=0,
                              ax: Optional[plt.Axes]=None,
                              linestyle='-',
                              short_ylabel: bool=False,
                              show: bool=True):
        fig, ax = get_fig_ax(ax=ax, ifig=ifig)
        if short_ylabel:
            ylabel = f'Pseudo-Accel SRS ({self.displacement_units}/$s^2$); Q={self.Q:g}'
        else:
            ylabel = f'PSA ({self.displacement_units}/$s^2$)'

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(ylabel)
        omega2 = (2 * np.pi * self.frequency) ** 2
        _plot(ax, self.frequency,
              self.rel_disp_max*omega2,
              self.rel_disp_min*omega2,
              linestyle, self.label, show=show)

def get_fig_ax(ax: Optional[plt.Axes]=None,
               ifig: int=0,) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None and ifig == 0:
        fig = ax.get_figure()
    elif ax is not None and ifig > 0:
        fig = plt.figure(ifig)
        ax = fig.gca()
    elif ax is None and ifig > 0:
        fig = plt.figure(ifig)
        ax = fig.gca()
    elif ax is None and ifig == 0:
        fig = plt.figure()
        ax = fig.gca()
    else:
        raise RuntimeError((ax, ifig))
    return fig, ax

def _plot(ax: plt.Figure,
          frequency: np.ndarray,
          ymax: np.ndarray,
          ymin: np.ndarray,
          linestyle: str, label: list[str],
          show: bool=True) -> None:
    ax.loglog(frequency, ymax[:, 0], linestyle,
              label=label[0]+' Max')
    ax.loglog(frequency, ymin[:, 0], '--',
              label=label[0]+' Min')
    ax.legend()
    _set_grid(ax, 'log', 'log')
    if show:
        plt.show()

def octave_spacing(fmin: float,
                   fmax: float,
                   noctave: int=6) -> np.ndarray:
    #self.fn.append(f1)
    n = np.ceil(np.log2(fmax/fmin) * noctave)
    ns = np.arange(n)
    frequency = fmin * 2 ** (ns/noctave)
    #for j in range(1,999):
        #self.fn.append(self.fn[j-1]*(2.**octave))
        #if  self.fn[j] > self.sr/8.:
            #break
    return frequency

def time_to_srs(time: np.ndarray,
                response: np.ndarray,
                Q: float,
                fmin: float=1.0,
                fmax: float=1000.0,
                noctave: int=6,
                calc_accel_srs: bool=True,
                calc_rel_disp_srs: bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    https://www.vibrationdata.com/tutorials_alt/TumaSRS_dm09.pdf
    https://www.vibrationdata.com/tutorials_alt/Ahlin_SRS.pdf
    https://www.vibrationdata.com/tutorials2/srs_intr.pdf
    """
    damp = 1 / (2 * Q)
    dt = time[1] - time[0]

    fn = octave_spacing(fmin, fmax, noctave=noctave)
    nfreq = len(fn)

    omega = 2. * np.pi * fn
    omegad = omega * np.sqrt(1.-(damp**2))

    # ------------------------------------------------------
    # a coeff
    ac = np.ones((nfreq, 3), dtype='float64')
    #ac[:, 0] = 1
    E = np.exp(-damp*omega*dt)
    K = omegad * dt
    C = E*np.cos(K)
    ac[:, 1] = -2*C
    ac[:, 2] = E**2

    # ------------------------------------------------------
    S = E*np.sin(K)
    if calc_accel_srs:
        # accel SRS
        Sp = S/K

        bc_accel = np.zeros((nfreq, 3), dtype='float64')
        bc_accel[:, 0] = 1. - Sp
        bc_accel[:, 1] = 2. * (Sp-C)
        bc_accel[:, 2] = E**2 - Sp

    # ------------------------------------------------------
    if calc_rel_disp_srs:
        # rel_disp_SRS
        E2 = np.exp(-2*damp*omega*dt)

        Omr = omega / omegad
        Omt = omega * dt
        P = 2*damp**2 - 1

        b00 = 2*damp*(C-1)
        b01 = S*Omr*P
        b02 = Omt

        b10 = -2*Omt*C
        b11 = 2*damp*(1-E2)
        b12 = -2*b01

        b20 = (2*damp+Omt)*E2
        b21 = b01
        b22 = -2*damp*C

        bi = -omega**3 * dt
        b0 = b00 + b01 + b02
        b1 = b10 + b11 + b12
        b2 = b20 + b21 + b22
        bc_disp = np.column_stack([b0, b1, b2]) / bi[:, np.newaxis]

    #---------------------------------------------------------
    nresponse = response.shape[1]
    accel_pos = np.zeros((nfreq, nresponse), dtype='float64')
    accel_neg = np.zeros((nfreq, nresponse), dtype='float64')

    rel_disp_pos = np.zeros((nfreq, nresponse), dtype='float64')
    rel_disp_neg = np.zeros((nfreq, nresponse), dtype='float64')

    if calc_accel_srs:
        plot_curves = False
        if plot_curves:
            fig = plt.figure()
            ax = fig.gca()
            #ax.plot(time, response, label='input')

        for ifreq, bci, aci in zip(count(), bc_accel, ac):
            for iresp in range(nresponse):
                responsei = response[:, iresp]
                resp = lfilter(bci, aci, responsei, axis=-1, zi=None)
                accel_pos[ifreq, iresp] = max(resp)
                accel_neg[ifreq, iresp] = abs(min(resp))
                if plot_curves:
                    ax.plot(time, resp, label=f'output (fn={fn[ifreq]:g}')
        if plot_curves:
            ax.grid(which='both')
            ax.legend()

    if calc_rel_disp_srs:
        plot_curves = False
        if plot_curves:
            fig = plt.figure()
            ax = fig.gca()
            #ax.plot(time, response, label='input')

        for ifreq, bcdi, aci in zip(count(), bc_disp, ac):
            for iresp in range(nresponse):
                responsei = response[:, iresp]
                resp = lfilter(bcdi, aci, responsei, axis=-1, zi=None)
                rel_disp_pos[ifreq, iresp] = max(resp)
                rel_disp_neg[ifreq, iresp] = abs(min(resp))
                if plot_curves:
                    ax.plot(time, resp, label=f'output (fn={fn[ifreq]:g}')

        if plot_curves:
            ax.grid(which='both')
            ax.legend()
            plt.show()

    frequency = fn
    return frequency, accel_neg, accel_pos, rel_disp_neg, rel_disp_pos

def half_sine_pulse(ymax: float,
                    tmax: float,
                    tpulse: float,
                    ntimes: 101) -> np.ndarray:
    assert tpulse < tmax
    pulse_freq = 1 / tpulse
    time = np.linspace(0., tmax, num=ntimes)
    response = np.sin(np.pi*pulse_freq*time) * ymax
    response[time > tpulse] = 0.
    return time, response

