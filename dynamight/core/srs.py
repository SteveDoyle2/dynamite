from itertools import count
from typing import Optional
import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from dynamight.core.load_utils import _update_label, _response_squeeze
from dynamight.core.plot_utils import _set_grid, _adjust_axes_limit


class ShockResponseSpectra:
    def __init__(self, frequency: np.ndarray,
                 srs_min: np.ndarray,
                 srs_max: np.ndarray,
                 Q: float, label: Optional[list[str]]):
        #if isinstance(response, (tuple, list)):
            #response = np.array(response).T
        if srs_min.ndim == 1:
            srs_min = srs_min.reshape(len(srs_min), 1)
        if srs_max.ndim == 1:
            srs_max = srs_max.reshape(len(srs_max), 1)
        self.frequency = frequency
        self.srs_min = srs_min
        self.srs_max = srs_max
        self.Q = Q
        self.label = _update_label(label)

    def plot(self, y_units: str='g', ifig: int=1,
             ax: Optional[plt.Axes]=None,
             linestyle='-',
             show: bool=True):
        if ax is None:
            fig = plt.figure(ifig)
            ax = fig.gca()
        ax.set_title('Shock Response Spectra')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(f'SRS Acceleration (g); Q={self.Q:g}')
        ax.loglog(self.frequency, self.srs_max[:, 0], linestyle,
                  label=self.label[0]+' Max')
        ax.loglog(self.frequency, self.srs_min[:, 0], '--',
                  label=self.label[0]+' Min')
        ax.legend()
        #ax.grid(True)
        _set_grid(ax, 'log', 'log')
        if show:
            plt.show()


def time_to_srs(time: np.ndarray,
                response: np.ndarray,
                Q: float=10.) -> tuple[np.ndarray, np.ndarray]:
    """
    https://www.vibrationdata.com/tutorials_alt/TumaSRS_dm09.pdf
    https://www.vibrationdata.com/tutorials2/srs_intr.pdf
    """
    #omegan = 10
    fmin = 1.
    fmax = 1000.
    n = 90
    qv = (fmax/fmin) ** (1/n)

    ns = np.arange(n)
    fn = fmin * qv ** ns
    wn = 2 * np.pi * fn
    T = time[1] - time[0]
    damping = 1 / (2 * Q)
    zeta = 1 / (2 * Q)

    if 0:
        A = wn * T / damping
        B = wn * T * np.sqrt(1 - 4/Q**2)

        b0 = 1 - np.exp(-A) * np.sin(B)/B
        b1 = 2*np.exp(-A) * (np.sin(B)/B - np.cos(B))
        b2 = np.exp(-2*A) - np.exp(-A) * np.sin(B)/B
        a1 = -2*np.exp(-A) * np.cos(B)
        a2 = np.exp(-2*A)
        BB = np.column_stack([b0, b1, b2])
        AA = np.column_stack([-a1, -a2])
    else:
        A = 2 * np.exp(-zeta*wn*dt) * np.cos(wd*dt)
        B = -np.exp(-2*zeta*wn*dt)
        C = beta * dt
        D = dt * np.exp(-2*zeta*omegan*dt)
        E = 0

    srs_data_list = []
    for i in np.arange(n):
        #fn=fmin * qv ** i
        #ff = [ff, fn]
        #wn = 2*np.pi*fn
        #A = wn * T/2/Q
        #B = wn * T * np.sqrt(1-1/4/Q/Q)
        #b0 = 1 - np.exp(-A) * np.sin(B)/B
        #b1 = 2*np.exp(-A) * (np.sin(B)/B - np.cos(B))
        #b2 = np.exp((-2)*A) - np.exp(-A) * np.sin(B)/B
        #a1 = (-2)*np.exp((-1)*A) * np.cos(B)
        #a2 = np.exp((-2)*A)
        #BB = [b0, b1, b2]
        #AA = [-a1,-a2]
        #y = lfilter(input1, BB, AA)
        y = lfilter(BB[i], AA[i], response)
        yy = max(y)
        #ymax = [ymax, yy]
        srs_data_list.append(yy)
    frequency = fn
    srs_data = np.array(srs_data_list, dtype='float64')
    return frequency, srs_data

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

def a_coeff(omega: np.ndarray, damp: float, dt: float):
    nfreq = len(omega)
    ac = np.ones((nfreq, 3), dtype='float64')

    #ac[:, 0] = 1
    omegad = omega*np.sqrt(1.-(damp**2))
    E = np.exp(-damp*omega*dt)
    K = omegad*dt
    C = E*np.cos(K)
    ac[:, 1] = -2*C
    ac[:, 2] = E**2
    return ac

def time_to_srs(time: np.ndarray,
                response: np.ndarray,
                Q: float,
                fmin: float=1.0,
                fmax: float=1000.0,
                noctave: int=3) -> tuple[np.ndarray, np.ndarray]:
    """
    https://www.vibrationdata.com/tutorials_alt/Ahlin_SRS.pdf
    """
    damp = 1 / (2 * Q)
    dt = time[1] - time[0]

    fn = octave_spacing(fmin, fmax, noctave=noctave)
    nfreq = len(fn)

    omega = 2. * np.pi * fn
    omegad = omega * np.sqrt(1.-(damp**2))

    # ------------------------------------------------------
    # accel SRS

    #  bc coefficients are applied to the excitation
    Q2 = Q ** 2
    A = omega * dt / (2 * Q)
    B = omega * dt * np.sqrt(1 - 1 / (4*Q2))
    #C = (2 * Q2 - 1) / np.sqrt(4*Q2-1)
    ema = np.exp(-A)
    em2a = np.exp(-2*A)
    b0 = 1 - ema * np.sin(B) / B
    b1 = 2 * ema * (np.sin(B) / B - np.cos(B))
    b2 = em2a - ema * np.sin(B) / B
    a1 = -2 * ema * np.cos(B)
    a2 = em2a
    #b0 = 2 * A
    #b1 = 2 * A * ema * (C * np.sin(B) - np.cos(B))
    #a1 = -2 * ema * np.cos(B)
    #a2 = np.exp(-2*A)

    E = np.exp(-damp*omega*dt)
    K = omegad*dt
    C = E*np.cos(K)
    S = E*np.sin(K)
    Sp = S/K

    bc_accel = np.zeros((nfreq, 3), dtype='float64')
    bc_accel[:, 0] = 1. - Sp
    bc_accel[:, 1] = 2. * (Sp-C)
    bc_accel[:, 2] = E**2 - Sp
    #bc = bc_accel # .flatten()
    ac = a_coeff(omega, damp, dt) # .flatten()

    b = np.array([b0, b1, b2]).flatten()
    a = np.array([np.ones(nfreq), a1, a2]).flatten()
    x = 1

    # ------------------------------------------------------
    if 0:
        # rel_disp_SRS

        E1 = np.exp(  -damp*omega*dt)
        E2 = np.exp(-2*damp*omega*dt)

        K = omegad*dt
        C = E1 * np.cos(K)
        S = E1 * np.sin(K)
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

        b0 = b00 + b01 + b02
        b1 = b10 + b11 + b12
        b2 = b20 + b21 + b22
        b_disp = np.column_stack([b0, b1, b2])

    #---------------------------------------------------------
    ac = a_coeff(omega, damp, dt)

    nresponse = response.shape[1]
    rd_pos = np.zeros((nfreq, nresponse), dtype='float64')
    rd_neg = np.zeros((nfreq, nresponse), dtype='float64')

    for ifreq, bci, aci in zip(count(), bc_accel, ac):
        #bci = -bc / (omega**3 * dt)  # rel-disp
        #ac = a_coeff(omega, damp, dt)
        for iresp in range(nresponse):
            responsei = response[:, iresp]
            resp = lfilter(bci, aci, responsei, axis=-1, zi=None)
            rd_pos[ifreq, iresp] = max(resp)
            rd_neg[ifreq, iresp] = abs(min(resp))

        if 0:
            #fn = 10
            #bc = array([ 1.10753612e-04,  3.21565544e-06, -1.09133695e-04])
            #ac = array([ 1.        , -1.99977528,  0.99978011])
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(time, response, label='input')
            ax.plot(time, resp, label='output')
            ax.grid()
            ax.legend()
            plt.show()

    frequency = fn
    return frequency, rd_pos, rd_neg

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

