from typing import Optional
import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt

class ShockResponseSpectra:
    def __init__(self, frequency: np.ndarray, response: np.ndarray,
                 Q: float, label: Optional[list[str]]):
        self.frequency = frequency
        self.response = response
        self.Q = Q
        self.label =

    def plot(self, y_units: str='g', ifig: int=1,
             ax: Optional[plt.Axes]=None,
             linestyle='-',
             show: bool=True):
        if ax is None:
            fig = plt.figure(ifig)
            ax = fig.gca()
        ax.set_title('Shock Response Spectra')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(f'SRS Acceleration (g); Q={self.Q}')
        ax.plot(self.frequency, self.response[:, 0], linestyle, label=self.label[0])
        ax.legend()
        ax.grid(True)
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

    A = wn * T / damping
    B = wn * T * np.sqrt(1 - 4/Q**2)

    b0 = 1 - np.exp(-A) * np.sin(B)/B
    b1 = 2*np.exp(-A) * (np.sin(B)/B - np.cos(B))
    b2 = np.exp(-2*A) - np.exp(-A) * np.sin(B)/B
    a1 = -2*np.exp(-A) * np.cos(B)
    a2 = np.exp(-2*A)

    BB = np.column_stack([b0, b1, b2])
    AA = np.column_stack([-a1, -a2])

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
