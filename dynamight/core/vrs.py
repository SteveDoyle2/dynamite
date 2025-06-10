from typing import Optional

import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt

#import dynamight.core.time as dynatime
from dynamight.typing import Limit
from dynamight.core.load_utils import _update_label, _response_squeeze

#from dynamight.core.time import TimeSeries
#import dynamight.core.fourier_transform as ft
#import dynamight.core.vrs as vrs # VibrationResponseSpectra
#import dynamight.core.psd as dypsd
from dynamight.core.freq_utils import _to_twosided_fsampling
from dynamight.core.plot_utils import _set_grid


class VibrationResponseSpectra:
    """A VRS """
    def __init__(self, frequency: np.ndarray, vrs_response: np.ndarray, label: list[str],
                 sided: int=1, is_onesided_center: bool=None, octave_spacing: int=0):
        if vrs_response.ndim == 1:
            vrs_response = vrs_response.reshape(len(vrs_response), 1)
        if 'complex' in vrs_response.dtype.name:
            raise TypeError(vrs_response)
        assert vrs_response.shape[1] == 1, vrs_response.shape
        self.frequency = frequency
        self.response = vrs_response
        self.label = _update_label(label)
        self.octave_spacing = octave_spacing
        self.sided = sided
        self.is_onesided_center = is_onesided_center
        #assert sided == 2
        assert is_onesided_center is not None
        assert sided in {1, 2}, sided
        assert octave_spacing >= 0, octave_spacing
        assert isinstance(frequency, np.ndarray), type(frequency)
        assert isinstance(vrs_response, np.ndarray), type(vrs_response)
        print('vrs-init', self.fsampling, self.df, self.is_onesided_center)

    @property
    def df(self):
        if self.octave_spacing == 0:
            return self.frequency[1] - self.frequency[0]
        raise RuntimeError(self.octave_spacing)

    @property
    def fsampling(self) -> float:
        assert self.sided in {1, 2}
        if self.octave_spacing == 0:
            fmax = self.frequency[-1]
            df = self.df
            fsampling = _to_twosided_fsampling(
                fmax, df, sided=self.sided,
                is_onesided_center=self.is_onesided_center)
            return fsampling
        raise RuntimeError(self.octave_spacing)

    def plot(self, ifig: int=1,
             ax: Optional[plt.Axes]=None,
             y_units: str='g', xscale: str='log', yscale: str='log',
             xlim: Optional[tuple[float, float]]=None,
             ylim: Optional[tuple[float, float]]=None,
             linestyle: str='-',
             show: bool=True) -> tuple[plt.Figure, plt.Axes]:
        #self.fsampling
        #self.df
        if ax is None:
            fig = plt.figure(ifig)
            ax = fig.gca()
        ax.set_title('VRS PSD')
        ax.set_xlabel('Frequency (Hz)')
        assert self.octave_spacing == 0, self.octave_spacing
        ax.set_ylabel(f'VRS (${y_units}^2$/Hz)')
        ax.plot(self.frequency, self.response[:, 0], linestyle, label=self.label[0])
        ax.legend()
        _set_grid(ax, xscale, yscale)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if show:
            plt.show()
        return fig, ax

