#import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt

#import dynomite.core.time as dynatime
#from dynomite.core.load_utils import _update_label

#from dynomite.core.time import TimeSeries
#import dynomite.core.fourier_transform as ft
#import dynomite.core.vrs as vrs # VibrationResponseSpectra
from dynomite.core.psd import PowerSpectralDensity
from dynomite.plotting.utils import _set_grid


class VibrationResponseSpectra(PowerSpectralDensity):
    #def __init__(self, frequency: np.ndarray, vrs_response: np.ndarray, label: list[str],
                 #sided: int=1, is_onesided_center: bool=None, octave_spacing: int=0):
        #if vrs_response.ndim == 1:
            #psd_response = psd_response.reshape(len(psd_response), 1)
        #if 'complex' in psd_response.dtype.name:
            #raise TypeError(psd_response)
        #assert psd_response.shape[1] == 1, psd_response.shape
        #self.frequency = frequency
        #self.response = psd_response
        #self.label = _update_label(label)
        #self.octave_spacing = octave_spacing
        #self.sided = sided
        #self.is_onesided_center = is_onesided_center
        ##assert sided == 2
        #assert is_onesided_center is not None
        #assert sided in {1, 2}, sided
        #assert octave_spacing >= 0, octave_spacing
        #assert isinstance(frequency, np.ndarray), type(frequency)
        #assert isinstance(psd_response, np.ndarray), type(psd_response)
        #print('psd-init', self.fsampling, self.df, self.is_onesided_center)

    def plot(self, ifig: int=1,
             ax: Optional[plt.Axes]=None,
             y_units: str='g', xscale: str='log', yscale: str='log',
             xlim: Optional[tuple[float, float]]=None,
             ylim: Optional[tuple[float, float]]=None,
             linestyle='-o',
             show: bool=True):
        #self.fsampling
        #self.df
        if ax is None:
            fig = plt.figure(ifig)
            ax = fig.gca()
        ax.set_title('PSD')
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

