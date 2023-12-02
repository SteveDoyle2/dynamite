#from dynomite.core.time_series import
from typing import Optional
import numpy as np
import scipy as sp
#import pandas as pd
import matplotlib.pyplot as plt
from dynomite.core.load_utils import _update_label, _response_squeeze
import dynomite.core.time as dytime
import dynomite.core.psd as dypsd # PowerSpectralDensity
from dynomite.core.freq_utils import _to_twosided_fsampling, fft_to_psd_df, dft_to_onesided
from dynomite.core.plot_utils import _set_grid, _adjust_axes_limit


class FourierTransform:
    def __init__(self, frequency: np.ndarray, fft_response: np.ndarray, label: list[str],
                 fft_type: str='real_imag', sided: int=1, is_onesided_center: bool=None):
        if fft_response.ndim == 1:
            fft_response = fft_response.reshape(len(fft_response), 1)
        self.frequency = frequency
        self.response = fft_response
        self.label = _update_label(label)
        self.fft_type = fft_type
        self.sided = sided
        self.is_onesided_center = is_onesided_center
        assert is_onesided_center is not None
        assert fft_type in {'real_imag', 'mag_phase'}, fft_type

    @property
    def df(self) -> float:
        df = self.frequency[1] - self.frequency[0]
        return df
    @property
    def fsampling(self) -> float:
        fmax = self.frequency[-1]
        fsampling = _to_twosided_fsampling(fmax, self.df, self.sided, self.is_onesided_center)
        return fsampling

    def to_time(self):
        assert self.sided == 2, self.sided
        #fnyq = self.frequency[-1]
        #fsampling = fnyq * 2
        tmax = 1 / self.df
        dt = 1 / self.fsampling
        ntimes = np.ceil(tmax / dt)

        response = _response_squeeze(self.response) * ntimes
        ifft = sp.fft.ifft(response, n=None, axis=-1, norm=None,
                           overwrite_x=False, workers=None, plan=None)
        time = np.arange(0, ntimes) * dt
        time_series = dytime.TimeSeries(time, ifft.real, label=self.label)
        return time_series

    def to_psd(self, sided: int=1) -> dypsd.PowerSpectralDensity:
        psd_response = fft_to_psd_df(self.frequency, self.response)
        #toonesided
        psd = dypsd.PowerSpectralDensity(
            self.frequency.copy(), psd_response, label=self.label,
            sided=sided, is_onesided_center=self.is_onesided_center)
        return psd

    def to_onesided(self, inplace: bool=True):
        if self.sided == 1:
            return self
        assert self.sided == 2, self.sided
        frequency, response, is_onesided_center = dft_to_onesided(self.frequency, self.response)

        self.sided = 1
        self.is_onesided_center = is_onesided_center
        return self

    def mag_phase(self) -> tuple[np.ndarray, np.ndarray]:
        response = self.response[:, 0]
        if self.fft_type == 'mag_phase':
            # e * (i theta) = cos(theta) + 1j*sin(theta)
            mag = response.real
            phase = response.imag
        else:
            mag = np.abs(response)
            phase = np.arctan2(response.imag, response.real)
        return mag, phase

    def real_imag(self) -> np.ndarray:
        response = self.response[:, 0]
        if self.fft_type == 'mag_phase':
            # e * (i theta) = cos(theta) + 1j*sin(theta)
            mag = response.real
            phase = response.imag
            real_imag = mag * (np.cos(phase) + 1j * np.sin(phase))
        else:
            real_imag = self.response
        return real_imag

    def plot_real_imag(self, ifig: int=1,
                       ax: Optional[tuple[plt.Axes, plt.Axes]]=None,
                       xscale: str='linear', show: bool=True):
        if ax is None:
            fig = plt.figure(ifig)
            ax1, ax2 = fig.subplots(nrows=2)
        else:
            ax1, ax2 = ax

        #ax.set_yscale(yscale)
        ax1.set_title('Fourier Transform - Real/Imag')
        ax1.set_xlabel('Frequency (Hz)')
        ax2.set_xlabel('Frequency (Hz)')

        ax1.set_ylabel('Real (g)')
        ax2.set_xlabel('Imag (g)')

        real_imag = self.real_imag()
        real = real_imag.real
        imag = real_imag.imag

        ax1.plot(self.frequency, real, '-o', label=self.label[0])
        ax2.plot(self.frequency, imag, '-o', label=self.label[0])
        ax2.legend()
        _adjust_axes_limit(ax2, imag)
        _set_grid(ax1, xscale, 'linear')
        _set_grid(ax2, xscale, 'linear')
        if show:
            plt.show()

    def plot_mag(self, ifig: int=1,
                 ax: Optional[plt.Axes]=None,
                 y_units: str='g',
                 xscale: str='linear',
                 yscale: str='linear',
                 linestyle: str='-o',
                 show: bool=True):
        if ax is None:
            fig = plt.figure(ifig)
            ax = fig.gca()
        #ax1.set_title('Fourier Transform - Mag')
        ax.set_xlabel('Frequency (Hz)')

        mag, phase = self.mag_phase()
        ax.plot(self.frequency, mag, linestyle, label=self.label[0])
        ax.legend()
        ax.set_ylabel(f'Magnitude ({y_units})')
        #ax.set_yticklabels(['{:,.0%}'.format(x) for x in current_values])
        _set_grid(ax, xscale, yscale)
        if show:
            plt.show()

    def plot_mag_phase(self, ifig: int=1,
                       ax: Optional[tuple[plt.Axes, plt.Axes]]=None,
                       xscale: str='linear', show: bool=True):
        if ax is None:
            fig = plt.figure(ifig)
            ax1, ax2 = fig.subplots(nrows=2)
        else:
            ax1, ax2 = ax
        ax1.set_title('Fourier Transform - Mag/Phase')
        ax1.set_xlabel('Frequency (Hz)')
        ax2.set_xlabel('Frequency (Hz)')

        mag, phase = self.mag_phase()
        ax1.plot(self.frequency, mag, '-o')
        ax2.plot(self.frequency, phase, '-o', label=self.label[0])
        ax2.legend()
        ax1.set_ylabel('Magnitude (g)')
        ax2.set_ylabel('Phase (rad)')
        _adjust_axes_limit(ax2, phase)

        _set_grid(ax1, xscale, 'linear') # yscale
        _set_grid(ax2, xscale, 'linear') # yscale
        if show:
            plt.show()


