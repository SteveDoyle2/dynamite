from __future__ import annotations
from typing import Optional # , Union
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from dynamight.core.load_utils import _update_label, _response_squeeze
import dynamight.core.fourier_transform as ft
#import dynamight.core.time as time
import dynamight.core.psd as dypsd # PowerSpectralDensity
from dynamight.core.freq_utils import fft_to_psd_df
from dynamight.core.srs import ShockResponseSpectra, time_to_srs


class TimeSeries:
    def __init__(self, time: np.ndarray,
                 time_response: np.ndarray,
                 label: list[str]=None):
        if time_response.ndim == 1:
            time_response = time_response.reshape(len(time_response), 1)
        if 'complex' in time_response.dtype.name:
            raise TypeError(time_response)

        assert isinstance(time, np.ndarray), time
        self.time = time

        assert isinstance(time_response, np.ndarray), time_response
        self.response = time_response

        self.label = _update_label(label)

    @property
    def dt(self) -> float:
        return self.time[1] - self.time[0]

    @property
    def tmax(self) -> float:
        return self.time[-1]

    def to_fft(self, sided: int=1, fft_type: str='real_imag') -> ft.FourierTransform:
        tmax = self.tmax
        dts = np.diff(self.time)
        dt = dts.mean()
        ntimes = len(self.time)

        is_onesided_center = (ntimes % 2 == 1)
        frequency = _onesided_frequency(dt, tmax, ntimes)
        #df = 1 / tmax
        #fsampling = 1 / dt
        #fnyquist = fsampling / 2
        #frequency = np.arange(0, ntimes) * df

        response = _response_squeeze(self.response)
        fft_response = sp.fft.fft(response, n=None, axis=-1, norm=None,
                         overwrite_x=False, workers=None, plan=None)
        fft_response /= ntimes

        if fft_type == 'mag_phase':
            mag = np.abs(fft_response)
            phase = np.arctan2(fft_response.imag, fft_response.real)
            fft_response = mag + 1j * phase

        assert fft_response.shape[0] == ntimes, (fft_response.shape, ntimes)
        #assert sided == 1
        fft = ft.FourierTransform(
            frequency, fft_response, label=self.label, fft_type=fft_type,
            sided=sided, is_onesided_center=is_onesided_center)
        return fft

    def to_psd_welch(self, sided: int=1, window: str='hann',
                     window_size_sec: float= 1.0,
                     overlap_sec: float=0.5) -> dypsd.PowerSpectralDensity:
        assert sided in {1, 2}, sided
        return_onesided = (sided == 1)
        #fsampling = 1 / self.dt

        #ntimes = len(self.time)
        #df = 1 / self.tmax
        fsampling = 1 / self.dt
        #fnyquist = fsampling / 2
        #frequency = np.arange(0, ntimes) * df

        window_size_int = int(fsampling * window_size_sec)
        overlap_int = int(fsampling * overlap_sec)

        #nfft - for 0 padded signals
        #ntimes = len(self.time)
        #is_onesided_center = (ntimes % 2 == 1)
        response = _response_squeeze(self.response)
        frequency, psd_response = sp.signal.welch(
            response, fs=fsampling, window=window,
            nperseg=window_size_int, noverlap=overlap_int, nfft=None,
            detrend='constant', return_onesided=return_onesided,
            scaling='density', axis=-1, average='mean')

        fmax = frequency[-1]
        if sided == 1:
            is_onesided_center = not np.allclose(fmax, fsampling)
        else:
            # doesn't matter
            is_onesided_center = True
        psd = dypsd.PowerSpectralDensity(
            frequency, psd_response, label=self.label,
            sided=sided, is_onesided_center=is_onesided_center,
            octave_spacing=0)
        return psd

    def to_psd(self, sided: int=1) -> dypsd.PowerSpectralDensity:
        assert sided in {1, 2}, sided
        tmax = self.time[-1]

        dt = self.dt
        #dts = np.diff(self.time)
        #dt = dts.mean()
        ntimes = len(self.time)

        frequency = _onesided_frequency(dt, tmax, ntimes)
        response = _response_squeeze(self.response)
        fft_response = sp.fft.fft(response, n=None, axis=-1, norm=None,
                                  overwrite_x=False, workers=None, plan=None)
        fft_response /= ntimes

        #psd_response = fft_response * np.conj(fft_response) / df
        psd_response = fft_to_psd_df(frequency, fft_response)
        assert fft_response.shape[0] == ntimes, (fft_response.shape, ntimes)
        is_onesided_center = (ntimes % 2 == 1)
        #assert sided == 1
        psd = dypsd.PowerSpectralDensity(
            frequency, psd_response, label=self.label,
            sided=sided, is_onesided_center=is_onesided_center)
        return psd

    def to_srs(self, Q: float=10.) -> srs.ShockResponseSpectra:
        freq, srs_min, srs_max = time_to_srs(self.time, self.response, Q, fmax=10_000)
        shock = ShockResponseSpectra(freq, srs_min, srs_max, Q, label=self.label)
        return shock

    def plot(self, y_units: str='g', ifig: int=1,
             ax: Optional[plt.Axes]=None,
             linestyle='-o',
             show: bool=True):
        if ax is None:
            fig = plt.figure(ifig)
            ax = fig.gca()
        ax.set_title('Time Series')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Response (g)')
        ax.plot(self.time, self.response[:, 0], linestyle, label=self.label[0])
        ax.legend()
        ax.grid(True)
        if show:
            plt.show()

def _onesided_frequency(dt: float, tmax: float, ntimes: int) -> np.ndarray:
    df = 1 / tmax
    #fsampling = 1 / dt
    #fnyquist = fsampling / 2
    frequency = np.arange(0, ntimes) * df
    return frequency
